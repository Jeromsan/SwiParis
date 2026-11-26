from contextlib import closing, redirect_stdout
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from swiparis.catalog import Catalog
from swiparis.logging_store import Conflict, LogStore
from swiparis.prepare_data import prepare, sha256
from swiparis.smoke import check
from swiparis.taxonomy import load_taxonomy
from test_paris_data import event
from swiparis.import_paris_data import run
import argparse

ROOT = Path(__file__).resolve().parents[1]


class TeamDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.fs, cls.pd = cls.root / "fs.sqlite", cls.root / "pd.sqlite"
        with closing(sqlite3.connect(cls.fs)) as db:
            db.executescript("""
                CREATE TABLE venues(venue_id TEXT PRIMARY KEY,category TEXT,latitude REAL,longitude REAL,country TEXT);
                CREATE TABLE checkins(user_id TEXT,venue_id TEXT,utc_time TEXT,timezone_offset INTEGER);
                INSERT INTO venues VALUES ('r','French Restaurant',48.85,2.35,'FR'),('m','Museum',48.85,2.35,'FR');
            """)
            db.executemany("INSERT INTO checkins VALUES (?,?,?,?)", [("u", "r" if n < 6 else "m", f"2012-04-03T18:00:{n:02d}Z", 120) for n in range(10)])
            db.commit()
        source = cls.root / "events.json"
        rows = [event(str(i), title=f"Activité {i}") for i in range(1, 9)]
        rows[-2]["price_type"] = "payant"
        rows[-1]["price_type"] = "gratuit sous condition"
        source.write_text(json.dumps(rows), encoding="utf-8")
        args = argparse.Namespace(source=source, output=cls.pd, report=cls.root / "pd-report.json",
                                  category_map=ROOT / "config/common_taxonomy.json", as_of="2026-11-18T12:00:00Z", days=30)
        with redirect_stdout(io.StringIO()):
            run(args)
            cls.common = cls.root / "common.sqlite"
            cls.report = prepare(cls.fs, cls.pd, args.category_map, cls.common, cls.root / "report.json")
        cls.catalog = Catalog(cls.common)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.logs_temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.logs_temp.cleanup)
        self.store = LogStore(Path(self.logs_temp.name) / "logs.sqlite", self.catalog)
        self.ids = [r["item_id"] for r in self.catalog.list_items(limit=20)["items"]]

    def session(self, **changes):
        data = dict(event_id="session", consent=True, constraints={}, catalog_version=self.catalog.version,
                    model_version="unit-test", data_origin="test")
        data.update(changes)
        return self.store.create_session(data)["session_id"]

    def test_taxonomy_and_report(self):
        tax = load_taxonomy(ROOT / "config/common_taxonomy.json")
        allowed = json.loads((ROOT / "config/foursquare_leisure_categories.json").read_text(encoding="utf-8"))
        self.assertEqual(set(tax["foursquare"]), set(allowed))
        self.assertEqual(len(tax["categories"]), 10)
        counts = {r["category"]: r for r in self.report["category_distribution"]}
        self.assertEqual(counts["restauration"]["visits"], 6)
        self.assertEqual(counts["culture"]["visits"], 4)
        self.assertEqual(sum(r["visits"] for r in counts.values()), 10)
        self.assertAlmostEqual(sum(r["paris_data_weighted_events"] for r in counts.values()), 8)
        self.assertIn("atelier", self.report["zero_training_categories"])
        with closing(sqlite3.connect(self.common)) as db:
            self.assertEqual(db.execute("SELECT min(user_id) FROM visits").fetchone()[0], "foursquare:u")
            self.assertFalse(db.execute("PRAGMA foreign_key_check").fetchall())
        self.assertEqual(sha256(self.fs), self.report["manifest"]["foursquare_sha256"])
        self.assertEqual(sha256(self.pd), self.report["manifest"]["paris_data_sha256"])

    def test_budget_and_slot_eligibility(self):
        free = self.catalog.list_items({"budget_eur": 0}, limit=None)["items"]
        self.assertEqual(len(free), 6)
        self.assertTrue(all(e["price_type"] == "free" for e in free))
        self.assertEqual(self.catalog.list_items({"start_utc": "2030-01-01T00:00:00Z", "end_utc": "2030-01-02T00:00:00Z"})["total"], 0)
        with self.assertRaises(ValueError):
            self.catalog.list_items({"budget_eur": float("nan")})
        with self.assertRaises(ValueError):
            self.catalog.require_eligible(["foursquare:r"], {})

    def test_consent_version_isolation_and_random_order(self):
        with self.assertRaises(ValueError):
            self.session(consent=False)
        with self.assertRaises(Conflict):
            self.session(catalog_version="wrong")
        with self.assertRaises(ValueError):
            LogStore(self.common, self.catalog)
        with self.assertRaises(ValueError):
            LogStore(self.fs, self.catalog)
        self.assertEqual(sha256(self.fs), self.report["manifest"]["foursquare_sha256"])
        sid = self.session()
        with patch("swiparis.logging_store.secrets.randbelow", return_value=1):
            pair = self.store.offer_comparison(dict(event_id="pair", session_id=sid, item_a=self.ids[0], item_b=self.ids[1]))
        self.assertEqual(pair["left_item_id"], self.ids[1])
        repeat = self.store.offer_comparison(dict(event_id="pair", session_id=sid, item_a=self.ids[0], item_b=self.ids[1]))
        self.assertEqual(repeat["left_item_id"], pair["left_item_id"])
        self.assertTrue(repeat["replayed"])

    def test_skip_and_request_conflicts(self):
        sid = self.session()
        pair = self.store.offer_comparison(dict(event_id="pair", session_id=sid, item_a=self.ids[0], item_b=self.ids[1]))
        answer = dict(event_id="skip", session_id=sid, comparison_id=pair["comparison_id"], choice="skip")
        self.assertIsNone(self.store.answer(answer)["chosen_item_id"])
        self.assertTrue(self.store.answer(answer)["replayed"])
        with self.assertRaises(Conflict):
            self.store.answer(dict(answer, event_id="another-answer", choice="left"))
        with self.assertRaises(Conflict):
            self.store.answer(dict(answer, choice="right"))
        pair = self.store.offer_comparison(dict(event_id="pair-2", session_id=sid,
                                                item_a=self.ids[2], item_b=self.ids[3]))
        selected = self.store.answer(dict(event_id="answer-2", session_id=sid,
                                          comparison_id=pair["comparison_id"], choice="left"))
        self.assertEqual(selected["chosen_item_id"], pair["left_item_id"])
        shown = self.store.show_recommendations(dict(event_id="selected-is-recommended", session_id=sid,
                                                     item_ids=[pair["left_item_id"]]))
        self.assertEqual(shown["impressions"][0]["item_id"], pair["left_item_id"])
        with self.assertRaises(ValueError):
            self.store.show_recommendations(dict(event_id="rejected-is-not-recommended", session_id=sid,
                                                 item_ids=[pair["right_item_id"]]))
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM comparisons WHERE choice='skip'").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM recommendations_shown").fetchone()[0], 1)

    def test_impression_actions_session_ownership(self):
        sid = self.session()
        shown = self.store.show_recommendations(dict(event_id="shown", session_id=sid, item_ids=self.ids[:2]))
        impression = shown["impressions"][0]["impression_id"]
        action = dict(event_id="open", session_id=sid, impression_id=impression, action="open")
        self.store.item_action(action)
        self.assertTrue(self.store.item_action(action)["replayed"])
        other = self.session(event_id="other-session")
        with self.assertRaises(ValueError):
            self.store.item_action(dict(action, event_id="foreign", session_id=other))
        with self.assertRaises(ValueError):
            self.store.item_action(dict(action, event_id="unclicked", action="negative"))

    def test_installed_style_http_workflow(self):
        result = check(self.common)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["pilot_database_modified"])


if __name__ == "__main__":
    unittest.main()
