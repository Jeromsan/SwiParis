import argparse
from contextlib import closing, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from swiparis.import_paris_data import parse_date, prepare, run, text

AS_OF = "2026-11-18T12:00:00Z"
END = "2026-12-18T12:00:00Z"
MAP = {"Expo": "exposition", "Atelier": "atelier"}


def event(identity="1", **changes):
    row = dict(id=identity, event_id=100 + int(identity), title="Une <b>exposition</b>",
               url=f"https://www.paris.fr/evenements/exposition-{identity}",
               description="<p>Peinture &amp; photo</p><script>bad()</script><p>Visite.</p>",
               date_start="2026-11-19T14:00:00+01:00", date_end="2026-11-19T16:00:00+01:00",
               occurrences="2026-11-19T14:00:00+01:00_2026-11-19T16:00:00+01:00",
               locations=[dict(address_name="Galerie", address_street="1 rue Exemple", address_city="Paris",
                               address_zipCode="75004", address_lat_lon="48.856,2.35", location_type="address")],
               qfap_tags="Expo;Atelier;Expo", price_type="gratuit", updated_at="2026-11-17T10:00:00+01:00")
    row.update(changes)
    return row


class ParisDataTest(unittest.TestCase):
    def prepare(self, rows):
        return prepare(rows, MAP, parse_date(AS_OF), parse_date(END))

    def test_text_and_no_naive_dates(self):
        self.assertEqual(text("<p>Café&nbsp;&amp; art</p><style>hidden</style><p>Bonjour<br>Paris</p>"), "Café & art Bonjour Paris")
        with self.assertRaises(ValueError):
            parse_date("2026-11-18T12:00:00")

    def test_schedule_boundaries_geography_and_duplicates(self):
        base = event()
        newest = event(updated_at="2026-11-18T10:00:00+01:00", description="<p>Version récente</p>")
        duplicate_slot = event("2", title="Autre activité", occurrences=base["occurrences"] + ";" + base["occurrences"])
        outside = event("3", locations=[dict(base["locations"][0], address_city="Clichy", address_zipCode="92110")])
        no_schedule = event("4", occurrences=None, date_start="2026-01-01T00:00:00+01:00", date_end="2027-01-01T00:00:00+01:00")
        boundary_end = event("5", occurrences="2026-11-18T11:00:00Z_2026-11-18T12:00:00Z")
        boundary_start = event("6", occurrences="2026-12-18T12:00:00Z_2026-12-18T13:00:00Z")
        valid_overlap = event("7", title="Déjà commencé", occurrences="2026-11-18T11:00:00Z_2026-11-18T13:00:00Z")
        events, rejected, stages, issues, _, _ = self.prepare([base, newest, duplicate_slot, outside, no_schedule, boundary_end, boundary_start, valid_overlap])
        self.assertEqual({e["source_id"] for e in events}, {"1", "2", "7"})
        self.assertEqual(events[0]["description"], "Version récente")
        self.assertEqual(events[0]["occurrences"][0]["start_utc"], "2026-11-19T13:00:00Z")
        self.assertEqual(events[0]["categories"], ["atelier", "exposition"])
        self.assertEqual(issues["duplicate_session"], 1)
        self.assertEqual(len(events) + len(rejected), stages["input"])
        self.assertIn("no_usable_schedule", {r["reason"] for r in rejected})

    def test_multi_locations_price_and_content_dedup(self):
        conditional = event(price_type="gratuit sous condition", locations=[
            event()["locations"][0], dict(event()["locations"][0], address_name="Autre galerie", address_lat_lon="48.857,2.351")])
        copy = deepcopy(conditional)
        copy.update(id="2", event_id=102, url="https://www.paris.fr/evenements/autre-id")
        another_time = event("3", occurrences="2026-11-20T14:00:00+01:00_2026-11-20T16:00:00+01:00")
        zero = event("4", locations=[dict(event()["locations"][0], address_lat_lon="0,0")])
        events, rejected, _, _, _, _ = self.prepare([conditional, copy, another_time, zero])
        self.assertEqual(len(events), 2)
        self.assertIsNone(events[0]["is_free"])
        self.assertIsNone(events[0]["price_amount_eur"])
        self.assertEqual(len(events[0]["locations"]), 2)
        self.assertIn("multi_location_schedule_unassigned", events[0]["quality_flags"])
        self.assertIn("duplicate_content", {r["reason"] for r in rejected})

    def test_full_import_freeze_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            rows = [event(), event("2", title="Séance unique", occurrences=None, price_type="payant", price_detail="<p>5 à 12 euros</p>"), event("3", title="Sans séance", occurrences="invalid")]
            source.write_text(json.dumps(rows), encoding="utf-8")
            original = source.read_bytes()
            mapping = root / "map.json"
            mapping.write_text(json.dumps(MAP), encoding="utf-8")
            args = argparse.Namespace(source=source, output=root / "catalog.sqlite", report=root / "report.json", category_map=mapping, as_of=AS_OF, days=30)
            with redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report["retained"]["events"], 2)
            self.assertEqual(report["stages"]["input"], 3)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(Path(report["manifest"]["frozen_source"]).read_bytes(), original)
            self.assertEqual(report["manifest"]["source_sha256"], hashlib.sha256(original).hexdigest())
            with closing(sqlite3.connect(args.output)) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 1)
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertEqual(db.execute("SELECT price_amount_eur FROM events WHERE source_id='2'").fetchone(), (None,))
                self.assertEqual(db.execute("SELECT count(*) FROM import_rejections").fetchone()[0], 1)
            args.output, args.report = root / "second.sqlite", root / "second.json"
            with redirect_stdout(io.StringIO()):
                second = run(args)
            self.assertEqual(report["manifest"]["snapshot_id"], second["manifest"]["snapshot_id"])
            self.assertEqual((root / "catalog.jsonl").read_bytes(), (root / "second.jsonl").read_bytes())
            with self.assertRaisesRegex(ValueError, "already exists"):
                run(args)
            mapping.write_text(json.dumps({"Expo": "culture", "Atelier": "atelier"}), encoding="utf-8")
            args.output, args.report = root / "third.sqlite", root / "third.json"
            with redirect_stdout(io.StringIO()):
                changed = run(args)
            self.assertNotEqual(report["manifest"]["snapshot_id"], changed["manifest"]["snapshot_id"])


if __name__ == "__main__":
    unittest.main()
