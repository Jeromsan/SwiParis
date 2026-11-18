import argparse
from contextlib import closing, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from swiparis.import_foursquare import DEFAULT_BBOX, inside, run, utc_time


class ImportTest(unittest.TestCase):
    def test_geography_and_utc(self):
        self.assertTrue(inside(48.8566, 2.3522, DEFAULT_BBOX))
        self.assertTrue(inside(48.815, 2.224, DEFAULT_BBOX))
        self.assertFalse(inside(48.8049, 2.1204, DEFAULT_BBOX))  # Versailles
        self.assertFalse(inside(51.5074, -0.1278, DEFAULT_BBOX))
        self.assertEqual(utc_time("Tue Apr 03 18:00:06 +0000 2012"), "2012-04-03T18:00:06Z")

    def fixture(self, root):
        pois = root / "pois.txt"
        pois.write_text("p\t48.8566\t2.3522\tMuseum\tFR\n"
                        "o\t48.8049\t2.1204\tMuseum\tFR\n"
                        "s\t48.85\t2.35\tOffice\tFR\n"
                        "x\t48.85\t2.35\tMuseum\tUS\n"
                        "bad\tnan\t2.35\tMuseum\tFR\n", encoding="utf-8")
        visits = []
        # Deliberately reversed: no assumption about input ordering.
        for user, count in (("ten", 10), ("nine", 9)):
            visits.extend(f"{user}\tp\tTue Apr 03 18:00:{i:02d} +0000 2012\t120\n"
                          for i in reversed(range(count)))
        visits.extend([visits[0],  # duplicate must not take 'nine' over threshold
                       "nine\tp\tTue Apr 03 18:00:00 +0000 2012\t120\n"])
        for venue in ("o", "s", "x", "unknown"):
            visits.append(f"nine\t{venue}\tTue Apr 03 18:00:30 +0000 2012\t120\n")
        visits.append("broken\trow\n")
        checkins = root / "checkins.txt"
        checkins.write_text("".join(visits), encoding="utf-8")
        categories = root / "categories.json"
        categories.write_text('["Museum"]', encoding="utf-8")
        return argparse.Namespace(pois=pois, checkins=checkins, categories=categories,
            output=root / "out.sqlite", report=root / "report.json", inventory_only=False,
            bbox=DEFAULT_BBOX, min_checkins=10, batch_size=3, small_min_users=100, small_min_visits=1000)

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertEqual(report["stages"]["valid_input"]["visits"], 25)
            self.assertEqual(report["stages"]["known_venue"]["visits"], 24)
            self.assertEqual(report["stages"]["geography"]["visits"], 22)
            self.assertEqual(report["stages"]["leisure"]["visits"], 21)
            self.assertEqual(report["stages"]["deduplicated"]["visits"], 19)
            self.assertEqual(report["stages"]["user_threshold"],
                             {"visits": 10, "visited_venues": 1, "users": 1, "venues": 1})
            self.assertEqual(report["input"]["checkins"]["rejected"], 1)
            self.assertEqual(report["input"]["pois"]["rejected"], 1)
            self.assertTrue(report["warnings"])
            with closing(sqlite3.connect(args.output)) as db:
                rows = db.execute("SELECT user_id,utc_time FROM checkins ORDER BY user_id,utc_time,venue_id").fetchall()
                self.assertEqual({row[0] for row in rows}, {"ten"})
                self.assertEqual(rows, sorted(rows))
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(json.loads(args.report.read_text(encoding="utf-8")), report)
            with self.assertRaisesRegex(ValueError, "already exists"):
                run(args)

    def test_inventory_and_unknown_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            args.categories.write_text('["Invented category"]', encoding="utf-8")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "absent"):
                run(args)
            self.assertFalse(args.output.exists())
            args.categories = None
            args.inventory_only = True
            with redirect_stdout(io.StringIO()):
                report = run(args)
            self.assertFalse(args.output.exists())
            self.assertEqual([c["category"] for c in report["categories_found"]], ["Museum", "Office"])


if __name__ == "__main__":
    unittest.main()
