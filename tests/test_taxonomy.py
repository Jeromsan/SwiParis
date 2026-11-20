import sqlite3
import tempfile
import unittest
from pathlib import Path

from swiparis.pairs import drop_common_categories, load_foursquare
from synthetic import CATEGORIES, make_common_db, make_db


class CommonDbTests(unittest.TestCase):
    def test_loader_reads_common_schema_like_raw_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, common = Path(tmp) / "raw.sqlite", Path(tmp) / "common.sqlite"
            make_db(raw, users=10, visits=8)
            make_common_db(raw, common)
            v_raw, c_raw = load_foursquare(raw)
            v_com, c_com = load_foursquare(common)
            self.assertEqual(len(v_raw), len(v_com))
            self.assertEqual(len(c_raw), len(c_com))
            self.assertTrue(v_com["venue_id"].str.startswith("foursquare:").all())
            self.assertTrue(c_com["user_id"].str.startswith("foursquare:").all())
            self.assertEqual(sorted(v_com["category"].unique()), sorted(v_raw["category"].unique()))

    def test_drop_common_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fs.sqlite"
            make_db(db, users=20, visits=10)
            venues, checkins = load_foursquare(db)
            kept_v, kept_c = drop_common_categories(venues, checkins, {"Bar": "vie_nocturne"}, ["vie_nocturne"])
            self.assertNotIn("Bar", set(kept_v["category"]))
            self.assertEqual(len(kept_v), len(venues) - 15)
            self.assertTrue(kept_c["venue_id"].isin(set(kept_v["venue_id"])).all())
            self.assertIn("Bar", CATEGORIES)


if __name__ == "__main__":
    unittest.main()
