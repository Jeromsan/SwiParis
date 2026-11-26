from pathlib import Path
import tempfile
import unittest

from common_fixture import build_common
from swiparis.catalog import Catalog, occurrence_utc

SLOT = {"start_utc": "2026-11-28T13:00:00Z", "end_utc": "2026-11-28T16:00:00Z"}


class SlotEligibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.catalog = Catalog(build_common(Path(cls.temp.name) / "common.sqlite"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def eligible(self, constraints):
        return {row["item_id"].split(":")[1]: row for row in self.catalog.list_items(constraints, limit=None)["items"]}

    def test_at_least_one_hour_or_the_whole_session(self):
        rows = self.eligible(SLOT)
        self.assertIn("expo", rows)        # ouverte toute la journée : 3 h de recouvrement
        self.assertIn("concert", rows)     # entièrement dans le créneau
        self.assertIn("court", rows)       # 30 min, mais entière
        self.assertNotIn("tardif", rows)   # spectacle de 2 h qui ne tient pas dans le créneau
        self.assertNotIn("court-bord", rows)  # séance courte coupée par la fin du créneau
        self.assertNotIn("diner", rows)    # après le créneau
        self.assertEqual(rows["expo"]["next_occurrence"]["start_utc"], "2026-11-28T09:00:00Z")

    def test_without_slot_the_snapshot_window_is_used(self):
        self.assertEqual(len(self.eligible({})), 10)

    def test_short_slot_keeps_long_sessions(self):
        rows = self.eligible({"start_utc": "2026-11-28T14:00:00Z", "end_utc": "2026-11-28T15:00:00Z"})
        self.assertEqual(set(rows), {"expo", "jardin", "jeux"})  # horaires d'ouverture, 1 h sur place

    def test_source_local_time_is_read_in_paris(self):
        # Paris Data écrit +02:00 toute l'année : en hiver, l'heure locale écrite fait foi.
        self.assertEqual(occurrence_utc("2026-11-28T14:00:00+02:00"), "2026-11-28T13:00:00Z")
        self.assertEqual(occurrence_utc("2026-07-01T14:00:00+02:00"), "2026-07-01T12:00:00Z")  # été : déjà juste
        self.assertEqual(occurrence_utc("2026-11-28T14:00:00+01:00"), "2026-11-28T13:00:00Z")  # déjà juste
        self.assertEqual(occurrence_utc("2026-11-28T14:00:00Z"), "2026-11-28T14:00:00Z")       # pas un décalage de Paris
        # Les colonnes start_utc de la base de test sont fausses d'une heure, comme dans common.sqlite :
        # le catalogue doit renvoyer la vraie heure (expo ouverte à 10 h à Paris = 09:00Z).
        self.assertEqual(self.eligible(SLOT)["expo"]["next_occurrence"]["start_utc"], "2026-11-28T09:00:00Z")


if __name__ == "__main__":
    unittest.main()
