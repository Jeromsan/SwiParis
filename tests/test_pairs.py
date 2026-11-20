import tempfile
import unittest
from pathlib import Path

import pandas as pd

from swiparis import baselines
from swiparis.pairs import (STRATEGIES, assign_splits, build_pairs, load_foursquare,
                            nearest_neighbors, time_bucket)
from synthetic import make_db


class PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "fs.sqlite"
        make_db(cls.db)
        cls.venues, cls.checkins = load_foursquare(cls.db)
        cls.splits = assign_splits(cls.checkins["user_id"].unique(), seed=7)
        cls.pairs, cls.info = build_pairs(cls.venues, cls.checkins, cls.splits, seed=7)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_split_by_user_60_20_20(self):
        counts = pd.Series(self.splits).value_counts()
        self.assertEqual((counts["train"], counts["val"], counts["test"]), (72, 24, 24))
        per_user = self.pairs.groupby("user_id")["split"].nunique()
        self.assertTrue((per_user == 1).all())

    def test_reproducible_with_same_seed(self):
        again, _ = build_pairs(self.venues, self.checkins, self.splits, seed=7)
        pd.testing.assert_frame_equal(self.pairs, again)
        other, _ = build_pairs(self.venues, self.checkins, self.splits, seed=8)
        self.assertFalse(self.pairs["rejected_venue_id"].equals(other["rejected_venue_id"]))

    def test_negatives_are_never_visited_by_the_user(self):
        visited = self.checkins.groupby("user_id")["venue_id"].apply(set)
        for row in self.pairs.itertuples():
            self.assertNotIn(row.rejected_venue_id, visited[row.user_id])
            self.assertIn(row.chosen_venue_id, visited[row.user_id])
            self.assertNotEqual(row.chosen_venue_id, row.rejected_venue_id)

    def test_one_pair_per_visit_and_labelled_weak(self):
        self.assertEqual(len(self.pairs) + self.info["skipped_visits"], len(self.checkins))
        self.assertTrue(set(self.pairs["negative_type"]) <= {"weak_" + s for s in STRATEGIES})

    def test_time_bucket_uses_local_time(self):
        out = time_bucket(pd.Series(["2012-05-01T23:30:00Z"]), pd.Series([120]))  # 01:30 local
        self.assertEqual(int(out[0]), 0)

    def test_neighbors_exclude_self(self):
        near = nearest_neighbors(self.venues["latitude"].to_numpy(), self.venues["longitude"].to_numpy(), 5)
        self.assertFalse(any(i in row for i, row in enumerate(near)))

    def test_baselines_report_shape(self):
        out = baselines.report(self.venues, self.checkins, self.pairs)
        for name in ("random", "popularity"):
            self.assertAlmostEqual(out[name]["train"]["pairs"], int((self.pairs["split"] == "train").sum()))
        self.assertAlmostEqual(out["random"]["val"]["pairwise_accuracy"], 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
