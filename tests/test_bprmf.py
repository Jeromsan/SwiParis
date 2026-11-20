import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except Exception:
    torch = None

from swiparis.pairs import assign_splits, build_pairs, load_foursquare
from synthetic import make_db


@unittest.skipIf(torch is None, "torch indisponible")
class BprTests(unittest.TestCase):
    def test_bpr_with_fold_in_beats_popularity_and_random(self):
        from swiparis.bprmf import BPRMF, evaluate_split, fit
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fs.sqlite"
            make_db(db, users=300, visits=30)
            venues, checkins = load_foursquare(db)
            splits = assign_splits(checkins["user_id"].unique(), seed=3)
            pairs, _ = build_pairs(venues, checkins, splits, seed=3)
            train_users = {u for u, s in splits.items() if s == "train"}
            model = fit(venues, checkins, train_users, dim=16, epochs=15, seed=3)
            result = evaluate_split(model, venues, checkins, pairs, "val")
            self.assertGreater(result["pairs"], 500)
            self.assertGreater(result["bpr_mf"], result["popularity"])
            self.assertGreater(result["bpr_mf"], 0.55)
            path = Path(tmp) / "bpr.npz"
            model.save(path)
            reloaded = BPRMF.load(path)
            self.assertEqual(reloaded.venue_ids, model.venue_ids)


if __name__ == "__main__":
    unittest.main()
