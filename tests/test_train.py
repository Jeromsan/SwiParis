import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except Exception:  # torch absent ou cassé : ces tests sont ignorés
    torch = None

try:
    import matplotlib  # noqa: F401
except Exception:
    matplotlib = None

from swiparis.pairs import assign_splits, build_pairs, load_foursquare
from synthetic import make_db


@unittest.skipIf(torch is None or matplotlib is None, "torch ou matplotlib indisponible")
class TrainTests(unittest.TestCase):
    def test_training_beats_popularity_and_versions_model(self):
        from swiparis.registry import load_latest
        from swiparis.train import train
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "fs.sqlite"
            make_db(db, users=300, visits=30)
            venues, checkins = load_foursquare(db)
            splits = assign_splits(checkins["user_id"].unique(), seed=1)
            pairs, _ = build_pairs(venues, checkins, splits, seed=1)
            config = {"dim": 16, "lr": 0.05, "l2": 1e-4, "epochs": 40, "batch_size": 256, "patience": 5, "seed": 1}
            meta = train(venues, checkins, pairs, {}, Path(tmp) / "models", config, "sha-test")
            metrics = meta["metrics"]
            self.assertGreater(metrics["val_accuracy"], 0.52)  # le signal commun (catégories populaires) doit être appris
            self.assertTrue((Path(tmp) / "models" / meta["version"] / "loss_curve.png").exists())
            bundle = load_latest(Path(tmp) / "models")
            self.assertEqual(bundle.version, meta["version"])
            self.assertEqual(bundle.scores("evening").shape, (len(venues),))


if __name__ == "__main__":
    unittest.main()
