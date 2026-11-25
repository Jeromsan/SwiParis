import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # dépendance optionnelle du cœur ML
    raise unittest.SkipTest('numpy absent : installer avec pip install -e ".[ml]"')

from swiparis.ml.features import FeatureSpace
from swiparis.ml.prior import accuracy, block_mask, fit, load_prior, logistic_loss


class PriorTest(unittest.TestCase):
    def test_fit_recovers_preference_direction(self):
        rng = np.random.default_rng(0)
        true_w = np.array([2.0, -1.0, 0.0])
        x = rng.normal(size=(4000, 3))
        chosen_first = rng.random(4000) < 1 / (1 + np.exp(-(x @ true_w)))
        delta = np.where(chosen_first[:, None], x, -x)  # Δ = φ(choisi) − φ(rejeté)
        w = fit(delta, lam=1e-4)
        self.assertGreater(w[0], 1.5)
        self.assertLess(w[1], -0.5)
        self.assertLess(abs(w[2]), 0.2)
        self.assertLess(logistic_loss(w, delta), logistic_loss(np.zeros(3), delta))

    def test_ties_count_half(self):
        delta = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(accuracy(np.array([1.0, 0.0]), delta), 0.75)

    def test_time_block_is_not_learned(self):
        space = FeatureSpace(("a", "b"), grid=(1, 1))
        mask = block_mask(space)
        self.assertFalse(mask[space.blocks["time"]].any())
        self.assertTrue(mask[space.blocks["cat"]].all())

    def test_load_prior_checks_feature_order(self):
        space = FeatureSpace(("a", "b"), grid=(1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prior.json"
            path.write_text(json.dumps({"feature_names": space.names(), "w0": list(range(space.dim))}))
            np.testing.assert_array_equal(load_prior(path, space), np.arange(space.dim))
            path.write_text(json.dumps({"feature_names": space.names()[::-1], "w0": list(range(space.dim))}))
            with self.assertRaises(ValueError):
                load_prior(path, space)


if __name__ == "__main__":
    unittest.main()
