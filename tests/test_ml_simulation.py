import unittest

try:
    import numpy as np
except ImportError:  # dépendance optionnelle du cœur ML
    raise unittest.SkipTest('numpy absent : installer avec pip install -e ".[ml]"')

from swiparis.ml.simulation import profile_similarity, simulate_session, uniform_pair


class SimulationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.phi = np.vstack([np.eye(4), rng.random((20, 4))])
        self.ids = [f"i{k}" for k in range(len(self.phi))]
        self.similarity = profile_similarity(self.phi, self.phi[[0]])  # profil : un lieu de type « 0 »

    def test_similarity_is_max_cosine(self):
        self.assertAlmostEqual(self.similarity[0], 1.0)
        self.assertAlmostEqual(self.similarity[1], 0.0)

    def test_sessions_follow_budget_and_oracle(self):
        for strategy in ("active", "uniform"):
            profile, log = simulate_session(np.zeros(4), self.phi, self.ids, self.similarity, strategy,
                                            np.random.default_rng(1))
            self.assertLessEqual(len(log), 8)
            self.assertTrue(profile.n_updates == 5 or len(log) == 8)
            index = {item: k for k, item in enumerate(self.ids)}
            for (left, right), choice in log:
                gap = self.similarity[index[left]] - self.similarity[index[right]]
                expected = "left" if gap > 1e-9 else "right" if gap < -1e-9 else "skip"
                self.assertEqual(choice, expected)
            self.assertGreater(profile.w[0], 0)  # le goût du profil est appris
        with self.assertRaises(ValueError):
            simulate_session(np.zeros(4), self.phi, self.ids, self.similarity, "greedy", np.random.default_rng(1))

    def test_uniform_pair_skips_shown_and_identical(self):
        phi = np.array([[1.0, 0], [1.0, 0], [0, 1.0]])
        pair = uniform_pair(phi, ["a", "b", "c"], {frozenset(("a", "c"))}, np.random.default_rng(0))
        self.assertEqual(set(pair), {"b", "c"})


if __name__ == "__main__":
    unittest.main()
