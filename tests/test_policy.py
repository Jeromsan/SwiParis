"""Réglage de η par séance (comparaison pendant le pilote) et habitudes tirées d'activités."""
import os
import unittest
from unittest.mock import patch

try:
    import numpy as np
    from swiparis.ml import policy
    from swiparis.ml.features import FeatureSpace, Item
    from swiparis.ml.personas import habit_from_phi
except ImportError:
    raise unittest.SkipTest('numpy absent : installer avec pip install -e ".[ml]"')


class EtaArmsTest(unittest.TestCase):
    def test_arms_are_read_from_the_environment(self):
        with patch.dict(os.environ, {"SWIPARIS_SESSION_ETA_AB": "0.2,0.5"}):
            self.assertEqual(policy.eta_arms(), (0.2, 0.5))
        with patch.dict(os.environ, {"SWIPARIS_SESSION_ETA_AB": ""}):
            self.assertIsNone(policy.eta_arms())
        for bad in ("0.5", "0.5,0.5", "0,0.5", "0.2,3"):
            with patch.dict(os.environ, {"SWIPARIS_SESSION_ETA_AB": bad}), self.assertRaises(ValueError):
                policy.eta_arms()

    def test_each_session_keeps_its_value_and_both_values_are_used(self):
        arms = (0.2, 0.5)
        values = [policy.eta_for_session(f"session-{n}", arms, 0.5) for n in range(400)]
        self.assertEqual(values, [policy.eta_for_session(f"session-{n}", arms, 0.5) for n in range(400)])
        self.assertTrue(150 < values.count(0.2) < 250)  # environ moitié-moitié
        self.assertEqual(policy.eta_for_session("x", None, 0.5), 0.5)

    def test_the_value_is_found_again_from_the_logged_version(self):
        arms = (0.2, 0.5)
        version = f"prior-x+session-v4+{policy.eta_version(arms, 0.5)}+personas6"
        self.assertEqual(version, "prior-x+session-v4+etaAB0.2_0.5+personas6")
        for n in range(20):
            sid = f"s{n}"
            self.assertEqual(policy.eta_from_version(version, sid), policy.eta_for_session(sid, arms, 0.5))
        self.assertEqual(policy.eta_from_version("prior-x+session-v4+eta0.05", "s"), 0.05)
        self.assertIsNone(policy.eta_from_version("prior-x+session-v3", "s"))


class HabitFromItemsTest(unittest.TestCase):
    def test_habits_follow_the_categories_times_and_zones_of_the_items(self):
        space = FeatureSpace(("culture", "spectacle"), grid=(1, 2))
        items = [Item("a", ("culture",), "free", 48.85, 2.25, (0, 0, 1, 0)),
                 Item("b", ("culture",), "free", 48.85, 2.25, (0, 0, 0, 1))]
        habit = habit_from_phi(space.encode_many(items), space)
        np.testing.assert_allclose(habit[:2], [1, 0])                      # catégories
        np.testing.assert_allclose(habit[2:6], np.sqrt([0, 0, 0.5, 0.5]))   # moments
        np.testing.assert_allclose(np.linalg.norm(habit[6:]), 1)            # zones


if __name__ == "__main__":
    unittest.main()
