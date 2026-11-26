"""Personas : regroupement des habitudes, profil de séance mélangé, attribution souple."""
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    from swiparis.ml.model import SessionProfile
    from swiparis.ml.personas import PersonaModel, PersonaProfile, fit_centered, kmeans, known_history
except ImportError:  # numpy absent : extra [ml]
    raise unittest.SkipTest('numpy absent : installer avec pip install -e ".[ml]"')


def model(weights, shares=None, temperature=1.0):
    weights = np.asarray(weights, dtype=float)
    k = len(weights)
    return PersonaModel(np.eye(k), weights, shares or [1 / k] * k, [f"p{i}" for i in range(k)],
                        [""] * k, [f"f{j}" for j in range(weights.shape[1])], temperature=temperature, beta=3.0)


class PersonaTest(unittest.TestCase):
    def test_kmeans_finds_separated_groups_reproducibly(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, 0.05, (30, 2)), rng.normal(3, 0.05, (30, 2))])
        centers, labels = kmeans(x, 2, seed=1)
        self.assertEqual(len(set(labels[:30])), 1)
        self.assertEqual(len(set(labels[30:])), 1)
        self.assertNotEqual(labels[0], labels[-1])
        np.testing.assert_array_equal(labels, kmeans(x, 2, seed=1)[1])

    def test_one_persona_behaves_like_the_global_session(self):
        w0 = np.array([0.3, -0.2, 0.1])
        persona, plain = PersonaProfile(model([w0]), eta=0.2), SessionProfile(w0, eta=0.2)
        for a, b in (([1, 0, 0], [0, 1, 0]), ([0, 0, 1], [1, 0, 0]), ([0, 1, 0], [0, 0, 1])):
            persona.update(np.array(a, float), np.array(b, float))
            plain.update(np.array(a, float), np.array(b, float))
        np.testing.assert_allclose(persona.w, plain.w)

    def test_choices_move_the_weights_towards_the_matching_persona(self):
        personas = model([[2.0, 0.0], [0.0, 2.0]])
        profile = PersonaProfile(personas, eta=0.05)
        for _ in range(3):
            profile.answer(np.array([1.0, 0.0]), np.array([0.0, 1.0]), "left")  # préfère la dimension 0
        self.assertGreater(profile.weights_pi[0], 0.9)
        self.assertGreater(profile.w[0], profile.w[1])
        profile.answer(np.array([1.0, 0.0]), np.array([0.0, 1.0]), "skip")  # un passage ne change rien
        self.assertEqual(profile.n_updates, 3)

    def test_same_choices_different_history_give_different_profiles(self):
        personas = model([[2.0, 0.0], [0.0, 2.0]])
        a = PersonaProfile(personas, start=personas.start(np.array([1.0, 0.0])), eta=0.05)
        b = PersonaProfile(personas, start=personas.start(np.array([0.0, 1.0])), eta=0.05)
        for profile in (a, b):
            profile.answer(np.array([1.0, 1.0]), np.array([0.0, 0.0]), "left")
        self.assertGreater(a.w[0], a.w[1])
        self.assertGreater(b.w[1], b.w[0])

    def test_centered_fit_stays_at_the_center_without_data_and_save_load(self):
        center = np.array([0.5, -0.5])
        np.testing.assert_allclose(fit_centered(np.zeros((0, 2)), 0.1, center), center)
        personas = model([[1.0, 0.0], [0.0, 1.0]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "personas.json"
            personas.save(path)
            loaded = PersonaModel.load(path)
        np.testing.assert_allclose(loaded.weights, personas.weights)
        self.assertEqual((loaded.names, loaded.temperature, loaded.beta), (personas.names, 1.0, 3.0))

    def test_known_history_stops_before_the_sixth_distinct_place(self):
        visits = [("t1", "a", 0), ("t2", "b", 0), ("t3", "a", 1), ("t4", "c", 2), ("t5", "d", 3),
                  ("t6", "e", 3), ("t7", "b", 2), ("t8", "f", 1)]
        self.assertEqual([v[1] for v in known_history(visits)], ["a", "b", "a", "c", "d", "e", "b"])


if __name__ == "__main__":
    unittest.main()
