import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from swiparis.ml.features import FeatureSpace, Item
from swiparis.serving import LATEST, load_model, register_prior, space_from_prior

CATEGORIES = ["culture", "nature", "sport", "spectacle"]
REAL_PRIOR = Path(__file__).resolve().parents[1] / "models" / "prior_w0_phi.json"


def make_prior(path, seed=0, created="2026-11-20T14:30:10Z"):
    space = FeatureSpace(CATEGORIES, popularity=True)
    w0 = np.random.default_rng(seed).normal(size=space.dim).round(6)
    prior = {"model": "bradley_terry_linear_phi", "created_utc": created, "feature_names": space.names(),
             "grid": list(space.grid), "trained_blocks": ["cat", "price", "zone", "pop"], "w0": w0.tolist(),
             "lambda": 1e-4, "pop_scale": 2, "metrics": {"val_accuracy": 0.57}, "inputs": {"pairs_sha256": "abc"}}
    Path(path).write_text(json.dumps(prior), encoding="utf-8")
    return space


def make_items():
    boxes = [(48.85, 2.35), (48.83, 2.30), (48.88, 2.40)]
    return [Item(f"i{n}", (CATEGORIES[n % 4],), "free" if n % 2 else "paid", boxes[n % 3][0], boxes[n % 3][1])
            for n in range(24)]


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.space = make_prior(self.root / "prior.json")
        self.models = self.root / "models"

    def tearDown(self):
        self.tmp.cleanup()

    def register(self):
        return register_prior(self.root / "prior.json", self.models)

    def test_register_writes_versioned_artifacts_and_is_idempotent(self):
        first = self.register()
        self.assertTrue(first.name.startswith("prior-20261120-"))
        for name in ("prior.json", "w0.npy", "metadata.json"):
            self.assertTrue((first / name).exists())
        self.assertEqual((self.models / LATEST).read_text().strip(), first.name)
        meta = json.loads((first / "metadata.json").read_text())
        self.assertEqual(meta["feature_dim"], self.space.dim)
        self.assertEqual(meta["metrics"]["val_accuracy"], 0.57)
        self.assertEqual(self.register(), first)  # deuxième appel : même dossier, rien de réécrit

    def test_same_prior_same_version_different_prior_different_version(self):
        a = self.register()
        make_prior(self.root / "prior.json", seed=1)
        b = self.register()
        self.assertNotEqual(a.name, b.name)
        self.assertEqual(load_model(self.models).version, b.name)  # latest pointe vers le dernier
        self.assertEqual(load_model(a).version, a.name)  # une ancienne version reste chargeable

    def test_load_rebuilds_the_training_feature_space(self):
        self.register()
        model = load_model(self.models)
        self.assertEqual(model.space.names(), self.space.names())
        np.testing.assert_allclose(model.w0, json.loads((self.root / "prior.json").read_text())["w0"])

    def test_tampered_prior_is_refused(self):
        version = self.register()
        prior = json.loads((version / "prior.json").read_text())
        prior["w0"][0] += 1
        (version / "prior.json").write_text(json.dumps(prior))
        with self.assertRaises(ValueError):
            load_model(self.models)

    def test_inconsistent_prior_is_refused_at_registration(self):
        prior = json.loads((self.root / "prior.json").read_text())
        prior["w0"] = prior["w0"][:-1]
        with self.assertRaises(ValueError):
            space_from_prior(prior)
        prior["w0"] = prior["w0"] + [0.0]
        prior["feature_names"] = list(reversed(prior["feature_names"]))
        with self.assertRaises(ValueError):
            space_from_prior(prior)

    def test_missing_registry_gives_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            load_model(self.root / "empty")

    def test_predict_returns_k_ranked_unique_items_and_honours_exclude(self):
        self.register()
        model = load_model(self.models)
        items = make_items()
        ids = [i.item_id for i in items]
        phi = model.encode(items)
        top = model.predict(phi, ids, k=8)
        self.assertEqual(len(top), 8)
        self.assertEqual(len({i for i, _ in top}), 8)
        banned = [top[0][0], top[1][0]]
        again = model.predict(phi, ids, k=8, exclude=banned)
        self.assertFalse(set(banned) & {i for i, _ in again})
        self.assertEqual(model.predict(phi, ids, k=8), top)  # déterministe

    def test_answers_move_the_ranking_towards_the_chosen_category(self):
        self.register()
        model = load_model(self.models)
        items = make_items()
        ids = [i.item_id for i in items]
        phi = model.encode(items)
        category = {i.item_id: i.categories[0] for i in items}
        sport = [i for i in ids if category[i] == "sport"]
        culture = [i for i in ids if category[i] == "culture"]
        answers = [(s, c) for s, c in zip(sport, culture)] * 3
        before = model.profile_from(phi, ids).scores(phi)
        after = model.profile_from(phi, ids, answers).scores(phi)
        idx = {i: p for p, i in enumerate(ids)}
        gain = lambda group: np.mean([after[idx[i]] - before[idx[i]] for i in group])
        self.assertGreater(gain(sport), gain(culture))

    def test_unknown_ids_in_answers_are_ignored_and_shapes_are_checked(self):
        self.register()
        model = load_model(self.models)
        items = make_items()
        ids = [i.item_id for i in items]
        phi = model.encode(items)
        self.assertEqual(model.predict(phi, ids, answers=[("ghost", "i1")], k=5), model.predict(phi, ids, k=5))
        with self.assertRaises(ValueError):
            model.predict(phi[:, :-1], ids)
        with self.assertRaises(ValueError):
            model.predict(phi, ids[:-1])

    def test_next_pair_skips_pairs_already_shown(self):
        self.register()
        model = load_model(self.models)
        items = make_items()
        ids = [i.item_id for i in items]
        phi = model.encode(items)
        pair = model.next_pair(phi, ids, rng=0)
        self.assertIsNotNone(pair)
        other = model.next_pair(phi, ids, shown_pairs=[pair], rng=0)
        self.assertNotEqual(frozenset(pair), frozenset(other))

    @unittest.skipUnless(REAL_PRIOR.exists(), "prior réel absent")
    def test_real_prior_registers_and_predicts(self):
        version = register_prior(REAL_PRIOR, self.models)
        model = load_model(version)
        self.assertEqual(model.space.dim, 28)
        self.assertTrue(model.space.popularity)
        items = [Item(f"x{n}", (model.space.categories[n % len(model.space.categories)],), "unknown",
                      48.85 + 0.001 * n, 2.35) for n in range(30)]
        top = model.predict(model.encode(items), [i.item_id for i in items], k=10)
        self.assertEqual(len(top), 10)


if __name__ == "__main__":
    unittest.main()
