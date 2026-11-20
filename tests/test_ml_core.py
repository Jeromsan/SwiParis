from contextlib import closing
from importlib.resources import files
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

try:
    import numpy as np
except ImportError:  # dépendance optionnelle du cœur ML
    raise unittest.SkipTest('numpy absent : installer avec pip install -e ".[ml]"')

from swiparis.ml.features import FeatureSpace, Item, foursquare_local_hour, source_local_hour, time_profile
from swiparis.ml.items import common_categories, common_items, encode_common
from swiparis.ml.model import SessionProfile
from swiparis.ml.selection import (acquisition_matrix, binary_entropy, intra_list_diversity,
                                   mmr_rerank, next_pair, rank, session_finished)

REAL_DB = Path(__file__).resolve().parents[1] / "data" / "processed" / "common.sqlite"

CATEGORIES = ("culture", "spectacle", "nature", "sport", "nightlife")
SPACE = FeatureSpace(CATEGORIES, grid=(2, 2))
CATALOG = [
    Item("louvre", ("culture",), "paid", 48.861, 2.336, (0, .6, .4, 0)),
    Item("concert", ("spectacle",), "paid", 48.88, 2.37, (0, 0, .2, .8)),
    Item("parc", ("nature",), "free", 48.82, 2.25, (0, .5, .5, 0)),
    Item("atelier", ("culture",), "conditional_free", 48.83, 2.40, (0, 0, 1, 0)),
    Item("foot", ("sport",), "unknown", 48.84, 2.25, None),
    Item("bar", ("nightlife",), "unknown", 48.865, 2.38, (.3, 0, 0, .7)),
]
PHI = SPACE.encode_many(CATALOG)
IDS = [item.item_id for item in CATALOG]


def phi_of(item_id):
    return PHI[IDS.index(item_id)]


class FeatureTest(unittest.TestCase):
    def test_blocks_and_names(self):
        self.assertEqual(SPACE.dim, 5 + 4 + 4 + 4)
        self.assertEqual(len(SPACE.names()), SPACE.dim)
        self.assertEqual(PHI.shape, (len(CATALOG), SPACE.dim))

    def test_unknown_price_is_not_free(self):
        price = SPACE.blocks["price"]
        self.assertEqual(SPACE.encode(Item("a", price_type="unknown"))[price].tolist(), [0, 0, 0, 1])
        self.assertEqual(SPACE.encode(Item("b", price_type="free"))[price].tolist(), [1, 0, 0, 0])

    def test_categories(self):
        x = SPACE.encode(Item("a", ("culture", "nature")))
        self.assertEqual(x[SPACE.blocks["cat"]].tolist(), [.5, 0, .5, 0, 0])
        with self.assertRaises(ValueError):
            SPACE.encode(Item("b", ("unknown",)))

    def test_zone_and_missing_values(self):
        self.assertEqual(SPACE.zone(48.903, 2.224), 0)  # coin nord-ouest
        self.assertEqual(SPACE.zone(48.815, 2.470), 3)  # coin sud-est
        self.assertIsNone(SPACE.zone(None, 2.3))
        with self.assertRaises(ValueError):
            SPACE.zone(48.80, 2.12)  # Versailles
        x = SPACE.encode(Item("a"))
        self.assertFalse(x[SPACE.blocks["zone"]].any())
        self.assertFalse(x[SPACE.blocks["time"]].any())
        with self.assertRaises(ValueError):
            SPACE.encode(Item("b", time_profile=(.5, 0, 0, 0)))

    def test_text_embedding(self):
        space = FeatureSpace(CATEGORIES, text_dim=3)
        x = space.encode(Item("a", text_embedding=(3, 4, 0)))
        np.testing.assert_allclose(x[space.blocks["text"]], [.6, .8, 0])

    def test_popularity_block(self):
        space = FeatureSpace(CATEGORIES, grid=(2, 2), popularity=True)
        self.assertEqual(space.dim, SPACE.dim + 1)
        self.assertEqual(space.names()[-1], "pop:log_visits")
        self.assertEqual(space.encode(Item("a", popularity=.25))[space.blocks["pop"]].tolist(), [.25])
        self.assertEqual(space.encode(Item("b"))[space.blocks["pop"]].tolist(), [0])  # inconnue
        with self.assertRaises(ValueError):
            space.encode(Item("c", popularity=2))

    def test_hours(self):
        self.assertEqual(foursquare_local_hour("2012-04-03T18:08:01Z", 120), 20)
        self.assertEqual(foursquare_local_hour("2012-04-03T01:00:00Z", -120), 23)
        self.assertEqual(source_local_hour("2026-11-20T19:30:00+01:00"), 19)
        self.assertEqual(time_profile([1, 7, 13, 19]), (.25, .25, .25, .25))
        self.assertIsNone(time_profile([]))


class ModelTest(unittest.TestCase):
    def test_sgd_matches_the_written_rule(self):
        profile = SessionProfile(np.zeros(SPACE.dim), eta=.5, lam=.05)
        delta = phi_of("louvre") - phi_of("concert")
        self.assertAlmostEqual(profile.update(phi_of("louvre"), phi_of("concert")), .5)
        np.testing.assert_allclose(profile.w, .5 * .5 * delta)  # w ← w + η(1 − σ(wᵀΔ))Δ − λ(w − w₀)
        prior = np.linspace(-1, 1, SPACE.dim)
        profile = SessionProfile(prior, eta=.5, lam=.05)
        profile.w = prior + 1
        profile.update(phi_of("louvre"), phi_of("louvre"))  # Δ = 0 : seul le rappel vers w₀ agit
        np.testing.assert_allclose(profile.w, prior + .95)

    def test_choice_increases_its_probability(self):
        for mode in ("sgd", "bayes"):
            profile = SessionProfile(np.zeros(SPACE.dim), mode=mode)
            before = profile.prob(phi_of("parc"), phi_of("bar"))
            profile.update(phi_of("parc"), phi_of("bar"))
            self.assertGreater(profile.prob(phi_of("parc"), phi_of("bar")), before)
            self.assertAlmostEqual(float(profile.prob(phi_of("parc"), phi_of("bar"))
                                         + profile.prob(phi_of("bar"), phi_of("parc"))), 1)

    def test_reversing_choices_changes_the_top_activity(self):
        for mode in ("sgd", "bayes"):
            tops = []
            for chosen, rejected in (("louvre", "concert"), ("concert", "louvre")):
                profile = SessionProfile(np.zeros(SPACE.dim), mode=mode)
                for _ in range(5):
                    profile.update(phi_of(chosen), phi_of(rejected))
                tops.append(rank(profile, PHI, IDS, k=1, diversity=0)[0][0])
            self.assertEqual(tops, ["louvre", "concert"], mode)

    def test_bayes_uncertainty_shrinks(self):
        profile = SessionProfile(np.zeros(SPACE.dim), mode="bayes", prior_var=2.0)
        delta = phi_of("parc") - phi_of("bar")
        before = profile.margin_variance(delta)[0]
        self.assertAlmostEqual(before, 2.0 * delta @ delta)
        profile.update(phi_of("parc"), phi_of("bar"))
        self.assertLess(profile.margin_variance(delta)[0], before)
        np.testing.assert_allclose(profile.cov, profile.cov.T)

    def test_state_round_trip(self):
        for mode in ("sgd", "bayes"):
            profile = SessionProfile(np.zeros(SPACE.dim), mode=mode)
            profile.update(phi_of("parc"), phi_of("bar"))
            copy = SessionProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
            np.testing.assert_allclose(copy.scores(PHI), profile.scores(PHI))
            self.assertEqual(copy.n_updates, 1)

    def test_log_answers(self):
        profile = SessionProfile(np.zeros(SPACE.dim))
        self.assertIsNone(profile.answer(phi_of("parc"), phi_of("bar"), "skip"))
        self.assertEqual(profile.n_updates, 0)
        self.assertFalse(profile.w.any())
        profile.answer(phi_of("bar"), phi_of("parc"), "right")  # parc était à droite et a été choisi
        self.assertGreater(profile.prob(phi_of("parc"), phi_of("bar")), .5)
        with self.assertRaises(ValueError):
            profile.answer(phi_of("parc"), phi_of("bar"), "both")

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            SessionProfile(np.zeros(3), lam=1)
        with self.assertRaises(ValueError):
            SessionProfile(np.zeros(3), mode="adam")
        with self.assertRaises(ValueError):
            SessionProfile(np.zeros(3)).update(np.zeros(2), np.zeros(2))


class SelectionTest(unittest.TestCase):
    def test_entropy(self):
        self.assertAlmostEqual(float(binary_entropy(.5)), 1)
        self.assertAlmostEqual(float(binary_entropy(0)), 0, places=6)
        self.assertAlmostEqual(float(binary_entropy(.2)), float(binary_entropy(.8)))

    def test_distance_term_penalizes_lookalike_pairs(self):
        twins = SPACE.encode_many([CATALOG[0], Item("louvre-2", *[getattr(CATALOG[0], f) for f in
                                   ("categories", "price_type", "latitude", "longitude", "time_profile")]),
                                   CATALOG[2]])
        values = acquisition_matrix(SessionProfile(np.zeros(SPACE.dim)), twins, beta=.5)
        self.assertAlmostEqual(values[0, 1], 1.0)  # P = 0,5 mais Δ = 0
        self.assertGreater(values[0, 2], values[0, 1])

    def test_next_pair_skips_shown_and_identical_pairs(self):
        profile = SessionProfile(np.zeros(SPACE.dim))
        first = next_pair(profile, PHI, IDS, rng=0)
        self.assertEqual(first, next_pair(profile, PHI, IDS, rng=0))
        self.assertEqual(len(set(first)), 2)
        self.assertTrue(set(first) <= set(IDS))
        second = next_pair(profile, PHI, IDS, shown_pairs=[first], rng=0)
        self.assertNotEqual(set(second), set(first))
        self.assertIsNone(next_pair(profile, PHI, IDS, shown_pairs=combinations(IDS, 2), rng=0))
        twins = SPACE.encode_many([CATALOG[0], CATALOG[0]])
        self.assertIsNone(next_pair(profile, twins, ["a", "b"], rng=0))

    def test_next_pair_falls_back_to_the_full_catalog(self):
        profile = SessionProfile(np.zeros(SPACE.dim))
        ids = IDS[:4]
        remaining = {"parc", "atelier"}
        shown = [pair for pair in combinations(ids, 2) if set(pair) != remaining]
        for seed in range(5):
            found = next_pair(profile, PHI[:4], ids, shown_pairs=shown, pool_size=2, rng=seed)
            self.assertEqual(set(found), remaining)

    def test_mmr(self):
        self.assertEqual(mmr_rerank([.2, .9, .5], np.eye(3), k=3, diversity=0), [1, 2, 0])
        phi = np.array([[1, 0], [1, .01], [1, .02], [0, 1]])
        scores = [1.0, .99, .98, .5]
        self.assertEqual(mmr_rerank(scores, phi, k=2, diversity=0), [0, 1])
        diverse = mmr_rerank(scores, phi, k=2, diversity=.5)
        self.assertEqual(diverse, [0, 3])
        self.assertGreater(intra_list_diversity(phi[diverse]), intra_list_diversity(phi[[0, 1]]))
        self.assertEqual(len(mmr_rerank(scores, phi, k=10)), 4)
        self.assertEqual(mmr_rerank(scores, phi, k=0), [])

    def test_session_stops_after_5_answers_or_8_offers(self):
        self.assertFalse(session_finished(4, 7))
        self.assertTrue(session_finished(5, 5))
        self.assertTrue(session_finished(2, 8))

    def test_rank_excludes_offered_items(self):
        profile = SessionProfile(np.zeros(SPACE.dim))
        profile.update(phi_of("parc"), phi_of("bar"))
        result = [item_id for item_id, _ in rank(profile, PHI, IDS, k=10, exclude={"parc", "bar"})]
        self.assertEqual(len(result), len(IDS) - 2)
        self.assertFalse({"parc", "bar"} & set(result))
        self.assertEqual(rank(profile, PHI, IDS, exclude=IDS), [])

    def test_rank_returns_ids_and_scores(self):
        profile = SessionProfile(np.zeros(SPACE.dim))
        profile.update(phi_of("parc"), phi_of("bar"))
        result = rank(profile, PHI, IDS, k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], "parc")
        self.assertAlmostEqual(result[0][1], float(profile.scores(phi_of("parc"))))


class CommonItemsTest(unittest.TestCase):
    def fixture(self, root):
        path = root / "common.sqlite"
        schema = files("swiparis").joinpath("common_schema.sql").read_text(encoding="utf-8")
        with closing(sqlite3.connect(path)) as db:
            db.executescript(schema)
            db.executemany("INSERT INTO categories VALUES (?,?)",
                           [("restauration", "R"), ("culture", "C"), ("autre", "A")])
            for item_id, source, price in (("foursquare:v1", "foursquare", "unknown"),
                                           ("foursquare:v2", "foursquare", "unknown"),
                                           ("paris_data:e1", "paris_data", "free")):
                db.execute("INSERT INTO items(item_id,source,source_id,title,description,lead_text,price_type,"
                           "price_detail,cover_credit,date_description,audience,quality_flags_json) "
                           "VALUES (?,?,?,'t','','',?,'','','','','[]')", (item_id, source, item_id.split(":")[1], price))
            db.executemany("INSERT INTO item_categories VALUES (?,?)", [
                ("foursquare:v1", "culture"), ("foursquare:v2", "restauration"),
                ("paris_data:e1", "culture"), ("paris_data:e1", "autre")])
            db.executemany("INSERT INTO locations(item_id,location_index,latitude,longitude) VALUES (?,?,?,?)", [
                ("foursquare:v1", 0, 48.86, 2.33), ("foursquare:v2", 0, 48.87, 2.36),
                ("paris_data:e1", 1, 48.88, 2.30), ("paris_data:e1", 0, 48.85, 2.35)])
            db.execute("INSERT INTO occurrences VALUES ('paris_data:e1','2026-11-20T18:30:00Z',"
                       "'2026-11-20T20:00:00Z','2026-11-20T19:30:00+01:00','x')")
            db.executemany("INSERT INTO visits VALUES (?,?,?,?,?)", [
                ("foursquare:train", "train", "foursquare:v1", "2012-04-03T10:00:00Z", 120),
                ("foursquare:test", "test", "foursquare:v1", "2012-04-03T22:00:00Z", 120)])
            db.commit()
        return path

    def test_items_and_leakage_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.fixture(Path(tmp))
            self.assertEqual(common_categories(path), ("autre", "culture", "restauration"))
            items = {item.item_id: item for item in common_items(path, users={"foursquare:train"})}
            self.assertEqual(sorted(items), ["foursquare:v1", "foursquare:v2", "paris_data:e1"])
            self.assertEqual(items["foursquare:v1"].time_profile, (0, 0, 1, 0))  # visite de test ignorée
            self.assertIsNone(items["foursquare:v2"].time_profile)
            self.assertEqual(items["paris_data:e1"].categories, ("autre", "culture"))
            self.assertEqual((items["paris_data:e1"].latitude, items["paris_data:e1"].longitude), (48.85, 2.35))
            self.assertEqual(items["paris_data:e1"].time_profile, (0, 0, 0, 1))
            everyone = {item.item_id: item for item in common_items(path, users=None)}
            self.assertEqual(everyone["foursquare:v1"].time_profile, (.5, 0, .5, 0))
            demo = common_items(path, users=None, sources=["paris_data"])
            self.assertEqual([item.item_id for item in demo], ["paris_data:e1"])
            with self.assertRaises(ValueError):
                common_items(path, users=None, sources=["yelp"])

    def test_popularity_uses_given_users_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoded = encode_common(self.fixture(Path(tmp)), users=None, popularity_users={"foursquare:train"})
            pop = encoded.space.blocks["pop"].start
            self.assertEqual(encoded.counts, {"foursquare:v1": 1})
            self.assertEqual(encoded.vector("foursquare:v1")[pop], 1.0)
            self.assertEqual(encoded.vector("foursquare:v2")[pop], 0.0)
            self.assertEqual(encoded.vector("paris_data:e1")[pop], 0.0)  # pas de visites Paris Data

    def test_encoded_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            encoded = encode_common(self.fixture(Path(tmp)), users=None)
            self.assertEqual(encoded.phi.shape, (3, encoded.space.dim))
            rows = encoded.rows(["paris_data:e1", "foursquare:v1"])
            np.testing.assert_allclose(rows[0], encoded.vector("paris_data:e1"))
            self.assertEqual(encoded.rows([]).shape, (0, encoded.space.dim))
            with self.assertRaises(KeyError):
                encoded.rows(["foursquare:inconnu"])

    @unittest.skipUnless(REAL_DB.exists(), "data/processed/common.sqlite absent (partagé hors Git)")
    def test_real_common_database(self):
        encoded = encode_common(REAL_DB, users=None)
        self.assertEqual(len(encoded.space.categories), 10)
        self.assertTrue(np.isfinite(encoded.phi).all())
        demo = encode_common(REAL_DB, users=None, sources=["paris_data"])
        price = demo.space.blocks["price"]
        self.assertFalse(demo.phi[:, price][:, -1].any())  # aucune annonce Paris Data au prix "unknown"
        profile = SessionProfile(np.zeros(demo.space.dim))
        first = next_pair(profile, demo.phi, demo.ids, rng=0)
        self.assertIsNotNone(first)
        profile.update(demo.vector(first[0]), demo.vector(first[1]))
        top = rank(profile, demo.phi, demo.ids, exclude=first)
        self.assertEqual(len(top), 10)
        self.assertFalse(set(first) & {item_id for item_id, _ in top})


if __name__ == "__main__":
    unittest.main()
