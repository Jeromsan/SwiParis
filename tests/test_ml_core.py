from contextlib import closing
from importlib.resources import files
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from swiparis.ml.catalog import foursquare_items, paris_data_items
from swiparis.ml.features import FeatureSpace, Item, foursquare_local_hour, source_local_hour, time_profile
from swiparis.ml.model import SessionProfile
from swiparis.ml.selection import (acquisition_matrix, binary_entropy, intra_list_diversity,
                                   mmr_rerank, next_pair, rank)

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

    def test_rank_returns_ids_and_scores(self):
        profile = SessionProfile(np.zeros(SPACE.dim))
        profile.update(phi_of("parc"), phi_of("bar"))
        result = rank(profile, PHI, IDS, k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], "parc")
        self.assertAlmostEqual(result[0][1], float(profile.scores(phi_of("parc"))))


class CatalogTest(unittest.TestCase):
    def test_foursquare_time_profile_uses_only_given_users(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "foursquare.sqlite"
            with closing(sqlite3.connect(path)) as db:
                db.executescript("""
                    CREATE TABLE venues(venue_id TEXT PRIMARY KEY, category TEXT NOT NULL,
                        latitude REAL NOT NULL, longitude REAL NOT NULL, country TEXT NOT NULL);
                    CREATE TABLE checkins(user_id TEXT NOT NULL, venue_id TEXT NOT NULL REFERENCES venues(venue_id),
                        utc_time TEXT NOT NULL, timezone_offset INTEGER NOT NULL,
                        PRIMARY KEY(user_id,utc_time,venue_id)) WITHOUT ROWID;
                    INSERT INTO venues VALUES ('v1','Museum',48.86,2.33,'FR'), ('v2','Bar',48.87,2.36,'FR');
                    INSERT INTO checkins VALUES ('train','v1','2012-04-03T10:00:00Z',120),
                                                ('test','v1','2012-04-03T22:00:00Z',120);
                """)
                db.commit()
            mapping = {"Museum": "culture", "Bar": ["nightlife"]}
            items = {item.item_id: item for item in foursquare_items(path, mapping, users={"train"})}
            self.assertEqual(items["v1"].time_profile, (0, 0, 1, 0))
            self.assertIsNone(items["v2"].time_profile)
            self.assertEqual(items["v2"].categories, ("nightlife",))
            self.assertEqual(items["v1"].price_type, "unknown")
            everyone = {item.item_id: item for item in foursquare_items(path, mapping, users=None)}
            self.assertEqual(everyone["v1"].time_profile, (.5, 0, .5, 0))
            with self.assertRaises(ValueError):
                foursquare_items(path, {"Museum": "culture"}, users=None)

    def test_paris_data_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paris.sqlite"
            schema = files("swiparis").joinpath("paris_data_schema.sql").read_text(encoding="utf-8")
            with closing(sqlite3.connect(path)) as db:
                db.executescript(schema)
                db.execute("INSERT INTO snapshots VALUES ('s','1','u','h','a','b','{}')")
                for event_id, price in (("e1", "free"), ("e2", "paid")):
                    db.execute("INSERT INTO events VALUES (?,'s',?,NULL,'u','t','','',?,NULL,NULL,'','',"
                               "NULL,NULL,'occurrences',NULL,NULL,'',NULL,'','','[]')", (event_id, event_id, price))
                db.executemany("INSERT INTO event_locations VALUES (?,?,?,?,?,?,?,?,?)", [
                    ("e1", 1, "B", "", "75011", "Paris", 11, 48.86, 2.38),
                    ("e1", 0, "A", "", "75004", "Paris", 4, 48.85, 2.35)])
                db.executemany("INSERT INTO event_occurrences VALUES (?,?,?,?,?)", [
                    ("e1", "2026-11-20T18:30:00Z", "2026-11-20T20:00:00Z", "2026-11-20T19:30:00+01:00", "x"),
                    ("e1", "2026-11-21T09:00:00Z", "2026-11-21T11:00:00Z", "2026-11-21T10:00:00+01:00", "x")])
                db.executemany("INSERT INTO event_categories VALUES (?,?)",
                               [("e1", "exposition"), ("e1", "visite"), ("e2", "musique")])
                db.commit()
            mapping = {"exposition": "culture", "visite": "culture", "musique": "spectacle"}
            items = {item.item_id: item for item in paris_data_items(path, mapping)}
            self.assertEqual(items["e1"].categories, ("culture",))
            self.assertEqual((items["e1"].latitude, items["e1"].longitude), (48.85, 2.35))
            self.assertEqual(items["e1"].time_profile, (0, .5, 0, .5))
            self.assertEqual(items["e2"].price_type, "paid")
            self.assertIsNone(items["e2"].latitude)
            SPACE.encode_many(list(items.values()))
            with self.assertRaises(ValueError):
                paris_data_items(path, {"exposition": "culture"})


if __name__ == "__main__":
    unittest.main()
