"""Parcours de l'API FastAPI sur une petite base de test, le prior et les personas du dépôt."""
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import uuid

try:
    from fastapi.testclient import TestClient
    import numpy as np
except (ImportError, RuntimeError):  # dépendances optionnelles [systems] (TestClient exige httpx)
    raise unittest.SkipTest('fastapi/httpx absents : installer avec pip install -e ".[systems,ml]"')

from common_fixture import build_common
from swiparis.api import create_app
from swiparis.ml import policy
from swiparis.ml.items import common_items
from swiparis.serving import load_model

ROOT = Path(__file__).resolve().parents[1]
SLOT = {"start_utc": "2026-11-29T12:00:00Z", "end_utc": "2026-11-29T17:00:00Z"}  # les 40 activités du dimanche


class ApiTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.catalog_path = build_common(Path(temp.name) / "common.sqlite", extra=40)
        env = patch.dict(os.environ, {"SWIPARIS_MODELS_DIR": str(ROOT / "models"),
                                      "SWIPARIS_PERSONAS": str(ROOT / "models" / "personas.json"),
                                      "SWIPARIS_SESSION_ETA": str(policy.DEFAULT_ETA)})
        env.start()
        self.addCleanup(env.stop)
        self.logs_path = Path(temp.name) / "logs.sqlite"
        self.client = TestClient(create_app(self.catalog_path, self.logs_path))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.health = self.client.get("/health").json()
        catalog = self.client.app.state.catalog
        self.categories = {r["item_id"]: r["categories"] for r in catalog.list_items({}, limit=None)["items"]}

    def post(self, route, **body):
        return self.client.post(route, json={"event_id": str(uuid.uuid4()), **body})

    def session(self, constraints=SLOT):
        response = self.post("/api/sessions", consent=True, constraints=constraints,
                             catalog_version=self.health["catalog_version"],
                             model_version=self.health["model_version"], data_origin="test")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session_id"]

    def play(self, session_id, prefer="spectacle"):
        """Répond comme quelqu'un qui préfère `prefer` ; renvoie le dernier état et les paires montrées."""
        offered = []
        while True:
            step = self.client.get("/next_pair", params={"session_id": session_id}).json()
            if step["done"]:
                return step, offered
            pair = self.post("/api/comparisons", session_id=session_id, **step["comparison"]).json()
            left, right = (prefer in pair[side]["categories"] for side in ("left", "right"))
            choice = "left" if left and not right else "right" if right and not left else "skip"
            answer = self.post("/log_interaction", session_id=session_id, comparison_id=pair["comparison_id"],
                               choice=choice)
            self.assertEqual(answer.status_code, 200, answer.text)
            offered.append((pair["left_item_id"], pair["right_item_id"], choice))

    def test_health_reports_the_versioned_prior_eta_and_personas(self):
        version = self.health["model_version"]
        self.assertTrue(version.startswith(load_model(ROOT / "models").version + "+session-v4"), version)
        self.assertIn(f"+eta{policy.DEFAULT_ETA:g}", version)
        self.assertTrue(version.endswith("+personas6"), version)

    def test_eta_and_personas_can_be_changed_without_code(self):
        with patch.dict(os.environ, {"SWIPARIS_SESSION_ETA": "0.5", "SWIPARIS_PERSONAS": "off"}):
            with TestClient(create_app(self.catalog_path, self.logs_path)) as other:
                version = other.get("/health").json()["model_version"]
        self.assertIn("+eta0.5", version)
        self.assertNotIn("personas", version)

    def test_vectors_match_the_offline_encoding(self):
        model = load_model(ROOT / "models")
        items = common_items(self.catalog_path, users=None, sources=["paris_data"])
        vectors = self.client.app.state.vectors
        for item, row in zip(items, model.encode(items)):
            np.testing.assert_allclose(vectors[item.item_id], row)

    def test_full_session_and_recommendations(self):
        sid = self.session()
        step, offered = self.play(sid)
        answered = [o for o in offered if o[2] != "skip"]
        self.assertIn(step["stop_reason"], {"max_answers", "max_offers", "no_informative_pair"})
        self.assertLessEqual(len(answered), policy.MAX_ANSWERS)
        self.assertLessEqual(len(offered), policy.MAX_OFFERS)
        event = str(uuid.uuid4())
        first = self.client.post("/recommendations", json={"event_id": event, "session_id": sid})
        self.assertEqual(first.status_code, 200, first.text)
        body = first.json()
        items = body["items"]
        self.assertEqual(len(items), policy.RECOMMENDATIONS)
        rejected = {(left if choice == "right" else right) for left, right, choice in answered}
        self.assertFalse(rejected & {item["item_id"] for item in items})  # jamais une sortie rejetée
        eligible = {r["item_id"] for r in self.client.app.state.catalog.list_items(SLOT, limit=None)["items"]}
        self.assertTrue({item["item_id"] for item in items} <= eligible)
        top = body["persona"]["top"]
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0]["share"], top[1]["share"])
        self.assertTrue(top[0]["name"] and top[0]["description"])
        replay = self.client.post("/recommendations", json={"event_id": event, "session_id": sid})
        self.assertEqual([i["item_id"] for i in replay.json()["items"]], [i["item_id"] for i in items])
        for action in ("open", "save"):
            response = self.post("/api/item_action", session_id=sid, impression_id=items[0]["impression_id"],
                                 action=action)
            self.assertEqual(response.status_code, 200, response.text)

    def test_choices_change_the_recommendations_and_the_persona(self):
        results = []
        for prefer in ("spectacle", "culture"):
            sid = self.session(constraints={})
            self.play(sid, prefer)
            body = self.post("/recommendations", session_id=sid).json()
            results.append(([item["item_id"] for item in body["items"]], body["persona"]["top"][0]["share"]))
        self.assertNotEqual(results[0][0], results[1][0])

    def recommended(self, sid):
        self.play(sid)
        response = self.post("/recommendations", session_id=sid)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["items"]

    def saved_ids(self, owner):
        response = self.client.get("/api/saved", params={"owner_id": owner})
        self.assertEqual(response.status_code, 200, response.text)
        return [item["item_id"] for item in response.json()["items"]]

    def test_saved_selection_is_persisted_listed_removed_and_restored(self):
        owner = uuid.uuid4().hex
        sid = self.session()
        step = self.client.get("/next_pair", params={"session_id": sid}).json()
        pair = self.post("/api/comparisons", session_id=sid, **step["comparison"]).json()
        from_pair = pair["left_item_id"]  # gardée pendant un duel, sans recommandation
        saved = self.post("/api/saved", owner_id=owner, session_id=sid, items=[{"item_id": from_pair}])
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["count"], 1)
        items = self.recommended(sid)
        # Une sortie choisie pendant les duels peut revenir dans les recommandations : on garde les autres.
        several = [{"item_id": i["item_id"], "impression_id": i["impression_id"]} for i in items
                   if i["item_id"] != from_pair]
        n = len(several)
        self.assertGreaterEqual(n, 2)
        saved = self.post("/api/saved", owner_id=owner, session_id=sid, items=several).json()
        self.assertEqual((saved["count"], {e["status"] for e in saved["items"]}), (1 + n, {"saved"}))
        again = self.post("/api/saved", owner_id=owner, session_id=sid, items=several[:1]).json()
        self.assertEqual((again["count"], again["items"][0]["status"]), (1 + n, "already_saved"))  # pas de doublon

        listed = self.client.get("/api/saved", params={"owner_id": owner}).json()
        self.assertEqual(listed["count"], 1 + n)
        first = listed["items"][-1]
        self.assertEqual((first["item_id"], first["available"], first["slot"]), (from_pair, True, SLOT))
        self.assertTrue(first["title"] and first["next_occurrence"]["start_utc"] and first["source_url"])
        self.assertEqual(self.saved_ids(uuid.uuid4().hex), [])  # un autre carnet ne voit rien

        removed = self.post("/api/saved/remove", owner_id=owner, item_ids=[from_pair, several[0]["item_id"]]).json()
        self.assertEqual((sorted(removed["removed"]), removed["count"]),
                         (sorted([from_pair, several[0]["item_id"]]), n - 1))
        self.assertNotIn(from_pair, self.saved_ids(owner))
        restored = self.post("/api/saved/restore", owner_id=owner, item_ids=[from_pair]).json()
        self.assertEqual((restored["restored"], restored["count"]), ([from_pair], n))
        re_saved = self.post("/api/saved", owner_id=owner, session_id=sid, items=several[:1]).json()
        self.assertEqual(re_saved["items"][0]["status"], "restored")

        # Persistance : une nouvelle instance de l'API (redémarrage) retrouve la même sélection.
        with TestClient(create_app(self.catalog_path, self.logs_path)) as other:
            ids = [i["item_id"] for i in other.get("/api/saved", params={"owner_id": owner}).json()["items"]]
        self.assertEqual(sorted(ids), sorted([from_pair] + [e["item_id"] for e in several]))
        # Le signal « save » de l'évaluation est journalisé une seule fois par recommandation.
        with closing(sqlite3.connect(self.logs_path)) as db:
            counts = db.execute("SELECT impression_id,count(*) FROM recommendation_actions "
                                "WHERE action='save' GROUP BY impression_id").fetchall()
        self.assertEqual(sorted(c for _, c in counts), [1] * n)

    def test_saved_selection_rejects_bad_input(self):
        owner = uuid.uuid4().hex
        sid = self.session()
        items = self.recommended(sid)
        other_sid = self.session()
        cases = [
            dict(owner_id="court", session_id=sid, items=[{"item_id": items[0]["item_id"]}]),
            dict(owner_id=owner, session_id=sid, items=[{"item_id": "paris_data:inconnue"}]),
            dict(owner_id=owner, session_id=sid, items=[{"item_id": "paris_data:diner"}]),  # hors créneau
            dict(owner_id=owner, session_id=other_sid,
                 items=[{"item_id": items[0]["item_id"], "impression_id": items[0]["impression_id"]}]),
            dict(owner_id=owner, session_id=sid,
                 items=[{"item_id": items[1]["item_id"], "impression_id": items[0]["impression_id"]}]),
            dict(owner_id=owner, session_id=sid, items=[{"item_id": items[0]["item_id"]}] * 2),
            dict(owner_id=owner, session_id=sid, items=[]),
        ]
        for body in cases:
            self.assertEqual(self.post("/api/saved", **body).status_code, 400, body)
        self.assertEqual(self.saved_ids(owner), [])
        self.assertEqual(self.client.get("/api/saved", params={"owner_id": "x"}).status_code, 400)
        self.assertEqual(self.post("/api/saved/remove", owner_id=owner, item_ids=["a", "a"]).status_code, 400)
        nothing = self.post("/api/saved/remove", owner_id=owner, item_ids=["paris_data:expo"]).json()
        self.assertEqual((nothing["removed"], nothing["count"]), ([], 0))

    def test_open_link_logs_then_redirects_to_the_source(self):
        sid = self.session()
        item = self.recommended(sid)[0]
        response = self.client.get(f"/go/{item['impression_id']}", follow_redirects=False)
        self.assertEqual((response.status_code, response.headers["location"]), (303, item["source_url"]))
        with closing(sqlite3.connect(self.logs_path)) as db:
            actions = db.execute("SELECT action FROM recommendation_actions WHERE impression_id=?",
                                 (item["impression_id"],)).fetchall()
        self.assertEqual(actions, [("open",)])
        self.assertEqual(self.client.get("/go/inconnue", follow_redirects=False).status_code, 404)

    def test_eta_comparison_during_the_pilot_and_its_report(self):
        from swiparis.ml.pilot_eta import report
        with patch.dict(os.environ, {"SWIPARIS_SESSION_ETA_AB": "0.2,0.5"}):
            with TestClient(create_app(self.catalog_path, self.logs_path)) as other:
                version = other.get("/health").json()["model_version"]
                self.assertIn("+etaAB0.2_0.5", version)
                etas, sessions = set(), []
                for _ in range(20):
                    sid = other.post("/api/sessions", json={
                        "event_id": str(uuid.uuid4()), "consent": True, "constraints": SLOT, "data_origin": "test",
                        "catalog_version": self.health["catalog_version"], "model_version": version}).json()["session_id"]
                    etas.add(other.get("/next_pair", params={"session_id": sid}).json()["session_eta"])
                    sessions.append(sid)
                self.assertEqual(etas, {0.2, 0.5})
                items = other.post("/recommendations", json={"event_id": str(uuid.uuid4()),
                                                              "session_id": sessions[0]}).json()["items"]
                for action in ("open", "save"):
                    other.post("/api/item_action", json={"event_id": str(uuid.uuid4()), "session_id": sessions[0],
                                                         "impression_id": items[0]["impression_id"], "action": action})
        result = report(self.logs_path, include_test=True)
        self.assertEqual(result["sessions"], 20)
        self.assertEqual(sum(arm["sessions"] for arm in result["arms"].values()), 20)
        rated = [arm for arm in result["arms"].values() if arm["recommendations_shown"]]
        self.assertEqual(len(rated), 1)
        self.assertAlmostEqual(rated[0]["save_rate"], 1 / len(items))
        self.assertAlmostEqual(rated[0]["open_rate"], 1 / len(items))
        self.assertEqual(report(self.logs_path)["sessions"], 0)  # séances de test exclues par défaut

    def test_my_selection_is_the_starting_point_of_the_personas(self):
        catalog = self.client.app.state.catalog
        rows = catalog.list_items(SLOT, limit=None)["items"]
        by_category = {}
        for row in rows:
            by_category.setdefault(row["categories"][0], []).append(row["item_id"])
        tastes = {}
        for category in ("culture", "spectacle"):
            owner = uuid.uuid4().hex
            sid = self.session()
            saved = self.post("/api/saved", owner_id=owner, session_id=sid,
                              items=[{"item_id": i} for i in by_category[category][:3]])
            self.assertEqual(saved.status_code, 200, saved.text)
            fresh = self.session()
            body = self.post("/recommendations", session_id=fresh, owner_id=owner).json()
            self.assertEqual(body["persona"]["from_selection"], 3)
            tastes[category] = [p["share"] for p in sorted(body["persona"]["top"], key=lambda p: p["index"])]
            tastes[category + "_top"] = body["persona"]["top"][0]["name"]
        self.assertNotEqual(tastes["culture"], tastes["spectacle"])
        lone = uuid.uuid4().hex
        sid = self.session()
        self.post("/api/saved", owner_id=lone, session_id=sid, items=[{"item_id": by_category["culture"][0]}])
        body = self.post("/recommendations", session_id=self.session(), owner_id=lone).json()
        self.assertEqual(body["persona"]["from_selection"], 0)  # une seule sortie : pas assez pour démarrer

    def test_invalid_requests(self):
        self.assertEqual(self.client.get("/next_pair", params={"session_id": "inconnue"}).status_code, 400)
        sid = self.session()
        step = self.client.get("/next_pair", params={"session_id": sid}).json()
        pair = self.post("/api/comparisons", session_id=sid, **step["comparison"]).json()
        bad = self.post("/log_interaction", session_id=sid, comparison_id=pair["comparison_id"], choice="both")
        self.assertEqual(bad.status_code, 400)
        wrong_model = self.post("/api/sessions", consent=True, constraints={},
                                catalog_version=self.health["catalog_version"], model_version="autre",
                                data_origin="test")
        self.assertEqual(wrong_model.status_code, 409)
        self.assertEqual(self.client.get("/health", headers={"host": "example.com"}).status_code, 403)
        self.assertEqual(self.post("/api/sessions", consent="oui").status_code, 400)


if __name__ == "__main__":
    unittest.main()
