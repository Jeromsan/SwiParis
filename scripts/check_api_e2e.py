"""Parcours complet de l'API sur le vrai catalogue : preuve pour Q5 (modèle appelé, journal, latence).

    python scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models

Nécessite fastapi et httpx. Les journaux vont dans un fichier temporaire.
"""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    os.environ["SWIPARIS_MODELS_DIR"] = str(args.models_dir)
    from fastapi.testclient import TestClient
    from swiparis.api import create_app

    latencies = {}
    with tempfile.TemporaryDirectory() as tmp, TestClient(create_app(args.catalog, Path(tmp) / "logs.sqlite")) as client:
        def call(method, url, **kw):
            start = time.perf_counter()
            response = getattr(client, method)(url, **kw)
            latencies.setdefault(url.split("?")[0], []).append(round((time.perf_counter() - start) * 1000))
            assert response.status_code == 200, f"{url}: {response.status_code} {response.text}"
            return response.json()

        health = call("get", "/health")
        catalog = call("get", "/api/catalog?limit=1")
        session = call("post", "/api/sessions", json={
            "event_id": str(uuid.uuid4()), "consent": True, "constraints": {}, "catalog_version": catalog["catalog_version"],
            "model_version": health["model_version"], "data_origin": "test"})
        sid = session["session_id"]
        answered = 0
        for _ in range(8):
            nxt = call("get", f"/next_pair?session_id={sid}")
            if nxt["done"]:
                break
            pair = call("post", "/api/comparisons", json={"event_id": str(uuid.uuid4()), "session_id": sid,
                                                          "item_a": nxt["comparison"]["item_a"], "item_b": nxt["comparison"]["item_b"]})
            call("post", "/log_interaction", json={"event_id": str(uuid.uuid4()), "session_id": sid,
                                                   "comparison_id": pair["comparison_id"], "choice": "left"})
            answered += 1
        recs = call("post", "/recommendations", json={"event_id": str(uuid.uuid4()), "session_id": sid})
        call("post", "/api/item_action", json={"event_id": str(uuid.uuid4()), "session_id": sid,
                                               "impression_id": recs["items"][0]["impression_id"], "action": "save"})
        with closing(sqlite3.connect(str(Path(tmp) / "logs.sqlite"))) as db:
            stored = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                      for t in ("sessions", "comparisons", "recommendations_shown", "recommendation_actions")}
            stored_model = db.execute("SELECT model_version FROM sessions").fetchone()[0]
    print(json.dumps({"health_model_version": health["model_version"], "session_model_version": stored_model,
                      "answers": answered, "recommendations": len(recs["items"]),
                      "recommendations_model_version": recs["model_version"], "stored_rows": stored,
                      "latency_ms_max_per_route": {k: max(v) for k, v in latencies.items()}}, indent=2))


if __name__ == "__main__":
    main()
