"""Parcours complet sur le vrai catalogue : API live, prior, journal et latence.

    python scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models

Nécessite uvicorn. L'API et son journal temporaire sont arrêtés/supprimés à la fin.
"""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--startup-timeout", type=float, default=45.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog = (root / args.catalog).resolve()
    models = (root / args.models_dir).resolve()
    if not catalog.is_file():
        parser.error(f"catalogue introuvable : {catalog}")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    latencies = {}

    with tempfile.TemporaryDirectory(prefix="swiparis-e2e-") as tmp:
        tmp_path = Path(tmp)
        env = os.environ.copy()
        env["SWIPARIS_CATALOG"] = str(catalog)
        env["SWIPARIS_LOGS"] = str(tmp_path / "logs.sqlite")
        env["SWIPARIS_MODELS_DIR"] = str(models)
        src = str(root / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        server_log = (tmp_path / "uvicorn.log").open("w", encoding="utf-8")
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "swiparis.api:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=server_log,
        )

        def call(method, path, payload=None):
            body = None if payload is None else json.dumps(payload).encode("utf-8")
            request = Request(base_url + path, data=body,
                              headers={"Content-Type": "application/json"}, method=method.upper())
            start = time.perf_counter()
            try:
                with urlopen(request, timeout=45) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    status = response.status
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise AssertionError(f"{method} {path}: {exc.code} {detail}") from exc
            elapsed = round((time.perf_counter() - start) * 1000)
            latencies.setdefault(path.split("?")[0], []).append(elapsed)
            print(f"{method.upper()} {path.split('?')[0]}: {status}, {elapsed} ms", flush=True)
            return data

        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if server.poll() is not None:
                    server_log.flush()
                    raise RuntimeError(f"Uvicorn s'est arrêté au démarrage :\n{(tmp_path / 'uvicorn.log').read_text(encoding='utf-8')}")
                try:
                    with urlopen(base_url + "/health", timeout=1):
                        break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"API non prête après {args.startup_timeout:g} s")
                    time.sleep(0.25)

            print("API prête; exécution du parcours réel...", flush=True)
            health = call("GET", "/health")
            catalog_info = call("GET", "/api/catalog?limit=1")
            window = catalog_info["snapshot_window"]
            constraints = {"start_utc": window["as_of_utc"],
                           "end_utc": window["window_end_utc"]}
            session = call("POST", "/api/sessions", {
                "event_id": str(uuid.uuid4()), "consent": True, "constraints": constraints,
                "catalog_version": catalog_info["catalog_version"],
                "model_version": health["model_version"], "data_origin": "test",
            })
            session_id = session["session_id"]
            answered = 0
            for _ in range(15):
                pair_offer = call("GET", f"/next_pair?session_id={session_id}")
                if pair_offer["done"]:
                    break
                comparison = call("POST", "/api/comparisons", {
                    "event_id": str(uuid.uuid4()), "session_id": session_id,
                    "item_a": pair_offer["comparison"]["item_a"],
                    "item_b": pair_offer["comparison"]["item_b"],
                })
                call("POST", "/log_interaction", {
                    "event_id": str(uuid.uuid4()), "session_id": session_id,
                    "comparison_id": comparison["comparison_id"], "choice": "left",
                })
                answered += 1
            if answered != 10:
                raise AssertionError(f"Expected 10 personalized answers, got {answered}")
            recommendations = call("POST", "/recommendations", {
                "event_id": str(uuid.uuid4()), "session_id": session_id,
            })
            call("POST", "/api/item_action", {
                "event_id": str(uuid.uuid4()), "session_id": session_id,
                "impression_id": recommendations["items"][0]["impression_id"], "action": "save",
            })
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            server_log.close()

        with closing(sqlite3.connect(str(tmp_path / "logs.sqlite"))) as db:
            stored = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                      for table in ("sessions", "comparisons", "recommendations_shown", "recommendation_actions")}
            stored_model = db.execute("SELECT model_version FROM sessions").fetchone()[0]

    assert health["model_version"] == stored_model == recommendations["model_version"], "prior version differs between API and logs"
    print(json.dumps({
        "health_model_version": health["model_version"], "session_model_version": stored_model,
        "answers": answered, "recommendations": len(recommendations["items"]),
        "recommendations_model_version": recommendations["model_version"], "stored_rows": stored,
        "latency_ms_max_per_route": {route: max(values) for route, values in latencies.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
