"""Exercise installed catalog + logging HTTP API in an isolated temporary pilot DB."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from threading import Thread
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .demo import make_server


def check(catalog_path):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="swiparis-smoke-") as temp:
        log_path = Path(temp) / "test-only.sqlite"
        server = make_server(catalog_path, log_path, port=0)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, data=None, expected=200):
            req = Request(base + path, data=json.dumps(data).encode() if data is not None else None,
                          headers={"Content-Type": "application/json"})
            try:
                with urlopen(req, timeout=30) as response:
                    status, body = response.status, response.read()
            except HTTPError as exc:
                status, body = exc.code, exc.read()
            if status != expected:
                raise AssertionError((path, status, body[:500]))
            return json.loads(body)

        try:
            health = request("/health")
            map_config = request("/api/map-config")
            assert isinstance(map_config["api_key"], str) and isinstance(map_config["map_id"], str)
            catalog = request("/api/catalog?limit=5")
            ids = [e["item_id"] for e in catalog["items"]]
            if len(ids) < 4:
                raise ValueError("Smoke check needs at least 4 eligible demo activities")
            session_payload = dict(event_id="smoke-session", consent=True, constraints={},
                                   catalog_version=health["catalog_version"], model_version="smoke-test", data_origin="test")
            session = request("/api/sessions", session_payload)
            replay = request("/api/sessions", session_payload)
            assert replay["session_id"] == session["session_id"] and replay["replayed"]
            sid = session["session_id"]
            pair = request("/api/comparisons", dict(event_id="smoke-pair", session_id=sid, item_a=ids[0], item_b=ids[1]))
            pair2 = request("/api/comparisons", dict(event_id="smoke-pair", session_id=sid, item_a=ids[0], item_b=ids[1]))
            assert pair["left_item_id"] == pair2["left_item_id"]
            answer = dict(event_id="smoke-answer", session_id=sid, comparison_id=pair["comparison_id"], choice="left")
            request("/log_interaction", answer)
            assert request("/log_interaction", answer)["replayed"]
            request("/log_interaction", dict(answer, choice="right"), 409)
            recs = request("/api/recommendations_shown", dict(event_id="smoke-top10", session_id=sid, item_ids=ids[2:4]))
            impression = recs["impressions"][0]["impression_id"]
            request("/api/item_action", dict(event_id="smoke-open", session_id=sid, impression_id=impression, action="open"))
            request("/api/item_action", dict(event_id="smoke-save", session_id=sid, impression_id=impression, action="save"))
            request("/api/item_action", dict(event_id="smoke-negative", session_id=sid, impression_id=impression, action="dislike"), 400)
            with urlopen(base, timeout=30) as response:
                assert b"SwiParis" in response.read()
            with closing(sqlite3.connect(log_path)) as db:
                assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
                assert db.execute("SELECT count(*) FROM comparisons").fetchone()[0] == 1
                assert db.execute("SELECT count(*) FROM recommendations_shown").fetchone()[0] == 2
                assert db.execute("SELECT count(*) FROM recommendation_actions").fetchone()[0] == 2
                assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert not db.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            server.shutdown()
            thread.join(timeout=10)
            server.server_close()
    return dict(status="ok", seconds=round(time.perf_counter() - started, 3), catalog_version=health["catalog_version"],
                checks=["catalog", "map_page", "map_configuration", "consent", "session", "randomized_pair_persistence", "choice", "idempotency", "conflict", "recommendation_impressions", "open_save", "no_inferred_negative", "sqlite_integrity"],
                google_maps_live_tested=False, pilot_database_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/common.sqlite"))
    args = parser.parse_args()
    print(json.dumps(check(args.catalog), indent=2))


if __name__ == "__main__":
    main()
