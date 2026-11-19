"""Small localhost-only catalog/map demo and pilot logging adapter. No model."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlsplit

from .catalog import Catalog
from .logging_store import Conflict, LogStore


def make_server(catalog_path, log_path, port=8000):
    catalog = Catalog(catalog_path)
    logs = LogStore(log_path, catalog)
    routes = {"/api/sessions": logs.create_session, "/api/comparisons": logs.offer_comparison,
              "/log_interaction": logs.answer, "/api/recommendations_shown": logs.show_recommendations,
              "/api/item_action": logs.item_action}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not write payloads, session IDs or keys to an access log.

        def send(self, status, value, content_type="application/json; charset=utf-8"):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host")
            if host not in allowed:
                self.send(403, {"error": "Localhost only"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in allowed}:
                self.send(403, {"error": "Cross-origin requests are not allowed"})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/health":
                    self.send(200, dict(status="ok", catalog_version=catalog.version))
                elif parsed.path == "/api/map-config":
                    # Maps JS uses a browser key. It is never stored in SQLite or a bundle.
                    self.send(200, dict(api_key=os.environ.get("GOOGLE_MAPS_API_KEY", ""),
                                        map_id=os.environ.get("GOOGLE_MAPS_MAP_ID", "DEMO_MAP_ID")))
                elif parsed.path == "/api/catalog":
                    query = parse_qs(parsed.query)
                    if set(query) - {"start_utc", "end_utc", "arrondissement", "budget_eur", "category", "limit", "offset"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError("Unknown or repeated query parameter")
                    params = {k: v[0] for k, v in query.items()}
                    limit, offset = int(params.pop("limit", 200)), int(params.pop("offset", 0))
                    if "arrondissement" in params:
                        params["arrondissement"] = int(params["arrondissement"])
                    if "budget_eur" in params:
                        params["budget_eur"] = float(params["budget_eur"])
                    self.send(200, catalog.list_items(params, limit, offset))
                elif parsed.path in ("/", "/app.js", "/app.css"):
                    name = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css"}[parsed.path]
                    content_type = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8"}[name]
                    self.send(200, files("swiparis").joinpath("web", name).read_bytes(), content_type)
                else:
                    self.send(404, {"error": "Unknown route"})
            except (ValueError, sqlite3.Error) as exc:
                self.send(400, {"error": str(exc)})

        def do_POST(self):
            if not self.local_request():
                return
            route = routes.get(urlsplit(self.path).path)
            if route is None:
                self.send(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    self.close_connection = True
                    self.send(413, {"error": "JSON body must be between 1 and 65536 bytes"})
                    return
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                self.send(200, route(data))
            except Conflict as exc:
                self.send(409, {"error": str(exc)})
            except (ValueError, TypeError, UnicodeError, sqlite3.IntegrityError) as exc:
                self.send(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.catalog, server.logs = catalog, logs
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--logs", type=Path, default=Path("data/pilot/interactions.sqlite"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        server = make_server(args.catalog, args.logs, args.port)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Demo failed: {exc}\nRun swiparis-prepare or use the offline demo bundle.\n")
    print(f"SwiParis: http://127.0.0.1:{server.server_port} — Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
