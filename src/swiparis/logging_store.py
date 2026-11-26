"""Validated, idempotent pilot logging; callable directly from a future FastAPI route."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import secrets
import sqlite3
import uuid

from .catalog import normalize_constraints


class Conflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def identifier(value, field):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f"Invalid {field}")
    return value


class LogStore:
    def __init__(self, path, catalog):
        self.path, self.catalog = Path(path), catalog
        if self.path.resolve() == Path(catalog.path).resolve():
            raise ValueError("Pilot logs must be separate from catalog/training databases")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            existing = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
            allowed = {"sessions", "comparisons", "recommendations_shown", "recommendation_actions", "request_events"}
            if existing - allowed or db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise ValueError("Refusing to use a source/catalog database or unknown schema as pilot logs")
            db.executescript(files("swiparis").joinpath("logs_schema.sql").read_text(encoding="utf-8"))

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _request(self, operation, data, allowed, write):
        if not isinstance(data, dict) or set(data) - set(allowed) - {"event_id"}:
            raise ValueError("Unknown JSON field")
        event_id = identifier(data.get("event_id"), "event_id")
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM request_events WHERE event_id=?", (event_id,)).fetchone()
            if old:
                if old["operation"] != operation or old["payload_sha256"] != digest:
                    raise Conflict("event_id already used with different content")
                return dict(json.loads(old["result_json"]), replayed=True)
            result = write(db)
            db.execute("INSERT INTO request_events VALUES (?,?,?,?,?)", (event_id, operation, digest, json.dumps(result), now()))
            return dict(result, replayed=False)

    def _session(self, db, session_id):
        row = db.execute("SELECT * FROM sessions WHERE session_id=?", (identifier(session_id, "session_id"),)).fetchone()
        if row is None:
            raise ValueError("Unknown session")
        if row["catalog_version"] != self.catalog.version:
            raise Conflict("Session belongs to another catalog version; create a new session")
        return row

    def create_session(self, data):
        def write(db):
            if data.get("consent") is not True:
                raise ValueError("Explicit consent is required before pilot logging")
            if data.get("catalog_version") != self.catalog.version:
                raise Conflict("catalog_version does not match the loaded catalog")
            model = identifier(data.get("model_version"), "model_version")
            origin = data.get("data_origin", "pilot")
            if origin not in ("pilot", "test"):
                raise ValueError("data_origin must be pilot or test")
            constraints = normalize_constraints(data.get("constraints", {}))
            if constraints.get("category") and constraints["category"] not in self.catalog.categories:
                raise ValueError("Unknown common category")
            session_id = str(uuid.uuid4())
            db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)", (session_id, now(), json.dumps(constraints), self.catalog.version, model, 1, origin))
            return dict(session_id=session_id, catalog_version=self.catalog.version, model_version=model)
        return self._request("session", data, {"consent", "constraints", "catalog_version", "model_version", "data_origin"}, write)

    def offer_comparison(self, data):
        def write(db):
            session = self._session(db, data.get("session_id"))
            a, b = identifier(data.get("item_a"), "item_a"), identifier(data.get("item_b"), "item_b")
            if a == b:
                raise ValueError("A comparison needs two different activities")
            self.catalog.require_eligible([a, b], json.loads(session["constraints_json"]))
            if secrets.randbelow(2):
                a, b = b, a
            comparison = str(uuid.uuid4())
            db.execute("INSERT INTO comparisons(comparison_id,session_id,left_item_id,right_item_id,offered_at_utc) VALUES (?,?,?,?,?)", (comparison, session["session_id"], a, b, now()))
            return dict(comparison_id=comparison, left_item_id=a, right_item_id=b)
        return self._request("comparison", data, {"session_id", "item_a", "item_b"}, write)

    def answer(self, data):
        def write(db):
            session = self._session(db, data.get("session_id"))
            row = db.execute("SELECT * FROM comparisons WHERE comparison_id=? AND session_id=?", (identifier(data.get("comparison_id"), "comparison_id"), session["session_id"])).fetchone()
            if row is None:
                raise ValueError("Comparison does not belong to session")
            if row["choice"] is not None:
                raise Conflict("Comparison already answered")
            choice = data.get("choice")
            if choice not in ("left", "right", "skip"):
                raise ValueError("choice must be left, right or skip")
            db.execute("UPDATE comparisons SET choice=?,answered_at_utc=?,answer_event_id=? WHERE comparison_id=?", (choice, now(), data["event_id"], row["comparison_id"]))
            chosen = row["left_item_id"] if choice == "left" else row["right_item_id"] if choice == "right" else None
            return dict(comparison_id=row["comparison_id"], choice=choice, chosen_item_id=chosen)
        return self._request("answer", data, {"session_id", "comparison_id", "choice"}, write)

    def show_recommendations(self, data):
        def write(db):
            session = self._session(db, data.get("session_id"))
            items = data.get("item_ids")
            if not isinstance(items, list) or not 1 <= len(items) <= 10 or any(not isinstance(i, str) for i in items) or len(set(items)) != len(items):
                raise ValueError("item_ids must contain 1 to 10 distinct activity IDs")
            self.catalog.require_eligible(items, json.loads(session["constraints_json"]))
            rejected = set()
            for comparison in db.execute(
                    "SELECT left_item_id,right_item_id,choice FROM comparisons "
                    "WHERE session_id=? AND choice IN ('left','right')", (session["session_id"],)):
                rejected.add(comparison["right_item_id"] if comparison["choice"] == "left"
                             else comparison["left_item_id"])
            if rejected & set(items):
                raise ValueError("Recommendations must not repeat explicitly rejected activities")
            shown = []
            at = now()
            for rank, item in enumerate(items, 1):
                impression = str(uuid.uuid4())
                db.execute("INSERT INTO recommendations_shown VALUES (?,?,?,?,?,?)", (impression, data["event_id"], session["session_id"], item, rank, at))
                shown.append(dict(impression_id=impression, item_id=item, rank=rank))
            return dict(impressions=shown)
        return self._request("recommendations", data, {"session_id", "item_ids"}, write)

    def item_action(self, data):
        def write(db):
            session = self._session(db, data.get("session_id"))
            impression = identifier(data.get("impression_id"), "impression_id")
            if db.execute("SELECT 1 FROM recommendations_shown WHERE impression_id=? AND session_id=?", (impression, session["session_id"])).fetchone() is None:
                raise ValueError("Impression does not belong to session")
            if data.get("action") not in ("open", "save"):
                raise ValueError("action must be open or save; unclicked items remain unlabeled")
            db.execute("INSERT INTO recommendation_actions VALUES (?,?,?,?)", (data["event_id"], impression, data["action"], now()))
            return dict(impression_id=impression, action=data["action"])
        return self._request("action", data, {"session_id", "impression_id", "action"}, write)
