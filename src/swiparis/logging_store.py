"""Validated, idempotent pilot logging; callable directly from a future FastAPI route."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
import secrets
import sqlite3
import uuid

from .catalog import normalize_constraints


class Conflict(ValueError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


OWNER_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")


def owner_id(value):
    """Identifiant du carnet « Ma sélection » : aléatoire, 8 à 64 caractères sûrs."""
    if not isinstance(value, str) or not OWNER_ID.fullmatch(value):
        raise ValueError("Invalid owner_id")
    return value


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
            allowed = {"sessions", "comparisons", "recommendations_shown", "recommendation_actions", "request_events",
                       "saved_items"}
            if existing - allowed or db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1, 2):
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

    # --- « Ma sélection » -------------------------------------------------------------

    def save_items(self, data):
        """Ajoute des activités à la sélection d'un carnet (plusieurs à la fois).

        Une activité déjà gardée n'est pas dupliquée ; une activité retirée puis
        re-gardée est restaurée. Si l'activité vient d'une recommandation
        (impression_id), le signal « save » est aussi journalisé une fois pour
        l'évaluation, comme avant.
        """
        def write(db):
            owner = owner_id(data.get("owner_id"))
            session = self._session(db, data.get("session_id"))
            entries = data.get("items")
            if not isinstance(entries, list) or not 1 <= len(entries) <= 50:
                raise ValueError("items must contain 1 to 50 activities")
            parsed = []
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) - {"item_id", "impression_id"}:
                    raise ValueError("Each item needs item_id and an optional impression_id")
                item = identifier(entry.get("item_id"), "item_id")
                impression = entry.get("impression_id")
                if impression is not None:
                    row = db.execute("SELECT item_id FROM recommendations_shown WHERE impression_id=? AND session_id=?",
                                     (identifier(impression, "impression_id"), session["session_id"])).fetchone()
                    if row is None or row["item_id"] != item:
                        raise ValueError("Impression does not belong to this session and activity")
                parsed.append((item, impression))
            if len({item for item, _ in parsed}) != len(parsed):
                raise ValueError("Duplicate activity in items")
            self.catalog.require_eligible([item for item, _ in parsed], json.loads(session["constraints_json"]))
            at, result = now(), []
            for item, impression in parsed:
                old = db.execute("SELECT removed_at_utc FROM saved_items WHERE owner_id=? AND item_id=?",
                                 (owner, item)).fetchone()
                if old is not None and old["removed_at_utc"] is None:
                    status = "already_saved"
                else:
                    db.execute(
                        "INSERT INTO saved_items(owner_id,item_id,session_id,impression_id,saved_at_utc,removed_at_utc) "
                        "VALUES (?,?,?,?,?,NULL) ON CONFLICT(owner_id,item_id) DO UPDATE SET "
                        "session_id=excluded.session_id,impression_id=excluded.impression_id,"
                        "saved_at_utc=excluded.saved_at_utc,removed_at_utc=NULL",
                        (owner, item, session["session_id"], impression, at))
                    status = "saved" if old is None else "restored"
                if impression is not None and db.execute(
                        "SELECT 1 FROM recommendation_actions WHERE impression_id=? AND action='save'",
                        (impression,)).fetchone() is None:
                    db.execute("INSERT INTO recommendation_actions VALUES (?,?,?,?)",
                               (f"{data['event_id']}:{item}", impression, "save", at))
                result.append(dict(item_id=item, status=status))
            return dict(items=result, count=self._saved_count(db, owner))
        return self._request("save", data, {"owner_id", "session_id", "items"}, write)

    def remove_items(self, data):
        """Retire des activités de la sélection (la ligne reste, datée, pour pouvoir annuler)."""
        def write(db):
            owner = owner_id(data.get("owner_id"))
            items = self._item_list(data.get("item_ids"))
            at, removed = now(), []
            for item in items:
                cursor = db.execute("UPDATE saved_items SET removed_at_utc=? WHERE owner_id=? AND item_id=? "
                                    "AND removed_at_utc IS NULL", (at, owner, item))
                if cursor.rowcount:
                    removed.append(item)
            return dict(removed=removed, count=self._saved_count(db, owner))
        return self._request("unsave", data, {"owner_id", "item_ids"}, write)

    def restore_items(self, data):
        """Annule un retrait : remet les activités dans la sélection, avec leur séance d'origine."""
        def write(db):
            owner = owner_id(data.get("owner_id"))
            items = self._item_list(data.get("item_ids"))
            restored = []
            for item in items:
                cursor = db.execute("UPDATE saved_items SET removed_at_utc=NULL WHERE owner_id=? AND item_id=? "
                                    "AND removed_at_utc IS NOT NULL", (owner, item))
                if cursor.rowcount:
                    restored.append(item)
            return dict(restored=restored, count=self._saved_count(db, owner))
        return self._request("restore", data, {"owner_id", "item_ids"}, write)

    def list_saved(self, owner):
        """Sélection courante d'un carnet, la plus récente d'abord, avec les contraintes de la séance."""
        owner = owner_id(owner)
        with closing(self.connect()) as db:
            rows = db.execute(
                "SELECT s.item_id,s.session_id,s.impression_id,s.saved_at_utc,x.constraints_json "
                "FROM saved_items s JOIN sessions x USING(session_id) "
                "WHERE s.owner_id=? AND s.removed_at_utc IS NULL ORDER BY s.saved_at_utc DESC,s.item_id",
                (owner,)).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _saved_count(db, owner):
        return db.execute("SELECT count(*) FROM saved_items WHERE owner_id=? AND removed_at_utc IS NULL",
                          (owner,)).fetchone()[0]

    @staticmethod
    def _item_list(items):
        if (not isinstance(items, list) or not 1 <= len(items) <= 200
                or any(not isinstance(i, str) or not i.strip() or len(i) > 200 for i in items)
                or len(set(items)) != len(items)):
            raise ValueError("item_ids must contain 1 to 200 distinct activity IDs")
        return items
