"""FastAPI adapter for pairwise elicitation, recommendations and pilot logs.

Run after preparing ``data/processed/common.sqlite`` with:
    uvicorn swiparis.api:app --host 127.0.0.1 --port 8000

The service uses the shared SQLite catalog and the separate pilot database.
It loads the versioned Bradley-Terry prior from ``models/`` and applies session
updates. If no prior is registered, it falls back to category frequency.
The session policy (pairs, preferences, ranking) lives in ``swiparis.ml.policy`` so
that the offline simulation measures exactly what the API does. When
``models/personas.json`` is present (``SWIPARIS_PERSONAS``), each session starts from
a mixture of personas updated after every choice.
"""
from __future__ import annotations

from contextlib import asynccontextmanager, closing
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import numpy as np

from .catalog import Catalog
from .logging_store import Conflict, LogStore
from .ml import policy
from .ml.features import FeatureSpace
from .ml.items import common_items
from .ml.model import SessionProfile
from .ml.personas import PersonaModel, PersonaProfile
from .serving import load_model


SESSION_POLICY_VERSION = "session-v4"


class SessionPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    consent: bool
    constraints: dict[str, Any] = Field(default_factory=dict)
    catalog_version: str
    model_version: str = "category-prior-v1"
    data_origin: str = "pilot"


class PairPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    session_id: str
    item_a: str
    item_b: str


class InteractionPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    session_id: str
    comparison_id: str
    choice: str


class RecommendationsPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    session_id: str


class SavedItem(BaseModel):
    item_id: str = Field(min_length=1, max_length=200)
    impression_id: str | None = Field(default=None, min_length=1, max_length=200)


class SavePayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    owner_id: str
    session_id: str
    items: list[SavedItem] = Field(min_length=1, max_length=50)


class SavedIdsPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    owner_id: str
    item_ids: list[str] = Field(min_length=1, max_length=200)


class ItemActionPayload(BaseModel):
    event_id: str = Field(min_length=1, max_length=200)
    session_id: str
    impression_id: str
    action: str


def _paths() -> tuple[Path, Path]:
    return (
        Path(os.environ.get("SWIPARIS_CATALOG", "data/processed/common.sqlite")),
        Path(os.environ.get("SWIPARIS_LOGS", "data/pilot/interactions.sqlite")),
    )


def _simple_prior(catalog: Catalog, feature_space: FeatureSpace) -> np.ndarray:
    """Use smoothed Foursquare category frequency as the initial global score."""
    prior = np.zeros(feature_space.dim, dtype=float)
    if not catalog.categories:
        return prior
    with closing(sqlite3.connect(str(catalog.path))) as db:
        counts = dict(db.execute(
            "SELECT category, count(*) FROM training_visits GROUP BY category"
        ).fetchall())
    total = sum(counts.values()) + len(feature_space.categories)
    category_block = feature_space.blocks["cat"]
    for index, category in enumerate(feature_space.categories):
        prior[category_block.start + index] = np.log((counts.get(category, 0) + 1) / total)
    return prior


def _load_prior(catalog: Catalog, feature_space: FeatureSpace) -> tuple[np.ndarray, str]:
    """Load an optional Paris Data-compatible prior, or use category popularity."""
    path = os.environ.get("SWIPARIS_PRIOR_W0")
    if not path:
        return _simple_prior(catalog, feature_space), "category-prior-v1"
    prior = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if prior.shape != (feature_space.dim,) or not np.isfinite(prior).all():
        raise ValueError(f"SWIPARIS_PRIOR_W0 must contain {feature_space.dim} finite values")
    fallback_version = Path(path).parent.name or Path(path).stem
    return prior, os.environ.get("SWIPARIS_MODEL_VERSION", fallback_version)


def _load_scorer(catalog: Catalog) -> tuple[FeatureSpace, np.ndarray, str]:
    """Trained prior from models/ (SWIPARIS_MODELS_DIR), else the category-frequency prior."""
    try:
        model = load_model(os.environ.get("SWIPARIS_MODELS_DIR", "models"))
    except FileNotFoundError:
        feature_space = FeatureSpace(sorted(catalog.categories))
        prior, version = _load_prior(catalog, feature_space)
        return feature_space, prior, version
    unknown = set(catalog.categories) - set(model.space.categories)
    if unknown:
        raise ValueError(f"Catalog categories unknown to the trained prior: {sorted(unknown)}")
    return model.space, model.w0, model.version


def _encode_catalog(catalog: Catalog, feature_space: FeatureSpace) -> dict[str, np.ndarray]:
    """φ de chaque annonce Paris Data, calculé une fois au démarrage.

    Même encodage que l'entraînement et l'évaluation (`swiparis.ml.items`) : le moment
    de la journée vient de toutes les séances de l'annonce, pas seulement de celle qui
    tombe dans le créneau demandé.
    """
    items = common_items(catalog.path, users=None, sources=["paris_data"])
    phi = feature_space.encode_many(items)
    return {item.item_id: phi[row] for row, item in enumerate(items)}


def _load_personas(feature_space: FeatureSpace) -> PersonaModel | None:
    """Personas de ``SWIPARIS_PERSONAS`` (défaut models/personas.json) ; ``off`` les désactive."""
    path = os.environ.get("SWIPARIS_PERSONAS", "models/personas.json")
    if path.lower() == "off" or not Path(path).exists():
        return None
    try:
        return PersonaModel.load(path, feature_space)
    except ValueError:
        return None  # autre espace φ (prior de repli) : séance sans personas


def _persona_summary(profile) -> dict | None:
    if not isinstance(profile, PersonaProfile):
        return None
    weights = profile.weights_pi
    personas = profile.personas
    return {"top": [{"index": int(k), "name": personas.names[k], "description": personas.descriptions[k],
                     "share": round(float(weights[k]), 4)} for k in np.argsort(-weights)[:2]]}


def create_app(catalog_path: str | Path | None = None,
               logs_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        load_services()
        yield

    app = FastAPI(title="SwiParis API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"error": "Invalid request"})

    def load_services() -> None:
        default_catalog, default_logs = _paths()
        catalog = Catalog(catalog_path or default_catalog)
        logs = LogStore(logs_path or default_logs, catalog)
        feature_space, prior, model_version = _load_scorer(catalog)
        personas = _load_personas(feature_space)
        eta = policy.session_eta()
        app.state.catalog = catalog
        app.state.logs = logs
        app.state.feature_space = feature_space
        app.state.prior = prior
        app.state.personas = personas
        app.state.eta = eta
        app.state.vectors = _encode_catalog(catalog, feature_space)
        version = f"{model_version}+{SESSION_POLICY_VERSION}+eta{eta:g}"
        app.state.model_version = version + (f"+personas{personas.k}" if personas else "")

    @app.middleware("http")
    async def localhost_only(request: Request, call_next):
        started = time.perf_counter()
        host = request.headers.get("host", "").split(":", 1)[0].lower()
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return _json_error(403, "Localhost only")
        origin = request.headers.get("origin")
        if origin and not (origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")):
            return _json_error(403, "Cross-origin requests are not allowed")
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    def services(request: Request):
        try:
            return (request.app.state.catalog, request.app.state.logs,
                    request.app.state.feature_space, request.app.state.prior)
        except AttributeError as exc:
            raise HTTPException(503, "Catalog is not ready") from exc

    def session_state(logs: LogStore, session_id: str, catalog: Catalog,
                      feature_space: FeatureSpace, prior: np.ndarray,
                      constraints: dict[str, Any]):
        rows = catalog.list_items(constraints, limit=None)["items"]
        ids = [row["item_id"] for row in rows]
        vectors = app.state.vectors
        phi = np.vstack([vectors[item_id] for item_id in ids]) if ids else np.zeros((0, feature_space.dim))
        index = {item_id: pos for pos, item_id in enumerate(ids)}
        if app.state.personas is not None:
            profile = PersonaProfile(app.state.personas, eta=app.state.eta, lam=policy.LAMBDA)
        else:
            profile = SessionProfile(prior, eta=app.state.eta, lam=policy.LAMBDA)
        with closing(logs.connect()) as db:
            session = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if session is None:
                raise ValueError("Unknown session")
            if session["catalog_version"] != catalog.version:
                raise Conflict("Session belongs to another catalog version; create a new session")
            comparisons = db.execute(
                "SELECT left_item_id,right_item_id,choice FROM comparisons "
                "WHERE session_id=? ORDER BY offered_at_utc,comparison_id", (session_id,)
            ).fetchall()
        shown, selected, rejected, history = policy.replay(
            profile, [(r["left_item_id"], r["right_item_id"], r["choice"]) for r in comparisons], index, phi)
        return rows, ids, phi, profile, shown, len(comparisons), selected, rejected, history

    @app.get("/health")
    def health(request: Request):
        catalog, _, _, _ = services(request)
        return {"status": "ok", "catalog_version": catalog.version,
                "model_version": request.app.state.model_version}

    @app.get("/api/map-config")
    def map_config():
        return {"api_key": os.environ.get("GOOGLE_MAPS_API_KEY", ""),
                "map_id": os.environ.get("GOOGLE_MAPS_MAP_ID", "DEMO_MAP_ID")}

    @app.get("/api/catalog")
    def list_catalog(request: Request, start_utc: str | None = None,
                     end_utc: str | None = None, arrondissement: int | None = None,
                     budget_eur: float | None = None, category: str | None = None,
                     limit: int = 200, offset: int = 0):
        catalog, _, _, _ = services(request)
        try:
            constraints = {k: v for k, v in {
                "start_utc": start_utc, "end_utc": end_utc,
                "arrondissement": arrondissement, "budget_eur": budget_eur,
                "category": category,
            }.items() if v is not None}
            return catalog.list_items(constraints, limit=limit, offset=offset)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/sessions")
    def create_session(payload: SessionPayload, request: Request):
        _, logs, _, _ = services(request)
        try:
            if payload.model_version != request.app.state.model_version:
                raise Conflict("model_version does not match the loaded scorer")
            return logs.create_session(payload.model_dump())
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/next_pair")
    def get_next_pair(session_id: str, request: Request):
        catalog, logs, feature_space, prior = services(request)
        try:
            with closing(logs.connect()) as db:
                session = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if session is None:
                    raise ValueError("Unknown session")
                constraints = json.loads(session["constraints_json"])
            (rows, ids, phi, profile, shown, offered, selected, rejected, history) = session_state(
                logs, session_id, catalog, feature_space, prior, constraints)
            if profile.n_updates >= policy.MAX_ANSWERS or offered >= policy.MAX_OFFERS:
                return {"comparison": None, "done": True,
                        "stop_reason": "max_answers" if profile.n_updates >= policy.MAX_ANSWERS else "max_offers",
                        "answers": profile.n_updates, "offers": offered, "eligible_count": len(ids)}
            chosen_pair, eligible = policy.choose_pair(profile, ids, phi, shown, offered, history, rejected,
                                                       feature_space, policy.seed(f"{session_id}:{offered}"))
            if chosen_pair is None:
                return {"comparison": None, "done": True, "stop_reason": "no_informative_pair",
                        "answers": profile.n_updates, "offers": offered, "eligible_count": eligible}
            item_a, item_b = sorted(chosen_pair)
            return {"comparison": {"item_a": item_a, "item_b": item_b}, "done": False,
                    "answers": profile.n_updates, "offers": offered, "eligible_count": eligible}
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/comparisons")
    def offer_comparison(payload: PairPayload, request: Request):
        catalog, logs, _, _ = services(request)
        try:
            comparison = logs.offer_comparison(payload.model_dump())
            with closing(logs.connect()) as db:
                session = db.execute("SELECT constraints_json FROM sessions WHERE session_id=?",
                                     (payload.session_id,)).fetchone()
            if session is None:
                raise ValueError("Unknown session")
            constraints = json.loads(session["constraints_json"])
            rows = catalog.list_items(constraints, limit=None)["items"]
            by_id = {row["item_id"]: row for row in rows}
            comparison["left"] = by_id[comparison["left_item_id"]]
            comparison["right"] = by_id[comparison["right_item_id"]]
            return comparison
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/item_action")
    def item_action(payload: ItemActionPayload, request: Request):
        _, logs, _, _ = services(request)
        try:
            return logs.item_action(payload.model_dump())
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    def _logged(call, payload):
        try:
            return call(payload)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/go/{impression_id}")
    def open_recommendation(impression_id: str, request: Request):
        """Lien « Voir l'annonce » : journalise l'ouverture puis redirige vers l'annonce source.

        Un seul clic pour l'utilisateur, au lieu de « journaliser » puis « ouvrir ».
        La redirection ne va que vers l'URL source enregistrée de l'activité recommandée.
        """
        catalog, logs, _, _ = services(request)
        with closing(logs.connect()) as db:
            row = db.execute("SELECT session_id,item_id FROM recommendations_shown WHERE impression_id=?",
                             (impression_id,)).fetchone()
        found = catalog.list_items({}, limit=None, item_ids=[row["item_id"]])["items"] if row else []
        url = found[0].get("source_url") if found else None
        if not url or not url.startswith(("https://", "http://")):
            return HTMLResponse("<p>Ce lien n'est plus disponible. Retourne sur SwiParis.</p>", status_code=404)
        try:
            logs.item_action({"event_id": str(uuid.uuid4()), "session_id": row["session_id"],
                              "impression_id": impression_id, "action": "open"})
        except (ValueError, sqlite3.Error):
            pass  # l'annonce doit s'ouvrir même si le journal refuse (catalogue changé, etc.)
        return RedirectResponse(url, status_code=303)

    @app.post("/api/saved")
    def save_items(payload: SavePayload, request: Request):
        _, logs, _, _ = services(request)
        return _logged(logs.save_items, payload.model_dump(exclude_none=False) | {
            "items": [item.model_dump(exclude_none=True) for item in payload.items]})

    @app.post("/api/saved/remove")
    def remove_saved(payload: SavedIdsPayload, request: Request):
        _, logs, _, _ = services(request)
        return _logged(logs.remove_items, payload.model_dump())

    @app.post("/api/saved/restore")
    def restore_saved(payload: SavedIdsPayload, request: Request):
        _, logs, _, _ = services(request)
        return _logged(logs.restore_items, payload.model_dump())

    @app.get("/api/saved")
    def list_saved(owner_id: str, request: Request):
        """Sélection d'un carnet, avec la fiche complète de chaque activité.

        La séance affichée est celle du créneau où l'activité a été gardée. Une
        activité sortie du catalogue reste listée avec ``available: false``.
        """
        catalog, logs, _, _ = services(request)
        try:
            saved = logs.list_saved(owner_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        by_constraints: dict[str, list[str]] = {}
        for row in saved:
            by_constraints.setdefault(row["constraints_json"], []).append(row["item_id"])
        details: dict[str, dict] = {}
        for constraints_json, item_ids in by_constraints.items():
            try:
                rows = catalog.list_items(json.loads(constraints_json), limit=None, item_ids=item_ids)["items"]
            except ValueError:
                rows = []
            details.update({row["item_id"]: row for row in rows})
        missing = [row["item_id"] for row in saved if row["item_id"] not in details]
        if missing:
            details.update({row["item_id"]: row for row in
                            catalog.list_items({}, limit=None, item_ids=missing)["items"]})
        items = []
        for row in saved:
            constraints = json.loads(row["constraints_json"])
            item = dict(details.get(row["item_id"], {"item_id": row["item_id"]}))
            item.update(available=row["item_id"] in details, saved_at_utc=row["saved_at_utc"],
                        session_id=row["session_id"], impression_id=row["impression_id"],
                        slot={k: constraints[k] for k in ("start_utc", "end_utc") if k in constraints})
            items.append(item)
        return {"items": items, "count": len(items), "categories": catalog.categories}

    @app.post("/log_interaction")
    def log_interaction(payload: InteractionPayload, request: Request):
        _, logs, _, _ = services(request)
        try:
            return logs.answer(payload.model_dump())
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/recommendations")
    def get_recommendations(payload: RecommendationsPayload, request: Request):
        catalog, logs, feature_space, prior = services(request)
        try:
            with closing(logs.connect()) as db:
                session = db.execute("SELECT * FROM sessions WHERE session_id=?", (payload.session_id,)).fetchone()
                if session is None:
                    raise ValueError("Unknown session")
                constraints = json.loads(session["constraints_json"])
            (rows, ids, phi, profile, shown, offered, selected, rejected, history) = session_state(
                logs, payload.session_id, catalog, feature_space, prior, constraints)
            ranked = policy.recommend(profile, ids, phi, history, rejected, feature_space,
                                      policy.seed(payload.session_id))
            persona = _persona_summary(profile)
            if not ranked:
                return {"items": [], "catalog_version": catalog.version, "persona": persona,
                        "model_version": request.app.state.model_version, "impressions": []}
            item_map = {row["item_id"]: row for row in rows}
            item_ids = [item_id for item_id, _ in ranked]
            shown_result = logs.show_recommendations({"event_id": payload.event_id,
                "session_id": payload.session_id, "item_ids": item_ids})
            impression_map = {entry["item_id"]: entry for entry in shown_result["impressions"]}
            results = []
            for item_id, score in ranked:
                result = dict(item_map[item_id])
                result["score"] = score
                result["selected_by_user"] = item_id in selected and item_id not in rejected
                result["impression_id"] = impression_map[item_id]["impression_id"]
                results.append(result)
            return {"items": results, "catalog_version": catalog.version, "persona": persona,
                    "model_version": request.app.state.model_version, "impressions": shown_result["impressions"]}
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, sqlite3.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


def _json_error(status: int, message: str):
    return JSONResponse(status_code=status, content={"error": message})


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run("swiparis.api:app", host="127.0.0.1", port=int(os.environ.get("SWIPARIS_PORT", "8000")), reload=False)

