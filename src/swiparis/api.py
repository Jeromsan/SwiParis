"""FastAPI adapter for pairwise elicitation, recommendations and pilot logs.

Run after preparing ``data/processed/common.sqlite`` with:
    uvicorn swiparis.api:app --host 127.0.0.1 --port 8000

The service uses the shared SQLite catalog and the separate pilot database.
Until a Paris Data compatible trained prior is available, it uses a transparent
category-frequency prior and the existing Bradley-Terry session updates.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

from .catalog import Catalog
from .logging_store import Conflict, LogStore
from .ml.features import FeatureSpace, Item, source_local_hour, time_profile
from .ml.model import SessionProfile
from .ml.selection import next_pair, rank


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


def _item_vector(row: dict[str, Any]) -> Item:
    location = row["locations"][0] if row.get("locations") else {}
    occurrence = row.get("next_occurrence") or {}
    hours = []
    if occurrence.get("source_start"):
        hours.append(source_local_hour(occurrence["source_start"]))
    return Item(
        item_id=row["item_id"],
        categories=tuple(row.get("categories", [])),
        price_type=row.get("price_type", "unknown"),
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        time_profile=time_profile(hours),
    )


def create_app(catalog_path: str | Path | None = None,
               logs_path: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="SwiParis API", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"error": "Invalid request"})

    @app.on_event("startup")
    def load_services() -> None:
        default_catalog, default_logs = _paths()
        catalog = Catalog(catalog_path or default_catalog)
        logs = LogStore(logs_path or default_logs, catalog)
        feature_space = FeatureSpace(sorted(catalog.categories))
        prior, model_version = _load_prior(catalog, feature_space)
        app.state.catalog = catalog
        app.state.logs = logs
        app.state.feature_space = feature_space
        app.state.prior = prior
        app.state.model_version = model_version

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
        items = [_item_vector(row) for row in rows]
        ids = [item.item_id for item in items]
        phi = feature_space.encode_many(items)
        index = {item_id: pos for pos, item_id in enumerate(ids)}
        profile = SessionProfile(prior)
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
        shown = [(row["left_item_id"], row["right_item_id"]) for row in comparisons]
        for row in comparisons:
            if row["choice"] == "skip":
                continue
            chosen = row["left_item_id"] if row["choice"] == "left" else row["right_item_id"]
            rejected = row["right_item_id"] if row["choice"] == "left" else row["left_item_id"]
            if chosen in index and rejected in index:
                profile.update(phi[index[chosen]], phi[index[rejected]])
        return rows, ids, phi, profile, shown, len(comparisons)

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
            rows, ids, phi, profile, shown, offered = session_state(
                logs, session_id, catalog, feature_space, prior, constraints)
            if profile.n_updates >= 5 or offered >= 8:
                return {"comparison": None, "done": True,
                        "stop_reason": "five_answers" if profile.n_updates >= 5 else "eight_offers",
                        "answers": profile.n_updates, "offers": offered}
            chosen_pair = next_pair(profile, phi, ids, shown_pairs=shown)
            if chosen_pair is None:
                return {"comparison": None, "done": True, "stop_reason": "no_informative_pair",
                        "answers": profile.n_updates, "offers": offered}
            item_a, item_b = sorted(chosen_pair)
            return {"comparison": {"item_a": item_a, "item_b": item_b}, "done": False,
                    "answers": profile.n_updates, "offers": offered}
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
            rows, ids, phi, profile, shown, offered = session_state(
                logs, payload.session_id, catalog, feature_space, prior, constraints)
            seen = {item for pair in shown for item in pair}
            keep = [i for i, item_id in enumerate(ids) if item_id not in seen]
            if not keep:
                return {"items": [], "catalog_version": catalog.version,
                        "model_version": request.app.state.model_version, "impressions": []}
            eligible_ids = [ids[i] for i in keep]
            eligible_phi = phi[keep]
            ranked = rank(profile, eligible_phi, eligible_ids, k=10, diversity=0.3)
            item_map = {row["item_id"]: row for row in rows}
            item_ids = [item_id for item_id, _ in ranked]
            shown_result = logs.show_recommendations({"event_id": payload.event_id,
                "session_id": payload.session_id, "item_ids": item_ids})
            impression_map = {entry["item_id"]: entry for entry in shown_result["impressions"]}
            results = []
            for item_id, score in ranked:
                result = dict(item_map[item_id])
                result["score"] = score
                result["impression_id"] = impression_map[item_id]["impression_id"]
                results.append(result)
            return {"items": results, "catalog_version": catalog.version,
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

