"""Personas apprises sur les historiques Foursquare, et attribution souple pendant la séance.

Une persona est un groupe d'utilisateurs aux habitudes proches, observées dans les
données (pas un stéréotype inventé) :

1. Habitudes d'un utilisateur : répartition de ses visites par catégorie, moment de
   la journée (heure locale de la visite) et zone de Paris. On garde la racine carrée
   de chaque répartition (distance de Hellinger entre répartitions).
2. Personas : k-moyennes sur les habitudes des utilisateurs d'entraînement.
3. Profil w_k de chaque persona : même modèle et mêmes blocs que le prior global w₀
   (catégorie, prix, zone, popularité), appris sur les paires des utilisateurs de la
   persona et rappelé vers le w global : (λ_p/2)‖w_k − w‖². k et λ_p sont choisis sur
   la perte des paires de validation ; le test n'est pas lu.
4. Pendant une séance, π_k = probabilité de ressembler à la persona k :
   - au départ, la part de chaque persona (personne inconnue), ou la proximité de
     son historique connu avec chaque persona : π_k ∝ part_k · exp(−β d²(h, c_k)) ;
   - après chaque choix (Δ = φ(choisi) − φ(rejeté)) : π_k ∝ π_k · σ(w_kᵀΔ)^τ ;
   - profil de départ = Σ π_k w_k, que les choix de la séance affinent avec la même
     mise à jour que `SessionProfile`.
   τ et β sont choisis sur la validation (sessions simulées).

    python -m swiparis.ml.personas train
    python -m swiparis.ml.personas evaluate --split val
    python -m swiparis.ml.personas evaluate --split test --final
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sqlite3

import numpy as np

from swiparis.ml.evaluation import _mean_ci, ranking_metrics, temporal_holdout
from swiparis.ml.features import TIME_BUCKETS, foursquare_local_hour, time_bucket
from swiparis.ml.items import encode_common, load_histories
from swiparis.ml.model import SessionProfile, sigmoid
from swiparis.ml.prior import (POP_SCALES, PRIOR_BLOCKS, accuracy, block_mask, deltas, load_prior, logistic_loss,
                               read_pairs, sha256_file)
from swiparis.ml.selection import intra_list_diversity
from swiparis.ml.simulation import PROFILE_ITEMS, profile_similarity, simulate_session
from swiparis.ml.splits import load_splits

KS = (2, 3, 4, 6, 8)
PERSONA_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)
TEMPERATURES = (0.5, 1.0, 2.0)
BETAS = (1.0, 3.0, 10.0)
MIN_USERS = 20
ROWS, COLS = ("nord", "centre", "sud"), ("ouest", "centre", "est")
TIME_LABELS = {"night": "la nuit", "morning": "le matin", "afternoon": "l'après-midi", "evening": "le soir"}


# --- Habitudes ------------------------------------------------------------------------

def visits_with_buckets(db_path, users) -> dict[str, list[tuple[str, str, int]]]:
    """{user: [(utc_time, item_id, moment), ...]} dans l'ordre du temps (moment = indice de TIME_BUCKETS)."""
    allowed, result = set(users), defaultdict(list)
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        for user, item, utc_time, offset in db.execute(
                "SELECT user_id, item_id, utc_time, timezone_offset FROM visits ORDER BY user_id, utc_time, item_id"):
            if user in allowed:
                result[user].append((utc_time, item, time_bucket(foursquare_local_hour(utc_time, offset))))
    return result


def known_history(visits, profile_items: int = PROFILE_ITEMS):
    """Visites connues avant la mesure : jusqu'à la première visite du 5e lieu distinct."""
    seen, kept = set(), []
    for visit in visits:
        if visit[1] not in seen and len(seen) == profile_items:
            break
        seen.add(visit[1])
        kept.append(visit)
    return kept


def habit_vector(visits, encoded) -> np.ndarray:
    """[√parts catégories, √parts moments, √parts zones] des visites (chaque bloc a une norme de 1)."""
    space = encoded.space
    rows = encoded.rows([item for _, item, _ in visits])
    blocks = []
    for values in (rows[:, space.blocks["cat"]].sum(axis=0),
                   np.bincount([b for _, _, b in visits], minlength=len(TIME_BUCKETS)).astype(float),
                   rows[:, space.blocks["zone"]].sum(axis=0)):
        total = values.sum()
        blocks.append(np.sqrt(values / total) if total > 0 else np.zeros_like(values))
    return np.concatenate(blocks)


def habit_from_phi(rows: np.ndarray, space) -> np.ndarray:
    """Habitudes (même format que `habit_vector`) à partir des φ d'activités, par exemple la
    sélection gardée dans l'application : catégories, moments (profil horaire des séances)
    et zones des activités."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    blocks = []
    for name in ("cat", "time", "zone"):
        values = rows[:, space.blocks[name]].sum(axis=0)
        total = values.sum()
        blocks.append(np.sqrt(values / total) if total > 0 else np.zeros_like(values))
    return np.concatenate(blocks)


def kmeans(x: np.ndarray, k: int, seed: int = 42, n_init: int = 10, iterations: int = 100):
    """k-moyennes (initialisation k-means++), meilleure de `n_init` exécutions."""
    best = None
    rng = np.random.default_rng(seed)
    for _ in range(n_init):
        centers = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        centers = np.array(centers)
        for _ in range(iterations):
            labels = ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)
            moved = np.array([x[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)])
            if np.allclose(moved, centers):
                break
            centers = moved
        inertia = float(((x - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels)
    return best[1], best[2]


def nearest(habits: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((habits[:, None, :] - centers[None]) ** 2).sum(-1).argmin(axis=1)


def fit_centered(delta: np.ndarray, lam: float, center: np.ndarray, max_iter: int = 50, tol: float = 1e-10):
    """Minimise mean(log(1 + exp(−wᵀΔ))) + (λ/2)‖w − c‖² par Newton (comme `prior.fit`)."""
    n, d = delta.shape
    w = center.copy()
    if n == 0:
        return w

    def objective(v):
        return logistic_loss(v, delta) + lam / 2 * (v - center) @ (v - center)

    for _ in range(max_iter):
        p = sigmoid(delta @ w)
        gradient = -(delta.T @ (1 - p)) / n + lam * (w - center)
        hessian = (delta.T * (p * (1 - p))) @ delta / n + lam * np.eye(d)
        step = np.linalg.solve(hessian, gradient)
        size, current = 1.0, objective(w)
        while objective(w - size * step) > current and size > 1e-8:
            size /= 2
        w = w - size * step
        if abs(gradient @ step) < tol:
            break
    return w


# --- Modèle de personas et profil de séance --------------------------------------------

class PersonaModel:
    """Personas apprises : centres d'habitudes, profils w_k, parts, réglages τ et β."""

    def __init__(self, centers, weights, shares, names, descriptions, feature_names,
                 temperature=1.0, beta=3.0, metadata=None):
        self.centers = np.asarray(centers, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.shares = np.asarray(shares, dtype=float)
        self.names, self.descriptions = list(names), list(descriptions)
        self.feature_names = list(feature_names)
        self.temperature, self.beta = float(temperature), float(beta)
        self.metadata = metadata or {}

    @property
    def k(self) -> int:
        return len(self.shares)

    def start(self, habit: np.ndarray | None = None, beta: float | None = None) -> np.ndarray:
        """π de départ : parts des personas, ou proximité de l'historique connu."""
        if habit is None:
            return self.shares.copy()
        beta = self.beta if beta is None else beta
        log_pi = np.log(self.shares) - beta * ((self.centers - habit) ** 2).sum(axis=1)
        log_pi -= log_pi.max()
        pi = np.exp(log_pi)
        return pi / pi.sum()

    def mixture(self, pi: np.ndarray) -> np.ndarray:
        return np.asarray(pi) @ self.weights

    def to_dict(self) -> dict:
        return {"model": "personas_bradley_terry_phi", "feature_names": self.feature_names,
                "centers": self.centers.round(8).tolist(), "weights": self.weights.round(8).tolist(),
                "shares": self.shares.round(8).tolist(), "names": self.names, "descriptions": self.descriptions,
                "temperature": self.temperature, "beta": self.beta, **self.metadata}

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path, space=None) -> "PersonaModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if space is not None and data["feature_names"] != space.names():
            raise ValueError("Personas were trained with another feature order; retrain them")
        keys = ("centers", "weights", "shares", "names", "descriptions", "feature_names", "temperature", "beta")
        return cls(*(data[k] for k in keys), metadata={k: v for k, v in data.items() if k not in keys})


class PersonaProfile(SessionProfile):
    """Profil de séance dont le point de départ est un mélange de personas, revu après chaque choix."""

    def __init__(self, personas: PersonaModel, start=None, temperature: float | None = None, **session_kw):
        if session_kw.get("mode", "sgd") != "sgd":
            raise ValueError("PersonaProfile only supports the sgd session update")
        self.personas = personas
        self.temperature = personas.temperature if temperature is None else temperature
        pi = personas.shares if start is None else np.asarray(start, dtype=float)
        self.log_pi = np.log(np.clip(pi, 1e-300, None))
        self.log_pi -= self.log_pi.max()
        super().__init__(personas.mixture(self.weights_pi), **session_kw)
        self._deltas: list[np.ndarray] = []

    @property
    def weights_pi(self) -> np.ndarray:
        pi = np.exp(self.log_pi - self.log_pi.max())
        return pi / pi.sum()

    def update(self, phi_chosen, phi_rejected) -> float:
        delta = np.asarray(phi_chosen, dtype=float) - np.asarray(phi_rejected, dtype=float)
        if delta.shape != self.w.shape:
            raise ValueError(f"Expected feature vectors of size {self.dim}")
        p = float(sigmoid(self.w @ delta))
        self.log_pi = self.log_pi + self.temperature * np.log(np.clip(sigmoid(self.personas.weights @ delta), 1e-12, 1))
        self.log_pi -= self.log_pi.max()
        self._deltas.append(delta)
        self.prior = self.personas.mixture(self.weights_pi)
        w = self.prior.copy()
        for past in self._deltas:  # mêmes pas que SessionProfile, depuis le nouveau départ
            q = float(sigmoid(w @ past))
            w = w + self.eta * (1 - q) * past - self.lam * (w - self.prior)
        self.w = w
        self.n_updates += 1
        return p


# --- Entraînement ---------------------------------------------------------------------

def _shares(center: np.ndarray, n_cat: int, n_time: int):
    parts = [center[:n_cat] ** 2, center[n_cat:n_cat + n_time] ** 2, center[n_cat + n_time:] ** 2]
    return [p / p.sum() if p.sum() > 0 else p for p in parts]


def describe(center: np.ndarray, space, labels: dict[str, str], population: np.ndarray) -> tuple[str, str]:
    """Nom court et description d'une persona, lus dans son centre d'habitudes.

    Le nom retient les catégories plus fréquentes que dans l'ensemble des utilisateurs
    (`population` : répartition moyenne des catégories), sinon la plus fréquente.
    """
    n_cat, n_time = len(space.categories), len(TIME_BUCKETS)
    cat, times, zones = _shares(center, n_cat, n_time)
    order = np.argsort(-cat)
    lift = np.where(cat >= 0.08, cat / np.maximum(population, 1e-9), 0)
    top = [space.categories[j] for j in np.argsort(-lift)[:2] if lift[j] > 1.15] or [space.categories[order[0]]]
    when = TIME_BUCKETS[int(np.argmax(times))]
    row, col = divmod(int(np.argmax(zones)), space.grid[1])
    where = "centre" if (row, col) == (1, 1) else f"{ROWS[row]}-{COLS[col]}".replace("-centre", "").replace("centre-", "")
    short = lambda code: labels.get(code, code).split(",")[0].split(" et ")[0]
    name = f"{' & '.join(short(c) for c in top)} · {TIME_LABELS[when]} · {where}"
    parts = ", ".join(f"{labels.get(space.categories[j], space.categories[j])} {cat[j]:.0%}" for j in order[:3])
    distinct = ", ".join(labels.get(c, c) for c in top)
    description = (f"Se distingue par : {distinct} (plus fréquent que chez l'ensemble des utilisateurs). "
                   f"Visites : {parts}. Surtout {TIME_LABELS[when]} ({times.max():.0%} des visites), "
                   f"zone la plus fréquentée : {where} ({zones.max():.0%}).")
    return name, description


def history_ndcg(weights: np.ndarray, persona_of: dict, encoded, histories) -> float:
    """NDCG@10 moyen quand chaque utilisateur est classé par le w de sa persona (protocole de prior.py)."""
    values = []
    for user, visits in histories.items():
        support, relevant = temporal_holdout(visits, PROFILE_ITEMS)
        if support and relevant and user in persona_of:
            excluded = set(support)
            keep = [i for i, item in enumerate(encoded.ids) if item not in excluded]
            scores = encoded.phi[keep] @ weights[persona_of[user]]
            values.append(ranking_metrics(dict(zip([encoded.ids[i] for i in keep], scores)), relevant)["ndcg"])
    return float(np.mean(values))


def pair_metrics(model: PersonaModel, db_path, pairs_path, prior_path, split: str) -> dict:
    """Paires d'un split après l'historique connu : persona la plus proche contre w global."""
    pairs = read_pairs(pairs_path)
    train_users = sorted({p["user_id"] for p in pairs if p["split"] == "train"})
    users = sorted({p["user_id"] for p in pairs if p["split"] == split})
    encoded = encode_common(db_path, users=train_users, sources=["foursquare"], popularity_users=train_users)
    prior_data = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    pop = encoded.space.blocks["pop"].start
    center = load_prior(prior_path, encoded.space)
    center[pop] /= prior_data["pop_scale"]
    weights = model.weights.copy()
    weights[:, pop] /= model.metadata.get("pop_scale", prior_data["pop_scale"])
    visits = visits_with_buckets(db_path, users)
    known = {u: known_history(visits[u]) for u in users}
    persona_of = dict(zip(users, nearest(np.array([habit_vector(known[u], encoded) for u in users]), model.centers)))
    later = [dict(p, split="later") for p in pairs
             if p["split"] == split and int(p["visit_rank"]) >= len(known[p["user_id"]])]
    delta = deltas(later, encoded, "later") * block_mask(encoded.space)
    margins = np.einsum("ij,ij->i", delta, weights[[persona_of[p["user_id"]] for p in later]])
    persona = {"loss": float(np.mean(np.logaddexp(0.0, -margins))),
               "accuracy": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0))}
    return {"split": split, "pairs": len(later), "persona": persona,
            "global": {"loss": logistic_loss(center, delta), "accuracy": accuracy(center, delta)}}


def train_personas(db_path, pairs_path, prior_path, ks=KS, lambdas=PERSONA_LAMBDAS, seed=42) -> tuple[PersonaModel, dict]:
    pairs = read_pairs(pairs_path)
    train_users = sorted({p["user_id"] for p in pairs if p["split"] == "train"})
    val_users = sorted({p["user_id"] for p in pairs if p["split"] == "val"})
    encoded = encode_common(db_path, users=train_users, sources=["foursquare"], popularity_users=train_users)
    prior_data = json.loads(Path(prior_path).read_text(encoding="utf-8"))
    global_w = load_prior(prior_path, encoded.space)
    pop = encoded.space.blocks["pop"].start
    center = global_w.copy()
    center[pop] /= prior_data["pop_scale"]  # w global avant calibration de la popularité
    own = Counter((user, item) for user, visits in load_histories(db_path, train_users).items() for _, item in visits)
    mask = block_mask(encoded.space)
    visits = visits_with_buckets(db_path, train_users + val_users)
    habits_train = np.array([habit_vector(visits[u], encoded) for u in train_users])
    # Validation sans fuite : une personne de validation est rattachée à une persona avec
    # son historique connu (jusqu'au 5e lieu distinct), et seules ses paires suivantes
    # servent à mesurer. Sinon, les visites à prédire serviraient à choisir la persona.
    known = {u: known_history(visits[u]) for u in val_users}
    habits_val = np.array([habit_vector(known[u], encoded) for u in val_users])
    train_rows = [p for p in pairs if p["split"] == "train"]
    later = [p for p in pairs if p["split"] == "val" and int(p["visit_rank"]) >= len(known[p["user_id"]])]
    for p in later:
        p["split"] = "val_later"
    val_rows = later
    delta_train = deltas(pairs, encoded, "train", own) * mask
    delta_val = deltas(pairs, encoded, "val_later") * mask
    user_train = np.array([p["user_id"] for p in train_rows])
    user_val = np.array([p["user_id"] for p in val_rows])
    baseline = {"val_loss": logistic_loss(center, delta_val), "val_accuracy": accuracy(center, delta_val),
                "val_pairs": len(val_rows)}
    search, candidates = [], []
    for k in ks:
        centers, labels = kmeans(habits_train, k, seed)
        sizes = np.bincount(labels, minlength=k)
        if sizes.min() < MIN_USERS:
            search.append({"k": k, "skipped": f"persona with {int(sizes.min())} users (< {MIN_USERS})"})
            continue
        of_train = dict(zip(train_users, labels))
        of_val = dict(zip(val_users, nearest(habits_val, centers)))
        pair_train = np.array([of_train[u] for u in user_train])
        pair_val = np.array([of_val[u] for u in user_val])
        for lam in lambdas:
            weights = np.array([fit_centered(delta_train[pair_train == j], lam, center) for j in range(k)])
            margins = np.einsum("ij,ij->i", delta_val, weights[pair_val])
            val_loss = float(np.mean(np.logaddexp(0.0, -margins)))
            val_accuracy = float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0))
            search.append({"k": k, "lambda": lam, "val_loss": val_loss, "val_accuracy": val_accuracy,
                           "sizes": sizes.tolist()})
            candidates.append((val_loss, k, lam, centers, weights, sizes))
    val_loss, k, lam, centers, weights, sizes = min(candidates, key=lambda c: c[0])
    # Calibration du poids de popularité sur le NDCG@10 de validation, comme pour w₀
    # (classement par la persona la plus proche de l'historique connu, sans réponses).
    histories = load_histories(db_path, val_users)
    of_val = dict(zip(val_users, nearest(habits_val, centers)))
    calibration = []
    for scale in POP_SCALES:
        scaled = weights.copy()
        scaled[:, pop] *= scale
        calibration.append({"pop_scale": scale, "val_ndcg10": history_ndcg(scaled, of_val, encoded, histories)})
    global_ndcg = history_ndcg(global_w[None, :], {u: 0 for u in val_users}, encoded, histories)
    pop_scale = max(calibration, key=lambda r: r["val_ndcg10"])["pop_scale"]
    weights = weights.copy()
    weights[:, pop] *= pop_scale
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as db:
        labels = dict(db.execute("SELECT category, label FROM categories"))
        manifest = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='manifest'").fetchone()[0])
    n_cat, n_time = len(encoded.space.categories), len(TIME_BUCKETS)
    population = np.average([_shares(c, n_cat, n_time)[0] for c in centers], axis=0, weights=sizes)
    named = [describe(c, encoded.space, labels, population) for c in centers]
    metadata = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "k": k, "lambda": lam, "pop_scale": pop_scale, "trained_blocks": list(PRIOR_BLOCKS),
        "users_per_persona": sizes.tolist(),
        "selection": {"criterion": "loss of val pairs after the known history (users assigned by that history)",
                      "global_prior": {**baseline, "val_ndcg10": global_ndcg}, "chosen": {"val_loss": val_loss},
                      "search": search, "pop_scale_search": calibration},
        "inputs": {"data_version": manifest.get("data_version"), "pairs_sha256": sha256_file(pairs_path),
                   "prior_sha256": sha256_file(prior_path), "train_users": len(train_users)},
        "note": "Test users are never read. Habits and popularity use training users' visits only.",
    }
    model = PersonaModel(centers, weights, sizes / sizes.sum(), [n for n, _ in named], [d for _, d in named],
                         encoded.space.names(), metadata=metadata)
    return model, metadata


# --- Évaluation par sessions simulées -------------------------------------------------

def _paired(per_user, a, b, metric, seed=42):
    diffs = [per_user[a][u][metric] - per_user[b][u][metric] for u in per_user[a]
             if u in per_user[b] and per_user[a][u][metric] is not None and per_user[b][u][metric] is not None]
    return _mean_ci(diffs, 2000, random.Random(seed))


def evaluate(db_path, prior_path, personas_path, splits_path, split, temperatures, betas,
             seed=42, max_answers=5, max_offers=8, users_limit=None, eta=None):
    """Sessions simulées (protocole de `simulation.py`) : prior global contre personas."""
    splits = load_splits(splits_path)
    train_users = {u for u, s in splits.items() if s == "train"}
    users = sorted(u for u, s in splits.items() if s == split)[:users_limit]
    fsq = encode_common(db_path, users=train_users, sources=["foursquare"], popularity_users=train_users)
    prior = load_prior(prior_path, fsq.space)
    personas = PersonaModel.load(personas_path, fsq.space)
    taste = np.ones(fsq.space.dim, dtype=bool)
    taste[fsq.space.blocks["pop"]] = False
    popularity = np.array([fsq.counts.get(i, 0) for i in fsq.ids], dtype=float)
    histories = load_histories(db_path, set(users))
    visits = visits_with_buckets(db_path, users)
    limits = {"max_answers": max_answers, "max_offers": max_offers}
    session_kw = {} if eta is None else {"eta": eta}
    methods = ["global_active", "global_prior_only", "popularity"]
    methods += [f"persona_new_t{t:g}" for t in temperatures]
    methods += [f"persona_history_only_b{b:g}" for b in betas]
    methods += [f"persona_history_t{t:g}_b{b:g}" for t in temperatures for b in betas]
    per_user = {m: {} for m in methods}
    tops, dominant = defaultdict(dict), {}
    for n, user in enumerate(users):
        support, relevant = temporal_holdout(histories[user], PROFILE_ITEMS)
        if not support or not relevant:
            continue
        keep = [i for i, item in enumerate(fsq.ids) if item not in set(support)]
        phi, ids = fsq.phi[keep], [fsq.ids[i] for i in keep]
        index = {item: i for i, item in enumerate(ids)}
        similarity = profile_similarity(phi[:, taste], fsq.rows(list(support))[:, taste])
        habit = habit_vector(known_history(visits[user]), fsq)
        scores = {"global_prior_only": phi @ prior, "popularity": popularity[keep]}

        def session(profile):
            rng = np.random.default_rng([seed, n])  # même aléa pour toutes les méthodes d'un utilisateur
            final, _ = simulate_session(prior, phi, ids, similarity, "active", rng, index, profile=profile, **limits)
            return final

        scores["global_active"] = session(SessionProfile(prior, **session_kw)).scores(phi)
        for t in temperatures:
            scores[f"persona_new_t{t:g}"] = session(PersonaProfile(personas, temperature=t, **session_kw)).scores(phi)
        for b in betas:
            start = personas.start(habit, beta=b)
            scores[f"persona_history_only_b{b:g}"] = phi @ personas.mixture(start)
            for t in temperatures:
                scores[f"persona_history_t{t:g}_b{b:g}"] = session(
                    PersonaProfile(personas, start=start, temperature=t, **session_kw)).scores(phi)
        dominant[user] = int(np.argmax(personas.start(habit, beta=betas[-1])))
        for method in methods:
            per_user[method][user] = ranking_metrics(dict(zip(ids, scores[method])), relevant)
            top = np.argsort(-scores[method], kind="stable")[:10]
            tops[method][user] = ([ids[i] for i in top], intra_list_diversity(phi[top]))
    rng_ci = random.Random(seed)
    summary = {}
    for method in methods:
        values = defaultdict(list)
        for result in per_user[method].values():
            for metric, value in result.items():
                if value is not None:
                    values[metric].append(value)
        summary[method] = {metric: _mean_ci(v, 1000, rng_ci) for metric, v in values.items()}
        summary[method]["top10_diversity"] = float(np.mean([d for _, d in tops[method].values()]))
        summary[method]["distinct_top10_lists"] = len({tuple(t) for t, _ in tops[method].values()})
    paired = {m: {metric: _paired(per_user, m, "global_active", metric) for metric in ("ndcg", "recall")}
              for m in methods if m != "global_active"}
    paired_prior = {m: {metric: _paired(per_user, m, "global_prior_only", metric) for metric in ("ndcg", "recall")}
                    for m in methods if m.startswith("persona_history_only")}
    return {"split": split, "users": len(users), "evaluated_users": len(per_user["global_active"]),
            "session": {**limits, **session_kw}, "summary": summary, "paired_vs_global_active": paired,
            "paired_history_only_vs_global_prior_only": paired_prior,
            "dominant_persona_counts": Counter(dominant.values())}, per_user, tops, dominant


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="apprendre les personas (k et λ_p choisis sur la validation)")
    train.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    train.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.csv"))
    train.add_argument("--prior", type=Path, default=Path("models/prior_w0_phi.json"))
    train.add_argument("--output", type=Path, default=Path("models/personas.json"))
    ev = sub.add_parser("evaluate", help="sessions simulées : prior global contre personas")
    ev.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    ev.add_argument("--prior", type=Path, default=Path("models/prior_w0_phi.json"))
    ev.add_argument("--personas", type=Path, default=Path("models/personas.json"))
    ev.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    ev.add_argument("--split", choices=("val", "test"), default="val")
    ev.add_argument("--final", action="store_true", help="test : seulement τ et β enregistrés dans le modèle")
    ev.add_argument("--max-answers", type=int, default=5)
    ev.add_argument("--max-offers", type=int, default=8)
    ev.add_argument("--users", type=int, help="limiter le nombre d'utilisateurs (essais rapides)")
    ev.add_argument("--eta", type=float, help="pas η de la séance (défaut : celui de SessionProfile)")
    ev.add_argument("--report", type=Path)
    tune = sub.add_parser("set", help="enregistrer τ et β choisis sur la validation")
    tune.add_argument("--personas", type=Path, default=Path("models/personas.json"))
    tune.add_argument("--temperature", type=float, required=True)
    tune.add_argument("--beta", type=float, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        model, metadata = train_personas(args.db, args.pairs, args.prior)
        model.save(args.output)
        print(json.dumps({"k": metadata["k"], "lambda": metadata["lambda"], "names": model.names,
                          "users_per_persona": metadata["users_per_persona"],
                          "selection": metadata["selection"]}, ensure_ascii=False, indent=1))
    elif args.command == "set":
        data = json.loads(args.personas.read_text(encoding="utf-8"))
        data.update(temperature=args.temperature, beta=args.beta)
        args.personas.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        if args.split == "test" and not args.final:
            parser.error("le test se lance une seule fois, avec --final (τ et β déjà choisis sur la validation)")
        model = PersonaModel.load(args.personas)
        temperatures, betas = ((model.temperature,), (model.beta,)) if args.final else (TEMPERATURES, BETAS)
        if args.final:
            pair_report = pair_metrics(model, args.db, Path("data/processed/pairs.csv"), args.prior, args.split)
        report, *_ = evaluate(args.db, args.prior, args.personas, args.splits, args.split, temperatures, betas,
                              max_answers=args.max_answers, max_offers=args.max_offers, users_limit=args.users,
                              eta=args.eta)
        if args.final:
            report["pairs_after_known_history"] = pair_report
        text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
