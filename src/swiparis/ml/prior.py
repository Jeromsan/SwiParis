"""Prior population w₀ appris hors ligne, dans l'espace φ du profil de session.

Même modèle que la session (Bradley-Terry linéaire, sans biais) :
    P(choisi ≻ rejeté) = σ(w₀ᵀ(φ(choisi) − φ(rejeté)))
w₀ minimise la perte logistique moyenne sur les paires des utilisateurs
d'entraînement, plus (λ/2)‖w₀‖². Le problème est convexe et n'a que
`FeatureSpace.dim` paramètres : il est résolu exactement par la méthode de
Newton. λ est choisi sur la perte des paires de validation ; le test n'est pas lu.

Les paires viennent de `python -m swiparis.pairs ... --splits data/processed/splits.csv`
(le découpage commun).

Blocs appris : catégorie, prix, zone et popularité. Le moment de la journée
n'est pas appris ici. Il est calculé à partir des
visites ; dans une paire d'entraînement, le lieu choisi a donc toujours un
moment connu (la visite qui crée la paire), contrairement aux utilisateurs
nouveaux. Le modèle apprendrait « moment connu = choisi », ce qui s'inverse en
validation (fuite mesurée le 20/11/2026). Les poids du moment restent à 0 dans
w₀ et sont appris pendant la session. Pour la même raison, la popularité du lieu
choisi d'une paire d'entraînement est recalculée sans les visites de
l'utilisateur de la paire.

Les négatifs des paires sont tirés selon la popularité : sur ces paires, être
populaire distingue peu le choisi du rejeté, alors que la popularité compte
beaucoup pour classer tout le catalogue. Le poids de popularité de w₀ est donc
multiplié par un facteur choisi sur le NDCG@10 des utilisateurs de validation
(classement par w₀ seul, protocole de `evaluation.py`). Le test n'est pas lu.

    python -m swiparis.ml.prior --pairs data/processed/pairs.csv --output models/prior_w0_phi.json
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np

from collections import Counter

from swiparis.ml.evaluation import ranking_metrics, temporal_holdout
from swiparis.ml.features import FeatureSpace
from swiparis.ml.items import EncodedItems, encode_common, load_histories, popularity_value
from swiparis.ml.model import sigmoid

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
POP_SCALES = (1, 2, 4, 8, 16)
PROFILE_ITEMS = 5
PRIOR_BLOCKS = ("cat", "price", "zone", "pop")


def block_mask(space: FeatureSpace, blocks=PRIOR_BLOCKS) -> np.ndarray:
    mask = np.zeros(space.dim, dtype=bool)
    for name in blocks:
        if name in space.blocks:
            mask[space.blocks[name]] = True
    return mask


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def deltas(pairs: list[dict], encoded: EncodedItems, split: str, own: Counter | None = None) -> np.ndarray:
    """Δ = φ(choisi) − φ(rejeté) pour les paires d'un split (une ligne par paire).

    `own` : visites par (utilisateur, lieu) déjà comptées dans la popularité. Elles
    sont retirées de la popularité des deux lieux de la paire de cet utilisateur.
    """
    rows = [p for p in pairs if p["split"] == split]
    if not rows:
        return np.zeros((0, encoded.space.dim))
    chosen = encoded.rows([p["chosen_venue_id"] for p in rows])
    rejected = encoded.rows([p["rejected_venue_id"] for p in rows])
    if own and "pop" in encoded.space.blocks:
        column, maximum = encoded.space.blocks["pop"].start, max(encoded.counts.values())
        for k, p in enumerate(rows):
            for matrix, item in ((chosen, p["chosen_venue_id"]), (rejected, p["rejected_venue_id"])):
                if own[(p["user_id"], item)]:
                    matrix[k, column] = popularity_value(encoded.counts[item] - own[(p["user_id"], item)], maximum)
    return chosen - rejected


def logistic_loss(w: np.ndarray, delta: np.ndarray) -> float:
    """Perte logistique moyenne −log σ(wᵀΔ), sans régularisation."""
    return float(np.mean(np.logaddexp(0.0, -(delta @ w)))) if len(delta) else float("nan")


def accuracy(w: np.ndarray, delta: np.ndarray) -> float:
    """Part des paires bien ordonnées ; une égalité de score compte pour 1/2 (comme un tirage)."""
    if not len(delta):
        return float("nan")
    margin = delta @ w
    return float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0))


def fit(delta: np.ndarray, lam: float, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Minimise mean(log(1 + exp(−wᵀΔ))) + (λ/2)‖w‖² par Newton (avec recherche linéaire)."""
    n, d = delta.shape
    w = np.zeros(d)

    def objective(v):
        return logistic_loss(v, delta) + lam / 2 * v @ v

    for _ in range(max_iter):
        p = sigmoid(delta @ w)
        gradient = -(delta.T @ (1 - p)) / n + lam * w
        hessian = (delta.T * (p * (1 - p))) @ delta / n + lam * np.eye(d)
        step = np.linalg.solve(hessian, gradient)
        size, current = 1.0, objective(w)
        while objective(w - size * step) > current and size > 1e-8:
            size /= 2
        w = w - size * step
        if abs(gradient @ step) < tol:
            break
    return w


def validation_ndcg(w: np.ndarray, encoded: EncodedItems, histories) -> float:
    """NDCG@10 moyen du classement par w seul, protocole de `evaluation.py`."""
    values = []
    for visits in histories.values():
        support, relevant = temporal_holdout(visits, PROFILE_ITEMS)
        if support and relevant:
            excluded = set(support)
            keep = [i for i, item in enumerate(encoded.ids) if item not in excluded]
            scores = encoded.phi[keep] @ w
            values.append(ranking_metrics(dict(zip([encoded.ids[i] for i in keep], scores)), relevant)["ndcg"])
    return float(np.mean(values))


def train_prior(db_path, pairs_path, lambdas=LAMBDAS, pop_scales=POP_SCALES) -> dict:
    pairs = read_pairs(pairs_path)
    train_users = {p["user_id"] for p in pairs if p["split"] == "train"}
    val_users = {p["user_id"] for p in pairs if p["split"] == "val"}
    encoded = encode_common(db_path, users=train_users, sources=["foursquare"], popularity_users=train_users)
    own = Counter((user, item) for user, visits in load_histories(db_path, train_users).items() for _, item in visits)
    mask = block_mask(encoded.space)
    train, val = deltas(pairs, encoded, "train", own) * mask, deltas(pairs, encoded, "val") * mask
    search = []
    for lam in lambdas:
        w = fit(train, lam)
        search.append({"lambda": lam, "train_loss": logistic_loss(w, train), "val_loss": logistic_loss(w, val),
                       "train_accuracy": accuracy(w, train), "val_accuracy": accuracy(w, val), "w": w})
    best = min(search, key=lambda r: r["val_loss"])
    histories = load_histories(db_path, val_users)
    column = encoded.space.blocks["pop"].start
    calibration = []
    for scale in pop_scales:
        w = best["w"].copy()
        w[column] *= scale
        calibration.append({"pop_scale": scale, "val_ndcg10": validation_ndcg(w, encoded, histories), "w": w})
    chosen = max(calibration, key=lambda r: r["val_ndcg10"])
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as db:
        manifest = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='manifest'").fetchone()[0])
    return {
        "model": "bradley_terry_linear_phi",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "feature_names": encoded.space.names(),
        "grid": list(encoded.space.grid),
        "trained_blocks": list(PRIOR_BLOCKS),
        "w0": [round(float(x), 8) for x in chosen["w"]],
        "lambda": best["lambda"],
        "pop_scale": chosen["pop_scale"],
        "pop_scale_search": [{k: v for k, v in r.items() if k != "w"} for r in calibration],
        "metrics": {k: best[k] for k in ("train_loss", "val_loss", "train_accuracy", "val_accuracy")},
        "lambda_search": [{k: v for k, v in r.items() if k != "w"} for r in search],
        "inputs": {"data_version": manifest.get("data_version"), "pairs_sha256": sha256_file(pairs_path),
                   "pairs_by_split": {s: sum(p["split"] == s for p in pairs) for s in ("train", "val", "test")},
                   "train_users": len(train_users)},
        "popularity_users": "train",
        "note": "Test users are never read. Time-of-day and popularity use training users' visits only.",
    }


def load_prior(path, space: FeatureSpace) -> np.ndarray:
    """Charge w₀ et vérifie qu'il est exprimé dans le même ordre de features que `space`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["feature_names"] != space.names():
        raise ValueError("Prior w0 was trained with another feature order; retrain it with this FeatureSpace")
    return np.asarray(data["w0"], dtype=float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/prior_w0_phi.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; use --force to replace it")
    result = train_prior(args.db, args.pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("lambda", "metrics", "pop_scale", "pop_scale_search")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
