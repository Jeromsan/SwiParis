"""Sessions simulées hors ligne : scores pour l'évaluation et profils exemples.

Protocole, pour chaque utilisateur du split demandé (aucune visite future n'est
lue pendant la session) :
- profil = ses 5 premiers lieux distincts (`evaluation.temporal_holdout`) ; les
  lieux distincts visités ensuite servent uniquement à mesurer les résultats ;
- candidats = tous les lieux Foursquare sauf ceux du profil ;
- l'utilisateur simulé répond à une paire en prenant l'activité la plus proche de
  son profil (cosinus maximal avec les φ de ses lieux de profil) ; égalité = passage ;
- la session part du prior w₀ et s'arrête après 5 réponses ou 8 paires ;
- `active` choisit les paires avec `next_pair`, `uniform` les tire au hasard :
  même budget, même prior, même mise à jour. Seule la sélection change (ablation).
Le moment de la journée des lieux vient des visites d'entraînement uniquement.

    python -m swiparis.ml.simulation --split val
    python -m swiparis.ml.simulation --split test --scores outputs/scores_test.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

import numpy as np

from swiparis.ml.evaluation import _mean_ci, ranking_metrics, temporal_holdout
from swiparis.ml.items import encode_common, load_histories
from swiparis.ml.model import SessionProfile
from swiparis.ml.prior import load_prior
from swiparis.ml.selection import intra_list_diversity, next_pair, session_finished
from swiparis.ml.splits import load_splits

STRATEGIES = ("active", "uniform")
PROFILE_ITEMS = 5


def _unit(phi: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    return np.divide(phi, norms, out=np.zeros_like(phi), where=norms > 0)


def profile_similarity(phi: np.ndarray, profile_phi: np.ndarray) -> np.ndarray:
    """Pour chaque activité, cosinus maximal avec un lieu du profil."""
    return (_unit(phi) @ _unit(profile_phi).T).max(axis=1)


def uniform_pair(phi, ids, shown, rng, tries: int = 200):
    """Paire tirée uniformément, sans paire déjà montrée ni vecteurs identiques."""
    for _ in range(tries):
        a, b = rng.choice(len(ids), size=2, replace=False)
        if frozenset((ids[a], ids[b])) not in shown and not np.allclose(phi[a], phi[b]):
            return ids[a], ids[b]
    return None


def simulate_session(prior, phi, ids, similarity, strategy, rng, index=None, **session_kw):
    """Une session simulée ; renvoie le profil final et le journal (paire, réponse)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    index = index or {item_id: i for i, item_id in enumerate(ids)}
    profile = SessionProfile(prior, **session_kw)
    offered, log = [], []
    while not session_finished(profile.n_updates, len(offered)):
        shown = {frozenset(p) for p in offered}
        pair = (next_pair(profile, phi, ids, offered, rng=rng) if strategy == "active"
                else uniform_pair(phi, ids, shown, rng))
        if pair is None:
            break
        left, right = index[pair[0]], index[pair[1]]
        gap = similarity[left] - similarity[right]
        choice = "left" if gap > 1e-9 else "right" if gap < -1e-9 else "skip"
        profile.answer(phi[left], phi[right], choice)
        offered.append(pair)
        log.append((pair, choice))
    return profile, log


def run(db_path, prior_path, splits_path, split, seed=42, scores_path=None, session_kw=None):
    splits = load_splits(splits_path)
    train_users = {u for u, s in splits.items() if s == "train"}
    users = sorted(u for u, s in splits.items() if s == split)
    fsq = encode_common(db_path, users=train_users, sources=["foursquare"], popularity_users=train_users)
    prior = load_prior(prior_path, fsq.space)
    session_kw = session_kw or {}
    taste = np.ones(fsq.space.dim, dtype=bool)
    taste[fsq.space.blocks["pop"]] = False  # l'utilisateur simulé juge sur ses goûts, pas la popularité
    popularity = np.array([fsq.counts.get(i, 0) for i in fsq.ids], dtype=float)
    pop_rank = np.empty(len(fsq.ids))
    pop_rank[np.argsort(-popularity, kind="stable")] = np.arange(len(fsq.ids))
    pop_percentile = 1 - pop_rank / (len(fsq.ids) - 1)  # 1 = le plus populaire
    histories = load_histories(db_path, set(users))
    methods = ("active", "uniform", "prior_only", "popularity", "random")
    per_user, sessions, top_stats = {m: {} for m in methods}, {}, defaultdict(list)
    scores_file = scores_path and Path(scores_path).open("w", newline="", encoding="utf-8")
    writer = scores_file and csv.writer(scores_file)
    if writer:
        writer.writerow(["user_id", "item_id", "active_score", "uniform_score"])
    try:
        for n, user in enumerate(users):
            support, relevant = temporal_holdout(histories[user], PROFILE_ITEMS)
            if not support or not relevant:
                continue
            keep = [i for i, item in enumerate(fsq.ids) if item not in set(support)]
            phi, ids = fsq.phi[keep], [fsq.ids[i] for i in keep]
            index = {item: i for i, item in enumerate(ids)}
            similarity = profile_similarity(phi[:, taste], fsq.rows(list(support))[:, taste])
            rng = np.random.default_rng([seed, n])
            scores = {}
            for strategy in STRATEGIES:
                profile, log = simulate_session(prior, phi, ids, similarity, strategy, rng, index, **session_kw)
                scores[strategy] = profile.scores(phi)
                sessions.setdefault(user, {})[strategy] = (profile, log)
            scores["prior_only"] = phi @ prior
            scores["popularity"] = popularity[keep]
            scores["random"] = rng.random(len(ids))
            for method in methods:
                per_user[method][user] = ranking_metrics(dict(zip(ids, scores[method])), relevant)
                top = np.argsort(-scores[method], kind="stable")[:10]
                top_stats[method].append((intra_list_diversity(phi[top]), float(pop_percentile[keep][top].mean()),
                                          [ids[i] for i in top]))
            sessions[user]["support"], sessions[user]["relevant"] = support, relevant
            if writer:
                writer.writerows(zip([user] * len(ids), ids, np.round(scores["active"], 10),
                                     np.round(scores["uniform"], 10)))
    finally:
        if scores_file:
            scores_file.close()
    rng_ci = random.Random(seed)
    summary = {}
    for method in methods:
        values = defaultdict(list)
        for result in per_user[method].values():
            for metric, value in result.items():
                if value is not None:
                    values[metric].append(value)
        stats = top_stats[method]
        summary[method] = {metric: _mean_ci(v, 1000, rng_ci) for metric, v in values.items()}
        summary[method]["top10_diversity"] = float(np.mean([s[0] for s in stats]))
        summary[method]["top10_popularity_percentile"] = float(np.mean([s[1] for s in stats]))
        summary[method]["catalog_coverage_top10"] = len({i for s in stats for i in s[2]}) / len(fsq.ids)
    return {"split": split, "users": len(users), "evaluated_users": len(per_user["active"]),
            "candidates": len(fsq.ids), "session": session_kw, "summary": summary}, per_user, sessions, fsq


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--prior", type=Path, default=Path("models/prior_w0_phi.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--scores", type=Path, help="CSV pour python -m swiparis.ml.evaluation (test)")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, help="pas η de la session (défaut : celui de SessionProfile)")
    args = parser.parse_args(argv)
    session_kw = {"eta": args.eta} if args.eta is not None else {}
    report, *_ = run(args.db, args.prior, args.splits, args.split, args.seed, args.scores, session_kw)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
