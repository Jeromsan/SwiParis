"""Mesure hors ligne de la configuration réelle de l'application (`swiparis.ml.policy`).

Même protocole que `simulation.py` (profil = 5 premiers lieux, réponses simulées par
ressemblance au profil, lieux visités ensuite pour mesurer), mais avec la politique de
l'API : jusqu'à 10 réponses ou 15 paires, préférences des choix récents, sorties
rejetées écartées, classement score + préférence + MMR.

Mesures par configuration (η, profil global ou personas) :
- NDCG@10 et Recall@20 sur le catalogue Foursquare (classement MMR des 300 meilleurs) ;
- Hit@3 : au moins un lieu visité ensuite dans les 3 recommandations affichées ;
- personnalisation : nombre moyen d'activités communes entre les top 3 de deux
  utilisateurs (0 = listes toutes différentes, 3 = identiques).

Règle de choix, fixée avant de lire les résultats de validation :
1. parmi les η dont le NDCG@10 de validation est à moins d'une erreur-type du meilleur,
   garder celui qui personnalise le plus (le moins d'activités communes) ;
2. garder les personas si, à ce η, elles ne baissent pas le NDCG@10 de validation de
   plus d'une erreur-type par rapport au profil global ;
3. garder les règles de préférence (`policy.preference_scores`) sauf si les retirer
   améliore le NDCG@10 de validation de la configuration retenue de plus d'une erreur-type
   (ablation `--no-preferences`).
Le test est lu une seule fois, avec la configuration retenue et les références.

    python -m swiparis.ml.app_eval --split val --eta 0.2 --profile global
    python -m swiparis.ml.app_eval summarize outputs/app_eval_val_*.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import random

import numpy as np

from swiparis.ml import policy
from swiparis.ml.evaluation import _mean_ci, ranking_metrics, temporal_holdout
from swiparis.ml.items import encode_common, load_histories
from swiparis.ml.model import SessionProfile
from swiparis.ml.personas import PersonaModel, PersonaProfile
from swiparis.ml.prior import load_prior
from swiparis.ml.simulation import PROFILE_ITEMS, profile_similarity
from swiparis.ml.splits import load_splits

RANK_POOL = 300


def run_session(profile, phi, ids, similarity, space, user_seed: int):
    """Une séance avec la politique de l'application ; renvoie profil, historique, rejetées."""
    index = {item: i for i, item in enumerate(ids)}
    shown, history, rejected, offered = [], [], set(), 0
    while profile.n_updates < policy.MAX_ANSWERS and offered < policy.MAX_OFFERS:
        pair, _ = policy.choose_pair(profile, ids, phi, shown, offered, history, rejected, space,
                                     policy.seed(f"{user_seed}:{offered}"))
        if pair is None:
            break
        left, right = index[pair[0]], index[pair[1]]
        gap = similarity[left] - similarity[right]
        offered += 1
        shown.append(pair)
        if abs(gap) <= 1e-9:
            continue  # passage
        chosen, other = (pair[0], pair[1]) if gap > 0 else (pair[1], pair[0])
        history.append((chosen, other))
        rejected.add(other)
        profile.update(phi[index[chosen]], phi[index[other]])
    return profile, history, rejected


def metrics_from_ranking(ranked_ids, all_ids, relevant, rejected):
    """NDCG@10, Recall@20 (via ranking_metrics) et Hit@3 à partir de la liste classée."""
    scores = {item: -1e9 if item in rejected else -1e6 for item in all_ids}
    for position, item in enumerate(ranked_ids):
        scores[item] = 1e6 - position
    result = ranking_metrics(scores, relevant)
    result.pop("auc", None)
    result["hit3"] = float(bool(set(ranked_ids[:3]) & set(relevant)))
    return result


def evaluate(db_path, prior_path, personas_path, splits_path, split, eta, profile_kind,
             seed=42, users_limit=None, baselines=False, popularity_signal=True):
    """`popularity_signal=False` : comme dans l'application, le recommandeur ne voit pas la
    popularité des lieux (les annonces Paris Data n'ont aucune visite, leur bloc vaut 0)."""
    splits = load_splits(splits_path)
    train_users = {u for u, s in splits.items() if s == "train"}
    users = sorted(u for u, s in splits.items() if s == split)[:users_limit]
    fsq = encode_common(db_path, users=train_users, sources=["foursquare"], popularity_users=train_users)
    prior = load_prior(prior_path, fsq.space)
    personas = PersonaModel.load(personas_path, fsq.space) if profile_kind == "personas" else None
    taste = np.ones(fsq.space.dim, dtype=bool)
    taste[fsq.space.blocks["pop"]] = False
    popularity = np.array([fsq.counts.get(i, 0) for i in fsq.ids], dtype=float)
    seen_phi = fsq.phi.copy()
    if not popularity_signal:
        seen_phi[:, fsq.space.blocks["pop"]] = 0.0
    histories = load_histories(db_path, set(users))
    per_user, tops, answers = {}, {}, []
    extra = defaultdict(dict)
    for n, user in enumerate(users):
        support, relevant = temporal_holdout(histories[user], PROFILE_ITEMS)
        if not support or not relevant:
            continue
        keep = [i for i, item in enumerate(fsq.ids) if item not in set(support)]
        phi, ids = seen_phi[keep], [fsq.ids[i] for i in keep]
        similarity = profile_similarity(phi[:, taste], fsq.rows(list(support))[:, taste])
        profile = (PersonaProfile(personas, eta=eta, lam=policy.LAMBDA) if personas is not None
                   else SessionProfile(prior, eta=eta, lam=policy.LAMBDA))
        profile, history, rejected = run_session(profile, phi, ids, similarity, fsq.space, seed * 100003 + n)
        ranked = [item for item, _ in policy.recommend(profile, ids, phi, history, rejected, fsq.space,
                                                       policy.seed(f"{seed}:{user}"), k=20, pool=RANK_POOL)]
        per_user[user] = metrics_from_ranking(ranked, ids, relevant, rejected)
        tops[user] = ranked[:3]
        answers.append(profile.n_updates)
        if baselines:
            for name, scores in (("prior_only", phi @ prior), ("popularity", popularity[keep])):
                order = np.argsort(-scores, kind="stable")[:20]
                extra[name][user] = metrics_from_ranking([ids[i] for i in order], ids, relevant, set())
    rng = random.Random(seed)

    def summarize(results):
        values = defaultdict(list)
        for result in results.values():
            for metric, value in result.items():
                if value is not None:
                    values[metric].append(value)
        return {metric: _mean_ci(v, 1000, rng) for metric, v in values.items()}

    lists = list(tops.values())
    shared = [len(set(a) & set(b)) for a, b in itertools.combinations(lists, 2)]
    report = {"split": split, "eta": eta, "profile": profile_kind, "evaluated_users": len(per_user),
              "popularity_signal": popularity_signal,
              "policy": {"max_answers": policy.MAX_ANSWERS, "max_offers": policy.MAX_OFFERS,
                         "pair_pool": policy.PAIR_POOL, "rank_pool": RANK_POOL},
              "mean_answers": float(np.mean(answers)), "summary": summarize(per_user),
              "shared_top3_between_two_users": float(np.mean(shared)),
              "distinct_top3_lists": len({tuple(t) for t in lists}),
              "per_user": per_user}
    if baselines:
        report["baselines"] = {name: summarize(results) for name, results in extra.items()}
        report["baselines_per_user"] = extra
    return report


def paired(a: dict, b: dict, metric: str, seed=42):
    diffs = [a[u][metric] - b[u][metric] for u in a if u in b and a[u][metric] is not None and b[u][metric] is not None]
    return _mean_ci(diffs, 2000, random.Random(seed))


def summarize_files(paths) -> str:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    lines = ["| Profil | η | NDCG@10 [IC 95 %] | Recall@20 | Hit@3 | Activités communes (top 3) | Listes différentes |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(reports, key=lambda r: (r["profile"], r["eta"])):
        s = r["summary"]
        label = r["profile"] + ("" if r.get("preferences", True) else " sans préférences")
        label += "" if r.get("popularity_signal", True) else " (sans popularité)"
        lines.append(f"| {label} | {r['eta']:g} | {s['ndcg']['mean']:.4f} [{s['ndcg']['ci95_low']:.4f} ; "
                     f"{s['ndcg']['ci95_high']:.4f}] | {s['recall']['mean']:.4f} | {s['hit3']['mean']:.3f} | "
                     f"{r['shared_top3_between_two_users']:.2f} | {r['distinct_top3_lists']}/{r['evaluated_users']} |")
    for r in reports:
        for name, s in r.get("baselines", {}).items():
            lines.append(f"| {name} | – | {s['ndcg']['mean']:.4f} [{s['ndcg']['ci95_low']:.4f} ; "
                         f"{s['ndcg']['ci95_high']:.4f}] | {s['recall']['mean']:.4f} | {s['hit3']['mean']:.3f} | – | – |")
        if r.get("baselines"):
            break
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    summary = sub.add_parser("summarize")
    summary.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--prior", type=Path, default=Path("models/prior_w0_phi.json"))
    parser.add_argument("--personas", type=Path, default=Path("models/personas.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--eta", type=float, default=policy.DEFAULT_ETA)
    parser.add_argument("--profile", choices=("global", "personas"), default="global")
    parser.add_argument("--users", type=int)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--no-preferences", action="store_true", help="ablation : sans les règles de préférence")
    parser.add_argument("--no-popularity", action="store_true",
                        help="comme l'application : le recommandeur ne voit pas la popularité des lieux")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        print(summarize_files(args.reports))
        return 0
    if args.no_preferences:
        policy.PREFERENCE_WEIGHT = policy.PAIR_PREFERENCE_WEIGHT = 0.0
    report = evaluate(args.db, args.prior, args.personas, args.splits, args.split, args.eta, args.profile,
                      users_limit=args.users, baselines=args.baselines, popularity_signal=not args.no_popularity)
    report["preferences"] = not args.no_preferences
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("per_user")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
