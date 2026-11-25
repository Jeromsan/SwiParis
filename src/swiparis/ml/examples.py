"""Profils exemples (rapport de semaine 3) et écarts appariés avec les baselines.

Règle de choix fixée avant de regarder les résultats : parmi les utilisateurs
test évalués, on prend les 3 catégories les plus fréquentes comme catégorie
dominante des 5 lieux du profil, et pour chacune l'utilisateur au plus petit
identifiant. Configuration figée sur la validation (voir docs/modele.md).

    python -m swiparis.ml.examples --output docs/profils_exemples.md --report docs/resultats_ml/examples_test.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import random
import sqlite3

import numpy as np

from swiparis.ml.evaluation import _mean_ci
from swiparis.ml.items import encode_common
from swiparis.ml.selection import rank
from swiparis.ml.simulation import run
from swiparis.ml.splits import load_splits

ROWS, COLS = ("nord", "centre", "sud"), ("ouest", "centre", "est")


def zone_label(space, vector) -> str:
    cell = int(np.argmax(vector[space.blocks["zone"]]))
    row, col = divmod(cell, space.grid[1])
    return "centre" if (row, col) == (1, 1) else f"{ROWS[row]}-{COLS[col]}".replace("-centre", "").replace("centre-", "")


def categories(space, vector) -> str:
    block = vector[space.blocks["cat"]]
    return " + ".join(space.categories[j] for j in np.nonzero(block)[0]) or "?"


def paired(per_user, a, b, metric, seed=42):
    diffs = [per_user[a][u][metric] - per_user[b][u][metric] for u in per_user[a]
             if per_user[a][u][metric] is not None and per_user[b][u][metric] is not None]
    return _mean_ci(diffs, 2000, random.Random(seed))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--prior", type=Path, default=Path("models/prior_w0_phi.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("docs/profils_exemples.md"))
    parser.add_argument("--report", type=Path, default=Path("docs/resultats_ml/examples_test.json"))
    args = parser.parse_args(argv)

    report, per_user, sessions, fsq = run(args.db, args.prior, args.splits, "test", session_kw={"eta": args.eta})
    S = fsq.space
    deltas = {f"{a}-{b}": {m: paired(per_user, a, b, m) for m in ("ndcg", "recall")}
              for a, b in (("active", "popularity"), ("prior_only", "popularity"), ("active", "uniform"),
                           ("active", "prior_only"))}
    users = sorted(per_user["active"])
    dominant = {u: Counter(categories(S, fsq.vector(i)) for i in sessions[u]["support"]).most_common(1)[0][0]
                for u in users}
    chosen = [min(u for u in users if dominant[u] == c) for c, _ in Counter(dominant.values()).most_common(3)]

    splits = load_splits(args.splits)
    train_users = {u for u, s in splits.items() if s == "train"}
    paris = encode_common(args.db, users=train_users, sources=["paris_data"], popularity_users=train_users)
    with closing(sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True)) as db:
        titles = dict(db.execute("SELECT item_id, title FROM items WHERE source='paris_data'"))
    pop_rank = {item: r + 1 for r, item in enumerate(sorted(fsq.ids, key=lambda i: -fsq.counts.get(i, 0)))}

    lines = ["# Profils exemples (rapport de semaine 3)", "",
             "Généré par `python -m swiparis.ml.examples` sur les utilisateurs **test**, après avoir figé",
             f"la configuration sur la validation (prior `{args.prior}`, η = {args.eta}). Règle de choix des",
             "3 utilisateurs, fixée avant de voir leurs résultats : les 3 catégories dominantes les plus",
             "fréquentes dans les profils test, et pour chacune l'utilisateur au plus petit identifiant.",
             "Les réponses aux paires sont **simulées** à partir des 5 lieux du profil (protocole de",
             "`swiparis.ml.simulation`) ; les visites suivantes ne servent qu'à vérifier les recommandations.",
             "Les lieux Foursquare n'ont pas de nom dans la base : ils sont décrits par catégorie et zone.", ""]
    examples = []
    for n, user in enumerate(chosen, 1):
        support, relevant = sessions[user]["support"], set(sessions[user]["relevant"])
        profile, log = sessions[user]["active"]
        keep = [i for i in fsq.ids if i not in set(support)]
        scores = profile.scores(fsq.rows(keep))
        top = [keep[i] for i in np.argsort(-scores, kind="stable")[:10]]
        hits = [i for i in top if i in relevant]
        paris_top = rank(profile, paris.phi, paris.ids, k=5)
        metrics = {m: per_user[m][user] for m in ("active", "prior_only", "popularity")}
        lines += [f"## Utilisateur {n} : `{user}` (profil dominé par « {dominant[user]} »)", "",
                  "**Profil (5 premiers lieux)** : " + " ; ".join(
                      f"{categories(S, fsq.vector(i))} ({zone_label(S, fsq.vector(i))})" for i in support), "",
                  "**Paires proposées par la sélection active et réponses simulées** :", ""]
        for (a, b), choice in log:
            pick = {"left": "← gauche", "right": "droite →", "skip": "passe"}[choice]
            lines.append(f"- {categories(S, fsq.vector(a))} ({zone_label(S, fsq.vector(a))}) vs "
                         f"{categories(S, fsq.vector(b))} ({zone_label(S, fsq.vector(b))}) : {pick}")
        lines += ["", "**Top 10 Foursquare après la session** :", "",
                  "| Rang | Catégorie | Zone | Rang de popularité | Visité plus tard |", "|---|---|---|---|---|"]
        for r, item in enumerate(top, 1):
            lines.append(f"| {r} | {categories(S, fsq.vector(item))} | {zone_label(S, fsq.vector(item))} | "
                         f"{pop_rank[item]} / {len(fsq.ids)} | {'oui' if item in relevant else ''} |")
        lines += ["", f"{len(hits)} des 10 lieux recommandés font partie des {len(relevant)} lieux qu'il a "
                  f"visités ensuite. NDCG@10 : session {metrics['active']['ndcg']:.3f}, w₀ seul "
                  f"{metrics['prior_only']['ndcg']:.3f}, popularité {metrics['popularity']['ndcg']:.3f}.", "",
                  "**Ce que la démo lui proposerait (catalogue Paris Data, avec diversité)** :", ""]
        lines += [f"{r}. {titles[item]} ({categories(paris.space, paris.vector(item))})"
                  for r, (item, _) in enumerate(paris_top, 1)]
        lines.append("")
        examples.append({"user_id": user, "dominant": dominant[user], "hits_top10": len(hits),
                         "future_items": len(relevant), "metrics": metrics,
                         "paris_top5": [titles[i] for i, _ in paris_top]})
    lines += ["## Lecture honnête", "",
              "Sur l'ensemble des utilisateurs test, les écarts appariés (moyenne et IC 95 % bootstrap) sont :", ""]
    for name, values in deltas.items():
        lines.append(f"- {name} : NDCG@10 {values['ndcg']['mean']:+.4f} [{values['ndcg']['ci95_low']:+.4f}, "
                     f"{values['ndcg']['ci95_high']:+.4f}], Recall@20 {values['recall']['mean']:+.4f} "
                     f"[{values['recall']['ci95_low']:+.4f}, {values['recall']['ci95_high']:+.4f}]")
    lines += ["", "Les recommandations restent des lieux très populaires (rang de popularité moyen du top 10 "
              "dans le premier centile) : le modèle classe d'abord par popularité, puis départage par "
              "catégorie et zone. Voir docs/modele.md pour l'analyse.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"summary": report, "paired_deltas": deltas, "examples": examples},
                                      indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"chosen": chosen, "paired_deltas": deltas}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
