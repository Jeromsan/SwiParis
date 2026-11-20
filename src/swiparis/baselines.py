"""Baselines de référence évaluées sur les paires : aléatoire et popularité.

Précision par paires : part des paires où le lieu choisi a un meilleur score que le lieu
rejeté (égalité = 0,5). ATTENTION : les négatifs sont tirés proportionnellement à la
popularité (calculée sur les utilisateurs d'entraînement), donc le rejeté est en moyenne plus
populaire que le choisi pour les utilisateurs val/test : la baseline « popularité » peut tomber
SOUS 0,5. Ce chiffre ne prouve donc pas qu'on la bat. La comparaison à citer est le NDCG@10 /
Recall@20 sur des candidats éligibles communs (script d'évaluation). Ici : contrôle rapide
et suivi pendant l'entraînement.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pairs import load_foursquare, read_table


def popularity_scores(venues, checkins, train_users, smoothing=1.0):
    """Part lissée des visites des utilisateurs d'ENTRAÎNEMENT : (n_v + a) / (N + a * V)."""
    train = checkins[checkins["user_id"].isin(train_users)]
    counts = venues["venue_id"].map(train["venue_id"].value_counts()).fillna(0).to_numpy(dtype=float)
    return (counts + smoothing) / (counts.sum() + smoothing * len(counts))


def random_scores(venues, seed=0):
    return np.random.default_rng([seed, 3]).random(len(venues))


def pairwise_accuracy(scores, venues, pairs):
    index = pd.Series(np.arange(len(venues)), index=venues["venue_id"])
    chosen = scores[index.loc[pairs["chosen_venue_id"]].to_numpy()]
    rejected = scores[index.loc[pairs["rejected_venue_id"]].to_numpy()]
    return float(((chosen > rejected) + 0.5 * (chosen == rejected)).mean()) if len(pairs) else float("nan")


def report(venues, checkins, pairs, seed=0):
    train_users = set(pairs.loc[pairs["split"] == "train", "user_id"])
    methods = {"random": random_scores(venues, seed),
               "popularity": popularity_scores(venues, checkins, train_users)}
    out = {}
    for name, scores in methods.items():
        out[name] = {split: {"pairs": int((pairs["split"] == split).sum()),
                             "pairwise_accuracy": pairwise_accuracy(scores, venues, pairs[pairs["split"] == split])}
                     for split in ("train", "val", "test")}
        out[name]["val_by_negative_type"] = {
            kind: pairwise_accuracy(scores, venues, part)
            for kind, part in pairs[pairs["split"] == "val"].groupby("negative_type")}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    venues, checkins = load_foursquare(args.db)
    result = report(venues, checkins, read_table(args.pairs), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
