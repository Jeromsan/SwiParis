"""Découpage unique des utilisateurs, partagé par l'entraînement et l'évaluation.

Source de vérité : `swiparis.ml.evaluation.split_users` (graine 42, 60/20/20 par
utilisateur), celle utilisée par le script d'évaluation. Ce module l'exporte au
format attendu par `python -m swiparis.pairs --splits` : CSV `user_id,split`
avec les valeurs train / val / test. Sans ce fichier, `swiparis.pairs` tire son
propre découpage, différent : des utilisateurs de test entreraient alors dans
l'entraînement du prior.

    python -m swiparis.ml.splits --db data/processed/common.sqlite --output data/processed/splits.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import csv
from pathlib import Path
import sqlite3

from swiparis.ml.evaluation import split_users

CSV_NAMES = {"train": "train", "validation": "val", "test": "test"}


def common_splits(db_path, seed: int = 42) -> dict[str, str]:
    """{user_id: train | val | test}, pour tous les utilisateurs ayant des visites."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        users = [row[0] for row in db.execute("SELECT DISTINCT user_id FROM visits")]
    return {user: CSV_NAMES[part] for part, members in split_users(users, seed).items() for user in members}


def write_splits(splits: dict[str, str], path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["user_id", "split"])
        writer.writerows(sorted(splits.items()))


def load_splits(path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return {row["user_id"]: row["split"] for row in csv.DictReader(handle)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    splits = common_splits(args.db, args.seed)
    write_splits(splits, args.output, args.force)
    print(dict(Counter(splits.values())), "->", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
