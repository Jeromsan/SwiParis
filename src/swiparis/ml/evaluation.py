"""User-level splits and offline ranking evaluation for implicit visits.

The evaluator consumes per-user scores produced by a recommender. It does not
turn visits into pairwise preferences; unvisited items are treated as negatives
only for the explicitly labelled, weak implicit-feedback AUC proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def split_users(users: Iterable[str], seed: int = 42) -> dict[str, tuple[str, ...]]:
    """Deterministic 60/20/20 user split, without sharing users across sets."""
    unique_users = sorted(set(users))
    random.Random(seed).shuffle(unique_users)
    exact = [len(unique_users) * ratio for ratio in (0.6, 0.2, 0.2)]
    sizes = [math.floor(value) for value in exact]
    for index in sorted(range(3), key=lambda i: (-(exact[i] - sizes[i]), i))[
            :len(unique_users) - sum(sizes)]:
        sizes[index] += 1
    train_end = sizes[0]
    validation_end = train_end + sizes[1]
    return {
        "train": tuple(unique_users[:train_end]),
        "validation": tuple(unique_users[train_end:validation_end]),
        "test": tuple(unique_users[validation_end:]),
    }


def temporal_holdout(
    visits: Iterable[tuple[str, str]], profile_items: int = 5
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Use each item's first visit; earliest distinct items profile, later ones test."""
    if profile_items < 1:
        raise ValueError("profile_items must be at least 1")
    first_visit: dict[str, str] = {}
    for timestamp, item_id in visits:
        if item_id not in first_visit or timestamp < first_visit[item_id]:
            first_visit[item_id] = timestamp
    ordered = sorted((timestamp, item_id) for item_id, timestamp in first_visit.items())
    return (
        tuple(item_id for _, item_id in ordered[:profile_items]),
        tuple(item_id for _, item_id in ordered[profile_items:]),
    )


def ranking_metrics(
    scores: Mapping[str, float],
    relevant_items: Iterable[str],
    ndcg_k: int = 10,
    recall_k: int = 20,
) -> dict[str, float | None]:
    """Binary NDCG, Recall and AUC over the scored candidate catalog."""
    if ndcg_k < 1 or recall_k < 1:
        raise ValueError("metric cutoffs must be positive")
    relevant = set(relevant_items)
    if not relevant:
        return {"ndcg": None, "recall": None, "auc": None}
    missing = relevant - scores.keys()
    if missing:
        raise ValueError(f"Relevant items are missing scores: {sorted(missing)[:5]}")
    ranked = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    total_relevant = len(relevant)
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item_id in enumerate(ranked[:ndcg_k])
        if item_id in relevant
    )
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(min(ndcg_k, total_relevant)))
    retrieved = sum(item_id in relevant for item_id in ranked[:recall_k])
    negatives = len(scores) - total_relevant
    auc: float | None = None
    if negatives > 0:
        ordered_scores = sorted((score, item_id) for item_id, score in scores.items())
        positive_rank_sum = 0.0
        index = 0
        while index < len(ordered_scores):
            end = index + 1
            while (end < len(ordered_scores)
                   and ordered_scores[end][0] == ordered_scores[index][0]):
                end += 1
            average_rank = ((index + 1) + end) / 2
            positive_rank_sum += average_rank * sum(
                item_id in relevant for _, item_id in ordered_scores[index:end]
            )
            index = end
        positives = len(relevant)
        auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
        "recall": retrieved / total_relevant,
        "auc": auc,
    }


def _mean_ci(
    values: Sequence[float], samples: int, rng: random.Random
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    bootstrap_means = [
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    ]
    bootstrap_means.sort()
    return {
        "mean": sum(values) / len(values),
        "ci95_low": bootstrap_means[math.floor(0.025 * (samples - 1))],
        "ci95_high": bootstrap_means[math.ceil(0.975 * (samples - 1))],
    }


def _evaluate(database: Path, scores_path: Path, seed: int, profile_items: int,
              bootstrap_samples: int) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    db = sqlite3.connect(str(database))
    try:
        users = [row[0] for row in db.execute("SELECT DISTINCT user_id FROM visits")]
        partitions = split_users(users, seed)
        test_users = set(partitions["test"])
        histories: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for user_id, item_id, timestamp in db.execute(
                "SELECT user_id, item_id, utc_time FROM visits ORDER BY user_id, utc_time, item_id"):
            if user_id in test_users:
                histories[user_id].append((timestamp, item_id))
        candidate_items = {
            row[0] for row in db.execute(
                "SELECT item_id FROM items WHERE source = 'foursquare'")
        }
    finally:
        db.close()
    if not candidate_items:
        raise ValueError("The database contains no Foursquare candidates")

    labels: dict[str, set[str]] = {}
    profile_sets: dict[str, set[str]] = {}
    for user_id in sorted(test_users):
        support, relevant = temporal_holdout(histories[user_id], profile_items)
        if support and relevant:
            labels[user_id] = set(relevant)
            profile_sets[user_id] = set(support)
    if not labels:
        raise ValueError("No test user has both profile visits and later distinct visits")

    methods = ("active_score", "uniform_score")
    user_scores: dict[str, dict[str, dict[str, float]]] = {
        method: defaultdict(dict) for method in methods
    }
    with scores_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"user_id", "item_id", *methods}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Scores CSV must contain columns: {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            user_id, item_id = row["user_id"], row["item_id"]
            if user_id not in labels:
                continue
            if item_id not in candidate_items:
                raise ValueError(f"Unknown Foursquare item at scores CSV line {line}: {item_id}")
            if item_id in profile_sets[user_id]:
                raise ValueError(
                    f"Scores include a profile visit as a candidate at line {line}: {item_id}"
                )
            for method in methods:
                score = float(row[method])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite {method} at scores CSV line {line}")
                if item_id in user_scores[method][user_id]:
                    raise ValueError(f"Duplicate user/item at scores CSV line {line}")
                user_scores[method][user_id][item_id] = score

    metrics_by_method: dict[str, dict[str, list[float]]] = {}
    for method in methods:
        missing_users = set(labels) - user_scores[method].keys()
        if missing_users:
            raise ValueError(f"{method} is missing scores for {len(missing_users)} test users")
        for user_id in labels:
            if user_scores[method][user_id].keys() != candidate_items - profile_sets[user_id]:
                missing = candidate_items - profile_sets[user_id] - user_scores[method][user_id].keys()
                extra = user_scores[method][user_id].keys() - candidate_items
                raise ValueError(
                    f"{method} candidate coverage mismatch for a test user "
                    f"(missing {len(missing)}, extra {len(extra)})"
                )
        metrics_by_method[method] = defaultdict(list)
        for user_id in labels:
            result = ranking_metrics(user_scores[method][user_id], labels[user_id])
            for metric, value in result.items():
                if value is not None:
                    metrics_by_method[method][metric].append(value)
    rng = random.Random(seed)
    report = {
        "protocol": {
            "user_split": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "seed": seed,
            "profile_items": profile_items,
            "test_users": len(partitions["test"]),
            "evaluated_users": len(labels),
            "users_without_later_distinct_items": len(test_users) - len(labels),
            "candidate_items": len(candidate_items),
            "auc_interpretation": "proxy: unvisited items are treated as negatives",
        },
        "results": {},
        "paired_delta_active_minus_uniform": {},
    }
    per_user: dict[str, dict[str, dict[str, float]]] = {
        method: {} for method in methods
    }
    for method in methods:
        report["results"][method] = {}
        for user_id in labels:
            per_user[method][user_id] = ranking_metrics(
                user_scores[method][user_id], labels[user_id]
            )
        for metric, values in metrics_by_method[method].items():
            report["results"][method][metric] = _mean_ci(values, bootstrap_samples, rng)
    for metric in ("ndcg", "recall", "auc"):
        differences = [
            per_user["active_score"][user_id][metric]
            - per_user["uniform_score"][user_id][metric]
            for user_id in labels
            if (per_user["active_score"][user_id][metric] is not None
                and per_user["uniform_score"][user_id][metric] is not None)
        ]
        report["paired_delta_active_minus_uniform"][metric] = _mean_ci(
            differences, bootstrap_samples, rng
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate active and uniform recommendation scores against future visits."
    )
    parser.add_argument("--database", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--scores", type=Path, required=True,
                        help="CSV columns: user_id,item_id,active_score,uniform_score")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile-items", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    report = _evaluate(args.database, args.scores, args.seed,
                       args.profile_items, args.bootstrap_samples)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["protocol"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
