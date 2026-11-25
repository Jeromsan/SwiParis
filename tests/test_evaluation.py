from contextlib import closing
import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from swiparis.ml.evaluation import (_evaluate, ranking_metrics, split_users,
                                    temporal_holdout)


class EvaluationTest(unittest.TestCase):
    def test_user_split_is_deterministic_disjoint_and_rounded(self):
        users = [f"user-{index}" for index in range(10)]
        first = split_users(users, seed=7)
        self.assertEqual(first, split_users(reversed(users), seed=7))
        self.assertEqual(tuple(map(len, first.values())), (6, 2, 2))
        self.assertEqual(len(set().union(*map(set, first.values()))), 10)
        self.assertFalse(set(first["train"]) & set(first["test"]))

    def test_temporal_holdout_deduplicates_before_splitting(self):
        profile, future = temporal_holdout([
            ("2020-01-03", "later"),
            ("2020-01-01", "first"),
            ("2020-01-02", "second"),
            ("2020-01-04", "later"),
        ], profile_items=2)
        self.assertEqual(profile, ("first", "second"))
        self.assertEqual(future, ("later",))

    def test_ranking_metrics_use_binary_future_relevance(self):
        result = ranking_metrics(
            {"a": 0.9, "b": 0.8, "c": 0.1},
            {"a", "c"},
            ndcg_k=2,
            recall_k=2,
        )
        self.assertAlmostEqual(result["ndcg"], 1 / (1 + 1 / 1.584962500721156))
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["auc"], 0.5)

    def test_missing_relevant_score_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "missing scores"):
            ranking_metrics({"known": 1.0}, {"unscored"})

    def test_end_to_end_evaluator_uses_test_users_and_paired_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "data.sqlite"
            score_file = root / "scores.csv"
            with closing(sqlite3.connect(database)) as db:
                db.executescript("""
                    CREATE TABLE items(item_id TEXT PRIMARY KEY, source TEXT NOT NULL);
                    CREATE TABLE visits(user_id TEXT, item_id TEXT, utc_time TEXT);
                """)
                db.executemany("INSERT INTO items VALUES (?, 'foursquare')",
                               [(f"item-{index}",) for index in range(4)])
                for user_index in range(10):
                    for visit_index in range(3):
                        db.execute("INSERT INTO visits VALUES (?, ?, ?)", (
                            f"user-{user_index}",
                            f"item-{visit_index}",
                            f"2020-01-0{visit_index + 1}T00:00:00Z",
                        ))
                db.commit()
            test_users = set(split_users(
                [f"user-{index}" for index in range(10)], seed=7
            )["test"])
            with score_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("user_id", "item_id", "active_score", "uniform_score"))
                for user_id in test_users:
                    for item_id, active, uniform in (
                        ("item-1", 0.9, 0.5),
                        ("item-2", 0.8, 0.5),
                        ("item-3", 0.1, 0.5),
                    ):
                        writer.writerow((user_id, item_id, active, uniform))

            report = _evaluate(database, score_file, seed=7, profile_items=1,
                               bootstrap_samples=50)
            self.assertEqual(report["protocol"]["test_users"], 2)
            self.assertEqual(report["protocol"]["evaluated_users"], 2)
            self.assertEqual(report["results"]["active_score"]["auc"]["mean"], 1.0)
            self.assertEqual(report["results"]["uniform_score"]["auc"]["mean"], 0.5)
            self.assertEqual(
                report["paired_delta_active_minus_uniform"]["auc"]["mean"], 0.5
            )


if __name__ == "__main__":
    unittest.main()
