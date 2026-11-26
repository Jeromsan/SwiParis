"""Comparer les réglages de séance (η) avec les vraies réponses du pilote.

Pendant le pilote, lancer l'API avec par exemple SWIPARIS_SESSION_ETA_AB="0.2,0.5" : chaque
séance reçoit une des valeurs, tirée à partir de son identifiant. Le η d'une séance se
retrouve ensuite à partir de la version enregistrée avec elle (`policy.eta_from_version`) ;
des séances lancées à des jours différents avec SWIPARIS_SESSION_ETA différents se
comparent de la même façon.

Mesures par réglage, sur les recommandations affichées : part des recommandations
ouvertes, part des recommandations gardées, part des séances avec au moins une sortie
gardée. Écart entre deux réglages : moyenne par séance, intervalle bootstrap à 95 %.

    python -m swiparis.ml.pilot_eta --logs data/pilot/interactions.sqlite
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import random
import sqlite3

from swiparis.ml.evaluation import _mean_ci
from swiparis.ml.policy import eta_from_version


def session_stats(logs_path, include_test: bool = False) -> list[dict]:
    """Une ligne par séance : η, recommandations affichées, ouvertes et gardées."""
    uri = f"{Path(logs_path).resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        origins = ("pilot", "test") if include_test else ("pilot",)
        sessions = db.execute(
            f"SELECT session_id, model_version FROM sessions WHERE data_origin IN ({','.join('?' * len(origins))})",
            origins).fetchall()
        shown = defaultdict(set)
        for row in db.execute("SELECT session_id, impression_id FROM recommendations_shown"):
            shown[row["session_id"]].add(row["impression_id"])
        actions = defaultdict(set)
        for row in db.execute("SELECT impression_id, action FROM recommendation_actions"):
            actions[row["action"]].add(row["impression_id"])
        answers = dict(db.execute("SELECT session_id, count(*) FROM comparisons WHERE choice IN ('left','right') "
                                  "GROUP BY session_id").fetchall())
    result = []
    for session in sessions:
        impressions = shown.get(session["session_id"], set())
        result.append({"session_id": session["session_id"],
                       "eta": eta_from_version(session["model_version"], session["session_id"]),
                       "answers": answers.get(session["session_id"], 0), "shown": len(impressions),
                       "opened": len(impressions & actions["open"]), "saved": len(impressions & actions["save"])})
    return result


def report(logs_path, include_test: bool = False, seed: int = 42) -> dict:
    stats = [s for s in session_stats(logs_path, include_test) if s["eta"] is not None]
    by_eta = defaultdict(list)
    for s in stats:
        by_eta[s["eta"]].append(s)
    rng = random.Random(seed)
    arms = {}
    for eta, rows in sorted(by_eta.items()):
        with_recos = [r for r in rows if r["shown"]]
        shown = sum(r["shown"] for r in with_recos)
        arms[f"{eta:g}"] = {
            "sessions": len(rows), "sessions_with_recommendations": len(with_recos),
            "mean_answers": sum(r["answers"] for r in rows) / len(rows),
            "recommendations_shown": shown,
            "open_rate": sum(r["opened"] for r in with_recos) / shown if shown else None,
            "save_rate": sum(r["saved"] for r in with_recos) / shown if shown else None,
            "sessions_with_a_save": sum(r["saved"] > 0 for r in with_recos) / len(with_recos) if with_recos else None,
            "save_rate_per_session": _mean_ci([r["saved"] / r["shown"] for r in with_recos], 2000, rng),
        }
    comparison = None
    if len(arms) == 2:
        (a, rows_a), (b, rows_b) = sorted(by_eta.items())
        rates_a = [r["saved"] / r["shown"] for r in rows_a if r["shown"]]
        rates_b = [r["saved"] / r["shown"] for r in rows_b if r["shown"]]
        if rates_a and rates_b:
            diffs = []
            for _ in range(2000):
                sample_a = [rng.choice(rates_a) for _ in rates_a]
                sample_b = [rng.choice(rates_b) for _ in rates_b]
                diffs.append(sum(sample_b) / len(sample_b) - sum(sample_a) / len(sample_a))
            diffs.sort()
            comparison = {"difference": f"η={b:g} − η={a:g}", "metric": "save_rate_per_session",
                          "mean": sum(rates_b) / len(rates_b) - sum(rates_a) / len(rates_a),
                          "ci95_low": diffs[49], "ci95_high": diffs[1949]}
    return {"sessions": len(stats), "arms": arms, "comparison": comparison,
            "note": "Peu de séances = intervalle large : ne pas conclure tant que l'intervalle contient 0."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", type=Path, default=Path("data/pilot/interactions.sqlite"))
    parser.add_argument("--include-test", action="store_true", help="compter aussi les séances de test")
    args = parser.parse_args(argv)
    print(json.dumps(report(args.logs, args.include_test), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
