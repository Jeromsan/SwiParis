"""Courbes de convergence du prior w0 (rapport de semaine 3, Q1).

Rejoue l'ajustement de `swiparis.ml.prior` (mêmes données, mêmes paires, mêmes lambdas) en
enregistrant la perte logistique d'entraînement et de validation à CHAQUE itération de Newton,
puis la perte finale pour chaque lambda. Le split test n'est pas lu.

    python scripts/plot_prior_curves.py --db data/processed/common.sqlite --pairs data/processed/pairs.csv \
        --output docs/resultats_ml/prior_loss_curve.png
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swiparis.ml.items import encode_common, load_histories
from swiparis.ml.model import sigmoid
from swiparis.ml.prior import LAMBDAS, accuracy, block_mask, deltas, logistic_loss, read_pairs


def newton_history(train, val, lam, max_iter=50, tol=1e-10):
    n, d = train.shape
    w = np.zeros(d)
    objective = lambda v: logistic_loss(v, train) + lam / 2 * v @ v
    rows = [(0, logistic_loss(w, train), logistic_loss(w, val), accuracy(w, val))]
    for it in range(1, max_iter + 1):
        p = sigmoid(train @ w)
        gradient = -(train.T @ (1 - p)) / n + lam * w
        hessian = (train.T * (p * (1 - p))) @ train / n + lam * np.eye(d)
        step = np.linalg.solve(hessian, gradient)
        size, current = 1.0, objective(w)
        while objective(w - size * step) > current and size > 1e-8:
            size /= 2
        w = w - size * step
        rows.append((it, logistic_loss(w, train), logistic_loss(w, val), accuracy(w, val)))
        if abs(gradient @ step) < tol:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lam", type=float, default=1e-4, help="lambda retenu (min de la perte de validation)")
    args = parser.parse_args()
    pairs = read_pairs(args.pairs)
    train_users = {p["user_id"] for p in pairs if p["split"] == "train"}
    encoded = encode_common(args.db, users=train_users, sources=["foursquare"], popularity_users=train_users)
    own = Counter((u, i) for u, visits in load_histories(args.db, train_users).items() for _, i in visits)
    mask = block_mask(encoded.space)
    train, val = deltas(pairs, encoded, "train", own) * mask, deltas(pairs, encoded, "val") * mask
    history = newton_history(train, val, args.lam)
    by_lambda = [(lam, newton_history(train, val, lam)[-1]) for lam in LAMBDAS]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    its = [r[0] for r in history]
    left.plot(its, [r[1] for r in history], "o-", label="train")
    left.plot(its, [r[2] for r in history], "s--", label="validation")
    left.set_xlabel("Newton iteration")
    left.set_xticks(its)
    left.set_ylabel("pairwise logistic loss")
    left.set_title(f"Convergence of the prior w0 (lambda={args.lam:g})")
    left.legend()
    right.semilogx([l for l, _ in by_lambda], [r[1] for _, r in by_lambda], "o-", label="train")
    right.semilogx([l for l, _ in by_lambda], [r[2] for _, r in by_lambda], "s--", label="validation")
    right.set_xlabel("lambda (L2)")
    right.set_title("Final loss by lambda")
    right.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(json.dumps({"iterations": [[r[0], round(r[1], 5), round(r[2], 5), round(r[3], 4)] for r in history],
                      "finite": bool(np.isfinite([r[1:] for r in history]).all()),
                      "n_train_pairs": int(len(train)), "n_val_pairs": int(len(val))}))


if __name__ == "__main__":
    main()
