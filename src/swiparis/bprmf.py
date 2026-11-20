"""BPR-MF : baseline de filtrage collaboratif (niveau 2 de l'échelle de baselines).

Apprentissage : vecteurs utilisateur/item + biais d'item sur les visites des utilisateurs
d'ENTRAÎNEMENT (perte BPR : lieu visité > lieu non visité tiré au hasard). Les utilisateurs
val/test sont nouveaux : leur vecteur est estimé par « fold-in » à partir des premières visites
(le profil), avec les item vectors figés. Évaluation : précision par paires sur les paires
dont la visite est APRÈS le profil (visit_rank >= ceil(fraction * n_visits)).
Mêmes limites que baselines.py : négatifs biaisés vers la popularité, donc à compléter par le
NDCG@10 / Recall@20 du script d'évaluation (utiliser BPRMF.scores(profil) comme score).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .baselines import pairwise_accuracy, popularity_scores, random_scores
from .pairs import load_foursquare, read_table


class BPRMF:
    def __init__(self, venue_ids, item_vectors, item_bias, seed=42):
        self.venue_ids = list(venue_ids)
        self.index = {v: i for i, v in enumerate(self.venue_ids)}
        self.q = np.asarray(item_vectors, dtype=np.float32)
        self.b = np.asarray(item_bias, dtype=np.float32)
        self.seed = seed

    def fold_in(self, profile_ids, steps=60, lr=0.1, l2=1e-3):
        """Vecteur utilisateur appris sur le profil, item vectors figés (mêmes pertes BPR)."""
        pos = np.array(sorted({self.index[v] for v in profile_ids if v in self.index}), dtype=np.int64)
        p = torch.zeros(self.q.shape[1], requires_grad=True)
        if len(pos) == 0:
            return p.detach().numpy()
        q, b = torch.from_numpy(self.q), torch.from_numpy(self.b)
        rng = np.random.default_rng([self.seed, 5])
        neg = torch.from_numpy(rng.integers(0, len(self.venue_ids), size=(steps, len(pos))))
        pos_t = torch.from_numpy(pos)
        optimizer = torch.optim.Adam([p], lr=lr)
        for step in range(steps):
            x_pos = q[pos_t] @ p + b[pos_t]
            x_neg = q[neg[step]] @ p + b[neg[step]]
            loss = F.softplus(-(x_pos - x_neg)).mean() + l2 * p.pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return p.detach().numpy()

    def scores(self, profile_ids):
        return self.q @ self.fold_in(profile_ids) + self.b

    def save(self, path):
        np.savez(path, venue_ids=np.array(self.venue_ids), item_vectors=self.q, item_bias=self.b)

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        return cls(data["venue_ids"].astype(str), data["item_vectors"], data["item_bias"])


def fit(venues, checkins, train_users, dim=32, epochs=30, lr=0.05, l2=1e-4, batch_size=4096, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 4])
    index = {v: i for i, v in enumerate(venues["venue_id"])}
    users = sorted(train_users)
    user_index = {u: i for i, u in enumerate(users)}
    train = checkins[checkins["user_id"].isin(train_users)]
    pairs_ui = np.unique(np.stack([train["user_id"].map(user_index).to_numpy(),
                                   train["venue_id"].map(index).to_numpy()], axis=1), axis=0)
    n_items = len(venues)
    visited = np.zeros((len(users), n_items), dtype=bool)
    visited[pairs_ui[:, 0], pairs_ui[:, 1]] = True
    p = torch.nn.Parameter(torch.randn(len(users), dim) * 0.1)
    q = torch.nn.Parameter(torch.randn(n_items, dim) * 0.1)
    b = torch.nn.Parameter(torch.zeros(n_items))
    optimizer = torch.optim.Adam([p, q, b], lr=lr)
    u_all = torch.from_numpy(pairs_ui[:, 0])
    i_all = torch.from_numpy(pairs_ui[:, 1])
    for _ in range(epochs):
        j_all = rng.integers(0, n_items, size=len(pairs_ui))
        for _retry in range(10):  # évite de tirer un lieu déjà visité par l'utilisateur
            clash = visited[pairs_ui[:, 0], j_all]
            if not clash.any():
                break
            j_all[clash] = rng.integers(0, n_items, size=int(clash.sum()))
        j_t = torch.from_numpy(j_all)
        order = torch.from_numpy(rng.permutation(len(pairs_ui)))
        for start in range(0, len(order), batch_size):
            take = order[start:start + batch_size]
            u, i, j = u_all[take], i_all[take], j_t[take]
            x = (p[u] * (q[i] - q[j])).sum(-1) + b[i] - b[j]
            reg = l2 * (p[u].pow(2).sum() + q[i].pow(2).sum() + q[j].pow(2).sum()) / len(take)
            loss = F.softplus(-x).mean() + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return BPRMF(venues["venue_id"], q.detach().numpy(), b.detach().numpy(), seed)


def evaluate_split(model, venues, checkins, pairs, split, profile_fraction=0.5, novel_only=False):
    """Précision par paires sur les visites postérieures au profil, pour BPR-MF, popularité, aléatoire.
    novel_only=True ne garde que les visites à un lieu ABSENT du profil (pas de re-visite)."""
    train_users = set(pairs.loc[pairs["split"] == "train", "user_id"])
    pop = popularity_scores(venues, checkins, train_users)
    rnd = random_scores(venues)
    history = checkins.groupby("user_id")["venue_id"].apply(list)
    index = pd.Series(np.arange(len(venues)), index=venues["venue_id"])
    hits = {"bpr_mf": 0.0, "popularity": 0.0, "random": 0.0}
    total = 0
    for user, group in pairs[pairs["split"] == split].groupby("user_id"):
        cutoff = math.ceil(profile_fraction * group["n_visits"].iloc[0])
        target = group[group["visit_rank"] >= cutoff]
        if novel_only:
            target = target[~target["chosen_venue_id"].isin(set(history[user][:cutoff]))]
        if target.empty:
            continue
        scores = {"bpr_mf": model.scores(history[user][:cutoff]), "popularity": pop, "random": rnd}
        c = index.loc[target["chosen_venue_id"]].to_numpy()
        r = index.loc[target["rejected_venue_id"]].to_numpy()
        for name, s in scores.items():
            hits[name] += float(((s[c] > s[r]) + 0.5 * (s[c] == s[r])).sum())
        total += len(target)
    return {"pairs": total, **{k: (v / total if total else float("nan")) for k, v in hits.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="rapport JSON")
    parser.add_argument("--save", type=Path, help="fichier .npz du modèle")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args(argv)
    venues, checkins = load_foursquare(args.db)
    pairs = read_table(args.pairs)
    train_users = set(pairs.loc[pairs["split"] == "train", "user_id"])
    model = fit(venues, checkins, train_users, args.dim, args.epochs, args.lr, args.l2, seed=args.seed)
    report = {"config": {k: getattr(args, k) for k in ("dim", "epochs", "lr", "l2", "seed")},
              "val": evaluate_split(model, venues, checkins, pairs, "val"),
              "val_novel_only": evaluate_split(model, venues, checkins, pairs, "val", novel_only=True)}
    if args.eval_test:
        report["test"] = evaluate_split(model, venues, checkins, pairs, "test")
        report["test_novel_only"] = evaluate_split(model, venues, checkins, pairs, "test", novel_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.save:
        model.save(args.save)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
