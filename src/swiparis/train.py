"""Entraînement hors ligne du modèle global (prior w0) par perte logistique sur paires.

P(chosen > rejected) = sigmoid(s(chosen) - s(rejected)), avec s(i | moment) = z(i) . (w_g + t[moment]),
z(i) = normalize(W phi(i)). phi(i) = catégorie commune (one-hot) + lat/lon standardisées
(+ features externes optionnelles, par ex. embeddings E5, via --features).
Le terme personnel de session (mise à jour par comparaison) est dans le module de scoring :
ici on apprend seulement le prior hors ligne. Le split test n'est évalué qu'avec --eval-test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import pairwise_accuracy, popularity_scores
from .pairs import TIME_BUCKETS, load_foursquare, load_taxonomy, read_table, sha256_file


def build_features(venues, taxonomy, external_path=None):
    category = venues["category"].map(lambda c: taxonomy.get(c, c)).to_numpy()
    cats = sorted(set(category))
    onehot = (category[:, None] == np.array(cats)[None, :]).astype(np.float32)
    coords = venues[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    coords = (coords - coords.mean(0)) / (coords.std(0) + 1e-8)
    blocks = [onehot, coords]
    if external_path:
        data = np.load(external_path, allow_pickle=False)
        ext = pd.DataFrame(data["X"], index=data["venue_ids"].astype(str))
        missing = set(venues["venue_id"]) - set(ext.index)
        if missing:
            raise ValueError(f"{len(missing)} venues missing from external features")
        blocks.append(ext.loc[venues["venue_id"]].to_numpy(dtype=np.float32))
    return np.hstack(blocks), cats


class PairModel(nn.Module):
    def __init__(self, d_in, dim):
        super().__init__()
        self.proj = nn.Linear(d_in, dim, bias=False)
        self.w_global = nn.Parameter(torch.randn(dim) * 0.1)
        self.w_time = nn.Embedding(len(TIME_BUCKETS), dim)
        nn.init.zeros_(self.w_time.weight)

    def item_vectors(self, x):
        return F.normalize(self.proj(x), dim=-1)

    def score(self, z, bucket):
        return (z * (self.w_global + self.w_time(bucket))).sum(-1)


def tensors(pairs, index):
    return (torch.tensor(pairs["chosen_venue_id"].map(index).to_numpy(), dtype=torch.long),
            torch.tensor(pairs["rejected_venue_id"].map(index).to_numpy(), dtype=torch.long),
            torch.tensor(pairs["time_bucket"].map({b: i for i, b in enumerate(TIME_BUCKETS)}).to_numpy(),
                         dtype=torch.long))


def evaluate(model, x, data, l2):
    chosen, rejected, bucket = data
    with torch.no_grad():
        z = model.item_vectors(x)
        margin = model.score(z[chosen], bucket) - model.score(z[rejected], bucket)
        loss = F.softplus(-margin).mean().item()
        accuracy = ((margin > 0).float() + 0.5 * (margin == 0).float()).mean().item()
    return loss, accuracy


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(venues, checkins, pairs, taxonomy, out_root, config, pairs_sha256, features_path=None,
          eval_test=False):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    x_np, categories = build_features(venues, taxonomy, features_path)
    x = torch.tensor(x_np)
    index = {v: i for i, v in enumerate(venues["venue_id"])}
    split = {name: tensors(pairs[pairs["split"] == name], index) for name in ("train", "val", "test")}
    if len(split["train"][0]) == 0 or len(split["val"][0]) == 0:
        raise ValueError("need non-empty train and val pairs")
    model = PairModel(x.shape[1], config["dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    generator = torch.Generator().manual_seed(config["seed"])
    n_train = len(split["train"][0])
    history, best, best_state, since_best = [], (float("inf"), 0), None, 0
    for epoch in range(1, config["epochs"] + 1):
        order = torch.randperm(n_train, generator=generator)
        losses = []
        for start in range(0, n_train, config["batch_size"]):
            take = order[start:start + config["batch_size"]]
            chosen, rejected, bucket = (t[take] for t in split["train"])
            z = model.item_vectors(x)
            margin = model.score(z[chosen], bucket) - model.score(z[rejected], bucket)
            reg = config["l2"] * (model.proj.weight.pow(2).sum() + model.w_time.weight.pow(2).sum())
            loss = F.softplus(-margin).mean() + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(F.softplus(-margin).mean().item())
        val_loss, val_acc = evaluate(model, x, split["val"], config["l2"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
                        "val_accuracy": val_acc})
        if val_loss < best[0] - 1e-5:
            best, since_best = (val_loss, epoch), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= config["patience"]:
                break
    model.load_state_dict(best_state)

    train_users = set(pairs.loc[pairs["split"] == "train", "user_id"])
    pop = popularity_scores(venues, checkins, train_users)
    val_pairs = pairs[pairs["split"] == "val"]
    metrics = {"best_epoch": best[1], "val_loss": best[0],
               "val_accuracy": history[best[1] - 1]["val_accuracy"],
               "baseline_popularity_val_accuracy": pairwise_accuracy(pop, venues, val_pairs),
               "baseline_random_val_accuracy": 0.5, "epochs_run": len(history)}
    if eval_test:
        metrics["test_loss"], metrics["test_accuracy"] = evaluate(model, x, split["test"], config["l2"])
        metrics["baseline_popularity_test_accuracy"] = pairwise_accuracy(
            pop, venues, pairs[pairs["split"] == "test"])

    stamp = datetime.now(timezone.utc)
    digest = hashlib.sha256(json.dumps({"c": config, "p": pairs_sha256}, sort_keys=True).encode()).hexdigest()[:8]
    version = f"v{stamp:%Y%m%d-%H%M%S}-{digest}"
    out = Path(out_root) / version
    out.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        np.save(out / "item_vectors.npy", model.item_vectors(x).numpy())
        np.save(out / "prior_w0.npy", model.w_global.detach().numpy())
        np.save(out / "time_vectors.npy", model.w_time.weight.detach().numpy())
    torch.save(model.state_dict(), out / "model.pt")
    (out / "venue_ids.json").write_text(json.dumps(list(venues["venue_id"])), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "losses.csv", index=False)
    plot_losses(pd.DataFrame(history), out / "loss_curve.png")
    metadata = {
        "version": version, "created_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "git_commit": git_commit(),
        "config": config, "metrics": metrics, "feature_dim": int(x.shape[1]), "categories": categories,
        "inputs": {"pairs_sha256": pairs_sha256, "features_sha256": sha256_file(features_path) if features_path else None,
                   "pairs_by_split": {k: int(v) for k, v in pairs["split"].value_counts().items()}},
        "libraries": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "note": "global prior only; session personalisation is applied on top of w0 by the scoring module",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (Path(out_root) / "latest.txt").write_text(version + "\n", encoding="utf-8")
    return metadata


def plot_losses(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("pairwise logistic loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--features", type=Path, help="npz avec venue_ids et X (ex. embeddings E5)")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true",
                        help="rapporter le split test (une seule fois, hyperparamètres figés)")
    args = parser.parse_args(argv)
    venues, checkins = load_foursquare(args.db)
    config = {k: getattr(args, k) for k in ("dim", "lr", "l2", "epochs", "batch_size", "patience", "seed")}
    metadata = train(venues, checkins, read_table(args.pairs), load_taxonomy(args.taxonomy), args.models_dir,
                     config, sha256_file(args.pairs), args.features, args.eval_test)
    print(json.dumps({"version": metadata["version"], "metrics": metadata["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
