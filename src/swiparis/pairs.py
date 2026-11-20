"""Génère des paires de préférence synthétiques à partir des check-ins Foursquare.

Le lieu visité est le « choisi ». Le « rejeté » est un lieu que l'utilisateur n'a
JAMAIS visité dans la base, tiré selon une stratégie (géographie, catégorie, moment,
popularité) avec un échantillonnage pondéré par popularité**alpha.
Ce sont des négatifs FAIBLES : « non visité » ne veut pas dire « pas aimé ».
La popularité n'est calculée que sur les utilisateurs d'entraînement.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

TIME_BUCKETS = ("night", "morning", "afternoon", "evening")  # heure locale : <6, <12, <18, reste
STRATEGIES = ("geo", "cat", "time", "pop")
DEFAULT_WEIGHTS = {"geo": 0.30, "cat": 0.25, "time": 0.20, "pop": 0.25}
SPLITS = ("train", "val", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def time_bucket(utc_time, offset_minutes):
    """Tranche horaire LOCALE (UTC + décalage en minutes) : 0 nuit, 1 matin, 2 après-midi, 3 soir."""
    utc = pd.to_datetime(utc_time, format="%Y-%m-%dT%H:%M:%SZ", utc=True)
    hour = (utc + pd.to_timedelta(offset_minutes, unit="m")).dt.hour.to_numpy()
    return np.select([hour < 6, hour < 12, hour < 18], [0, 1, 2], default=3).astype(np.int8)


def load_foursquare(db_path):
    """Lit les lieux et visites, en lecture seule. Deux schémas sont reconnus :
    la base commune de l'équipe (items, item_categories, locations, visits ; catégories déjà
    communes, ids « foursquare:<id> ») ou le SQLite brut de import_foursquare (venues, checkins)."""
    db = sqlite3.connect(str(db_path))
    try:
        db.execute("PRAGMA query_only=ON")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        if {"items", "item_categories", "locations", "visits"} <= tables:
            venues = pd.read_sql_query(
                "SELECT i.item_id AS venue_id, c.category AS category, l.latitude AS latitude, "
                "l.longitude AS longitude FROM items i "
                "JOIN item_categories c ON c.item_id = i.item_id "
                "JOIN locations l ON l.item_id = i.item_id AND l.location_index = "
                "(SELECT MIN(location_index) FROM locations WHERE item_id = i.item_id) "
                "WHERE i.source = 'foursquare' ORDER BY i.item_id", db)
            checkins = pd.read_sql_query(
                "SELECT user_id, item_id AS venue_id, utc_time, timezone_offset FROM visits "
                "ORDER BY user_id, utc_time, item_id", db)
        else:
            venues = pd.read_sql_query(
                "SELECT venue_id, category, latitude, longitude FROM venues ORDER BY venue_id", db)
            # Tri explicite, comme le demande docs/foursquare.md : une table SQL n'a pas d'ordre implicite.
            checkins = pd.read_sql_query(
                "SELECT user_id, venue_id, utc_time, timezone_offset FROM checkins "
                "ORDER BY user_id, utc_time, venue_id", db)
    finally:
        db.close()
    return venues, checkins


def load_taxonomy(path):
    """JSON {catégorie Foursquare: catégorie commune}, à plat ou sous la clé "mapping"."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("mapping", data)


def drop_common_categories(venues, checkins, taxonomy, drop):
    """Retire les lieux (et leurs visites) dont la catégorie commune est dans `drop`."""
    if not drop:
        return venues, checkins
    common = venues["category"].map(lambda c: taxonomy.get(c, c))
    kept = venues[~common.isin(set(drop))].reset_index(drop=True)
    return kept, checkins[checkins["venue_id"].isin(set(kept["venue_id"]))].reset_index(drop=True)


def assign_splits(users, seed, fractions=(0.6, 0.2, 0.2)):
    """Découpage PAR UTILISATEUR : tout l'historique d'un utilisateur reste dans un seul split."""
    users = sorted(set(users))
    order = np.random.default_rng([seed, 1]).permutation(len(users))
    n_train = int(round(fractions[0] * len(users)))
    n_val = int(round(fractions[1] * len(users)))
    labels = np.empty(len(users), dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:]] = "test"
    return dict(zip(users, labels))


def load_splits(path, users):
    """Découpage externe (ex. celui de l'évaluation) : CSV user_id,split."""
    table = pd.read_csv(path, dtype=str)
    if not {"user_id", "split"} <= set(table.columns):
        raise ValueError("splits CSV needs columns user_id,split")
    if not set(table["split"]) <= set(SPLITS):
        raise ValueError(f"split values must be in {SPLITS}")
    mapping = dict(zip(table["user_id"], table["split"]))
    missing = sorted(set(users) - set(mapping))
    if missing:
        raise ValueError(f"{len(missing)} users absent from splits file, e.g. {missing[:3]}")
    return mapping


def nearest_neighbors(lat, lon, k):
    """Indices des k lieux les plus proches de chaque lieu (haversine), soi-même exclu."""
    count = len(lat)
    k = min(k, count - 1)
    la, lo = np.radians(lat), np.radians(lon)
    out = np.empty((count, k), dtype=np.int32)
    for start in range(0, count, 512):
        stop = min(start + 512, count)
        dlat = la[start:stop, None] - la[None, :]
        dlon = lo[start:stop, None] - lo[None, :]
        a = np.sin(dlat / 2) ** 2 + np.cos(la[start:stop, None]) * np.cos(la[None, :]) * np.sin(dlon / 2) ** 2
        a[np.arange(stop - start), np.arange(start, stop)] = np.inf
        out[start:stop] = np.argsort(a, axis=1, kind="stable")[:, :k]
    return out


class WeightedPool:
    """Tirage pondéré par rejet possible : fonction de répartition précalculée."""

    def __init__(self, indices, weights):
        self.indices = np.asarray(indices, dtype=np.int64)
        cdf = np.cumsum(np.asarray(weights, dtype=np.float64))
        self.cdf = cdf / cdf[-1]

    def draw(self, rng):
        position = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        return int(self.indices[min(position, len(self.indices) - 1)])


def build_pairs(venues, checkins, split_of, taxonomy=None, seed=42, neg_per_visit=1,
                weights=None, alpha=0.75, k_geo=50, max_tries=30):
    weights = weights or DEFAULT_WEIGHTS
    taxonomy = taxonomy or {}
    venues = venues.reset_index(drop=True)
    count = len(venues)
    index_of = {v: i for i, v in enumerate(venues["venue_id"])}
    category = venues["category"].map(lambda c: taxonomy.get(c, c)).to_numpy()
    unmapped = sorted(set(venues["category"]) - set(taxonomy)) if taxonomy else []

    check = checkins.copy()
    check["v"] = check["venue_id"].map(index_of)
    if check["v"].isna().any():
        raise ValueError("checkins reference unknown venues")
    check["v"] = check["v"].astype(int)
    check["bucket"] = time_bucket(check["utc_time"], check["timezone_offset"])
    check["split"] = check["user_id"].map(split_of)

    # Popularité : utilisateurs d'entraînement uniquement (pas de fuite vers val/test).
    train = check[check["split"] == "train"]
    pop = np.bincount(train["v"], minlength=count).astype(np.float64)
    pop_w = (pop + 1.0) ** alpha
    pop_bucket = np.stack([
        np.bincount(train.loc[train["bucket"] == b, "v"], minlength=count) for b in range(4)
    ]).astype(np.float64)
    time_w = (pop_bucket + 1.0) ** alpha

    neighbors = nearest_neighbors(venues["latitude"].to_numpy(), venues["longitude"].to_numpy(), k_geo)
    all_idx = np.arange(count)
    pool_pop = WeightedPool(all_idx, pop_w)
    pool_time = [WeightedPool(all_idx, time_w[b]) for b in range(4)]
    pool_cat = {c: WeightedPool(np.flatnonzero(category == c), pop_w[category == c])
                for c in np.unique(category)}

    rng = np.random.default_rng([seed, 2])
    names = list(STRATEGIES)
    probs = np.array([weights[s] for s in names], dtype=np.float64)
    probs /= probs.sum()
    visited = np.zeros(count, dtype=bool)
    rows, skipped = [], 0

    def try_rejection(pool, v):
        for _ in range(max_tries):
            r = pool.draw(rng)
            if r != v and not visited[r]:
                return r
        return None

    def sample(strategy, v, bucket):
        if strategy == "geo":
            cands = neighbors[v][~visited[neighbors[v]]]
            if len(cands) == 0:
                return None
            w = pop_w[cands]
            return int(rng.choice(cands, p=w / w.sum()))
        if strategy == "cat":
            return try_rejection(pool_cat[category[v]], v)
        if strategy == "time":
            return try_rejection(pool_time[bucket], v)
        return try_rejection(pool_pop, v)

    for user_id, group in check.groupby("user_id", sort=True):
        user_venues = group["v"].to_numpy()
        visited[user_venues] = True
        n_visits = len(group)
        for rank, (v, bucket, stamp) in enumerate(zip(user_venues, group["bucket"], group["utc_time"])):
            chosen_negatives = set()
            for _ in range(neg_per_visit):
                first = names[int(rng.choice(len(names), p=probs))]
                result = None
                for strategy in [first] + [s for s in names if s != first]:
                    r = sample(strategy, v, int(bucket))
                    if r is not None and r not in chosen_negatives:
                        result = (r, strategy)
                        break
                if result is None:
                    skipped += 1
                    continue
                chosen_negatives.add(result[0])
                rows.append((user_id, split_of[user_id], rank, n_visits, stamp, TIME_BUCKETS[int(bucket)],
                             venues.at[v, "venue_id"], venues.at[result[0], "venue_id"],
                             "weak_" + result[1], category[v], category[result[0]]))
        visited[user_venues] = False

    pairs = pd.DataFrame(rows, columns=[
        "user_id", "split", "visit_rank", "n_visits", "timestamp", "time_bucket",
        "chosen_venue_id", "rejected_venue_id", "negative_type", "chosen_category", "rejected_category"])
    pairs.insert(0, "pair_id", [f"p{i:07d}" for i in range(len(pairs))])
    info = {"skipped_visits": int(skipped), "unmapped_categories": unmapped}
    return pairs, info


def write_table(frame, path):
    path = Path(path)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"user_id": str,
            "chosen_venue_id": str, "rejected_venue_id": str})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="SQLite Foursquare (venues, checkins)")
    parser.add_argument("--output", type=Path, required=True, help="pairs.csv ou pairs.parquet")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, help="JSON catégorie Foursquare -> catégorie commune")
    parser.add_argument("--splits", type=Path, help="CSV user_id,split (sinon 60/20/20 par utilisateur)")
    parser.add_argument("--drop-categories", default="",
                        help="catégories communes à exclure, séparées par des virgules (ex. restauration,vie_nocturne) ; "
                             "avec la base commune, les catégories sont déjà communes (pas de --taxonomy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neg-per-visit", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--force", action="store_true", help="écraser les sorties existantes")
    args = parser.parse_args(argv)
    for target in (args.output, args.manifest):
        if target.exists() and not args.force:
            parser.error(f"{target} existe déjà (utiliser --force)")
    venues, checkins = load_foursquare(args.db)
    taxonomy = load_taxonomy(args.taxonomy)
    dropped = [c for c in args.drop_categories.split(",") if c]
    visits_before = len(checkins)
    venues, checkins = drop_common_categories(venues, checkins, taxonomy, dropped)
    users = checkins["user_id"].unique()
    split_of = load_splits(args.splits, users) if args.splits else assign_splits(users, args.seed)
    pairs, info = build_pairs(venues, checkins, split_of, taxonomy, args.seed, args.neg_per_visit, alpha=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    write_table(pairs, args.output)
    manifest = {
        "script": "swiparis.pairs", "seed": args.seed, "alpha": args.alpha, "neg_per_visit": args.neg_per_visit,
        "strategy_weights": DEFAULT_WEIGHTS,
        "negatives": "weak: never visited by the user; popularity from train users only",
        "inputs": {"db": str(args.db), "db_sha256": sha256_file(args.db),
                   "taxonomy_sha256": sha256_file(args.taxonomy) if args.taxonomy else None,
                   "splits": "external:" + str(args.splits) if args.splits else "internal 60/20/20 by user"},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "dropped_common_categories": dropped, "visits_before_drop": int(visits_before),
        "venues": int(len(venues)), "users": int(len(users)), "visits": int(len(checkins)),
        "pairs": int(len(pairs)),
        "pairs_by_split": pairs["split"].value_counts().to_dict(),
        "users_by_split": pd.Series(split_of).value_counts().to_dict(),
        "pairs_by_negative_type": pairs["negative_type"].value_counts().to_dict(),
        "same_category_share": float((pairs["chosen_category"] == pairs["rejected_category"]).mean()) if len(pairs) else None,
        **info,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("pairs", "pairs_by_split", "pairs_by_negative_type",
                                               "same_category_share", "skipped_visits")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
