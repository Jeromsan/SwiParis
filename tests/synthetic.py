"""Base Foursquare synthétique au même schéma que import_foursquare (venues, checkins)."""
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np

CATEGORIES = ["Museum", "Art Gallery", "Bar", "Park", "Theater", "Gym"]
# Popularité inégale des catégories : sans signal commun à tous, un modèle global ne peut rien apprendre.
CAT_P = [0.30, 0.15, 0.25, 0.20, 0.08, 0.02]


def make_db(path, users=120, venues_per_cat=15, visits=25, seed=0):
    rng = np.random.default_rng(seed)
    db = sqlite3.connect(str(path))
    db.executescript("""
        CREATE TABLE venues(venue_id TEXT PRIMARY KEY, category TEXT NOT NULL,
            latitude REAL NOT NULL, longitude REAL NOT NULL, country TEXT NOT NULL);
        CREATE TABLE checkins(user_id TEXT NOT NULL, venue_id TEXT NOT NULL REFERENCES venues(venue_id),
            utc_time TEXT NOT NULL, timezone_offset INTEGER NOT NULL,
            PRIMARY KEY(user_id,utc_time,venue_id)) WITHOUT ROWID;
    """)
    venue_rows, by_cat = [], {c: [] for c in CATEGORIES}
    for c in CATEGORIES:
        for i in range(venues_per_cat):
            vid = f"{c[:3].lower()}{i:03d}"
            venue_rows.append((vid, c, 48.82 + rng.random() * 0.08, 2.23 + rng.random() * 0.23, "FR"))
            by_cat[c].append(vid)
    db.executemany("INSERT INTO venues VALUES (?,?,?,?,?)", venue_rows)
    start = datetime(2012, 5, 1, tzinfo=timezone.utc)
    rows = []
    for u in range(users):
        favorites = rng.choice(CATEGORIES, size=2, replace=False, p=CAT_P)  # goûts marqués
        t = start
        for _ in range(visits):
            c = rng.choice(favorites) if rng.random() < 0.8 else rng.choice(CATEGORIES, p=CAT_P)
            t = t + timedelta(hours=float(rng.integers(5, 90)))
            rows.append((f"u{u:04d}", str(rng.choice(by_cat[c])), t.strftime("%Y-%m-%dT%H:%M:%SZ"), 60))
    db.executemany("INSERT OR IGNORE INTO checkins VALUES (?,?,?,?)", rows)
    db.commit()
    db.close()


def make_common_db(raw_path, common_path):
    """Convertit la base synthétique brute en schéma commun minimal (items, item_categories, locations, visits)."""
    raw = sqlite3.connect(str(raw_path))
    db = sqlite3.connect(str(common_path))
    db.executescript("""
        CREATE TABLE items(item_id TEXT PRIMARY KEY, source TEXT NOT NULL);
        CREATE TABLE item_categories(item_id TEXT NOT NULL, category TEXT NOT NULL, PRIMARY KEY(item_id, category));
        CREATE TABLE locations(item_id TEXT NOT NULL, location_index INTEGER NOT NULL, latitude REAL NOT NULL,
            longitude REAL NOT NULL, PRIMARY KEY(item_id, location_index));
        CREATE TABLE visits(user_id TEXT NOT NULL, source_user_id TEXT NOT NULL, item_id TEXT NOT NULL,
            utc_time TEXT NOT NULL, timezone_offset INTEGER NOT NULL, PRIMARY KEY(user_id, utc_time, item_id));
    """)
    for vid, cat, lat, lon, _ in raw.execute("SELECT * FROM venues"):
        db.execute("INSERT INTO items VALUES (?, 'foursquare')", ("foursquare:" + vid,))
        db.execute("INSERT INTO item_categories VALUES (?, ?)", ("foursquare:" + vid, cat))
        db.execute("INSERT INTO locations VALUES (?, 1, ?, ?)", ("foursquare:" + vid, lat, lon))
    for uid, vid, t, off in raw.execute("SELECT * FROM checkins"):
        db.execute("INSERT INTO visits VALUES (?, ?, ?, ?, ?)", ("foursquare:" + uid, uid, "foursquare:" + vid, t, off))
    db.commit()
    db.close()
    raw.close()
