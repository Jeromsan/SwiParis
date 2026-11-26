"""Petite base `common.sqlite` de test (schéma réel), sans les données partagées hors Git."""
from contextlib import closing
from importlib.resources import files
from datetime import datetime, timedelta
import json
import sqlite3
from zoneinfo import ZoneInfo

CATEGORIES = ("atelier", "autre", "culture", "loisirs", "nature", "restauration", "spectacle",
              "sport", "vie_nocturne", "visite")
WINDOW = {"as_of_utc": "2026-11-18T11:30:00Z", "window_end_utc": "2026-12-18T11:30:00Z"}

# (identifiant, catégorie, prix, séances UTC) — samedi 28/11, créneau de référence 13:00-16:00 UTC
ACTIVITIES = [
    ("expo", "culture", "free", [("2026-11-28T09:00:00Z", "2026-11-28T17:00:00Z")]),       # ouverte toute la journée
    ("concert", "spectacle", "paid", [("2026-11-28T14:00:00Z", "2026-11-28T15:30:00Z")]),  # dans le créneau
    ("tardif", "spectacle", "paid", [("2026-11-28T15:30:00Z", "2026-11-28T17:30:00Z")]),   # 30 min de recouvrement
    ("court", "atelier", "free", [("2026-11-28T15:00:00Z", "2026-11-28T15:30:00Z")]),      # court, entier
    ("court-bord", "atelier", "free", [("2026-11-28T15:50:00Z", "2026-11-28T16:20:00Z")]), # court, coupé
    ("balade", "visite", "free", [("2026-11-28T13:00:00Z", "2026-11-28T15:00:00Z")]),
    ("match", "sport", "paid", [("2026-11-28T13:30:00Z", "2026-11-28T15:30:00Z")]),
    ("jardin", "nature", "free", [("2026-11-28T08:00:00Z", "2026-11-28T16:00:00Z")]),
    ("jeux", "loisirs", "free", [("2026-11-28T12:00:00Z", "2026-11-28T18:00:00Z")]),
    ("diner", "restauration", "paid", [("2026-11-28T18:00:00Z", "2026-11-28T21:00:00Z")]),
]


def extra_activities(count):
    """Activités supplémentaires le dimanche 29/11 (hors du créneau de référence), pour l'API."""
    rows = []
    for n in range(count):
        start_hour = 13 + n % 3
        rows.append((f"extra{n:02d}", CATEGORIES[n % len(CATEGORIES)], "free" if n % 2 else "paid",
                     [(f"2026-11-29T{start_hour}:00:00Z", f"2026-11-29T{start_hour}:45:00Z")]))
    return rows


def paris_source(utc_text):
    """Heure telle que Paris Data l'écrit : heure locale de Paris, suivie de +02:00 même en hiver."""
    instant = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    return instant.astimezone(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%dT%H:%M:%S") + "+02:00"


def stored_utc(utc_text):
    """Colonne start_utc/end_utc telle que l'import la remplit : heure locale - 2 h (fausse en hiver)."""
    wall = datetime.fromisoformat(paris_source(utc_text)[:19])
    return (wall - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_common(path, extra=0):
    schema = files("swiparis").joinpath("common_schema.sql").read_text(encoding="utf-8")
    with closing(sqlite3.connect(path)) as db:
        db.executescript(schema)
        manifest = {"data_version": "test-catalog", "catalog_manifest": WINDOW}
        db.execute("INSERT INTO metadata VALUES ('manifest', ?)", (json.dumps(manifest),))
        db.executemany("INSERT INTO categories VALUES (?,?)", [(c, c.title()) for c in CATEGORIES])
        for n, (name, category, price, sessions) in enumerate(ACTIVITIES + extra_activities(extra)):
            item_id = f"paris_data:{name}"
            db.execute("INSERT INTO items(item_id,source,source_id,title,description,lead_text,source_url,price_type,"
                       "price_amount_eur,price_detail,cover_credit,date_description,audience,quality_flags_json) "
                       "VALUES (?,'paris_data',?,?,'','',?,?,?,'','','','','[]')",
                       (item_id, name, name.title(), f"https://www.paris.fr/evenements/{name}", price,
                        0.0 if price == "free" else None))
            db.execute("INSERT INTO item_categories VALUES (?,?)", (item_id, category))
            db.execute("INSERT INTO locations(item_id,location_index,arrondissement,latitude,longitude) "
                       "VALUES (?,0,?,?,?)", (item_id, 1 + n % 20, 48.82 + 0.0015 * n, 2.25 + 0.004 * n))
            for start, end in sessions:
                # Comme la vraie source : heure locale de Paris (UTC+1 en novembre) avec un décalage +02:00.
                # Les sessions des tests sont les vrais instants UTC ; les colonnes UTC stockées
                # sont décalées d'une heure, comme dans common.sqlite.
                db.execute("INSERT INTO occurrences VALUES (?,?,?,?,?)",
                           (item_id, stored_utc(start), stored_utc(end), paris_source(start), paris_source(end)))
        db.commit()
    return path
