"""Activités encodées depuis la base commune `data/processed/common.sqlite`.

La base préparée par Jeromsan contient déjà les catégories communes (table
`categories`) et met les lieux Foursquare et les annonces Paris Data dans une
seule table `items`. Les identifiants sont préfixés par leur source
(`foursquare:...`, `paris_data:...`), comme les utilisateurs (`foursquare:...`).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Collection, Sequence

import numpy as np

from swiparis.ml.features import FeatureSpace, Item, foursquare_local_hour, source_local_hour, time_profile

SOURCES = ("foursquare", "paris_data")


def _connect(db_path) -> sqlite3.Connection:
    """Connexion en lecture seule : le code ML ne modifie jamais la base commune."""
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def common_categories(db_path) -> tuple[str, ...]:
    """Codes des catégories communes, triés : l'ordre des dimensions de φ ne dépend
    pas de l'ordre d'insertion dans la base."""
    with closing(_connect(db_path)) as db:
        return tuple(sorted(row[0] for row in db.execute("SELECT category FROM categories")))


def visit_counts(db_path, users: Collection[str]) -> Counter:
    """Nombre de visites par lieu, sur les visites de `users` seulement."""
    allowed = set(users)
    with closing(_connect(db_path)) as db:
        return Counter(item_id for user_id, item_id in db.execute("SELECT user_id, item_id FROM visits")
                       if user_id in allowed)


def load_histories(db_path, users: Collection[str]) -> dict[str, list[tuple[str, str]]]:
    """{user_id: [(utc_time, item_id), ...]} triées dans le temps, pour `users` seulement."""
    allowed, histories = set(users), defaultdict(list)
    with closing(_connect(db_path)) as db:
        for user_id, item_id, utc_time in db.execute(
                "SELECT user_id, item_id, utc_time FROM visits ORDER BY user_id, utc_time, item_id"):
            if user_id in allowed:
                histories[user_id].append((utc_time, item_id))
    return histories


def popularity_value(count: int, maximum: int) -> float:
    """log(1 + visites) / log(1 + maximum), dans [0, 1]."""
    return math.log1p(count) / math.log1p(maximum) if maximum > 0 else 0.0


def common_items(db_path, *, users: Collection[str] | None,
                 sources: Sequence[str] = SOURCES, counts: Counter | None = None) -> list[Item]:
    """Activités de la base commune, triées par identifiant.

    Position : première adresse. Moment de la journée : séances pour Paris Data,
    visites pour Foursquare. Pour Foursquare, seules les visites de `users`
    comptent : passer les utilisateurs d'entraînement pour qu'aucune visite de
    validation ou de test n'entre dans les features. `users=None` prend toutes
    les visites (démo seulement). `counts` (voir `visit_counts`) remplit la
    popularité des lieux Foursquare ; elle reste inconnue sinon.
    """
    sources = tuple(sources)
    if not sources or set(sources) - set(SOURCES):
        raise ValueError(f"sources must be taken from {SOURCES}")
    with closing(_connect(db_path)) as db:
        marks = ",".join("?" * len(sources))
        rows = db.execute(f"SELECT item_id, price_type FROM items WHERE source IN ({marks}) ORDER BY item_id",
                          sources).fetchall()
        categories = defaultdict(list)
        for item_id, category in db.execute("SELECT item_id, category FROM item_categories ORDER BY item_id, category"):
            categories[item_id].append(category)
        locations = {}
        for item_id, latitude, longitude in db.execute(
                "SELECT item_id, latitude, longitude FROM locations ORDER BY item_id, location_index"):
            locations.setdefault(item_id, (latitude, longitude))
        hours = defaultdict(list)
        if "paris_data" in sources:
            for item_id, start in db.execute("SELECT item_id, source_start FROM occurrences"):
                hours[item_id].append(source_local_hour(start))
        if "foursquare" in sources:
            allowed = None if users is None else set(users)
            for user_id, item_id, utc_time, offset in db.execute(
                    "SELECT user_id, item_id, utc_time, timezone_offset FROM visits"):
                if allowed is None or user_id in allowed:
                    hours[item_id].append(foursquare_local_hour(utc_time, offset))
    maximum = max(counts.values(), default=0) if counts is not None else 0
    return [Item(item_id, tuple(categories[item_id]), price_type, *locations.get(item_id, (None, None)),
                 time_profile(hours[item_id]),
                 popularity=(popularity_value(counts.get(item_id, 0), maximum)
                             if counts is not None and item_id.startswith("foursquare:") else None))
            for item_id, price_type in rows]


@dataclass
class EncodedItems:
    """φ précalculé pour un ensemble d'activités, avec accès par identifiant."""

    space: FeatureSpace
    ids: list[str]
    phi: np.ndarray
    counts: Counter | None = None  # visites par lieu ayant servi au bloc popularité
    _row: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self._row = {item_id: i for i, item_id in enumerate(self.ids)}
        if len(self._row) != len(self.ids) or self.phi.shape != (len(self.ids), self.space.dim):
            raise ValueError("ids must be unique and match the rows of phi")

    def rows(self, item_ids: Sequence[str]) -> np.ndarray:
        """Lignes de φ dans l'ordre donné, par exemple les activités éligibles du catalogue."""
        missing = [item_id for item_id in item_ids if item_id not in self._row]
        if missing:
            raise KeyError(f"Unknown activities: {missing[:5]}")
        return self.phi[[self._row[item_id] for item_id in item_ids]]

    def vector(self, item_id: str) -> np.ndarray:
        return self.rows([item_id])[0]


def encode_common(db_path, *, users: Collection[str] | None, sources: Sequence[str] = SOURCES,
                  grid: tuple[int, int] = (3, 3), popularity_users: Collection[str] | None = None) -> EncodedItems:
    """Charge et encode les activités ; à faire une fois au démarrage du service.

    `popularity_users` ajoute le bloc popularité, calculé sur leurs visites
    (les utilisateurs d'entraînement). Il est nécessaire avec le prior w₀ appris.
    """
    counts = visit_counts(db_path, popularity_users) if popularity_users is not None else None
    space = FeatureSpace(common_categories(db_path), grid=grid, popularity=counts is not None)
    items = common_items(db_path, users=users, sources=sources, counts=counts)
    return EncodedItems(space, [item.item_id for item in items], space.encode_many(items), counts)
