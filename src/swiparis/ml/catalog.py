"""Construction des `Item` depuis les bases SQLite produites par les imports.

`category_map` associe chaque libellé source à une ou plusieurs catégories de la
taxonomie commune. Un libellé absent du mapping arrête le chargement : mieux
vaut une erreur qu'une activité silencieusement sans catégorie.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sqlite3
from typing import Collection, Mapping, Sequence

from swiparis.ml.features import Item, foursquare_local_hour, source_local_hour, time_profile

CategoryMap = Mapping[str, "str | Sequence[str]"]


def _common(label: str, category_map: CategoryMap) -> tuple[str, ...]:
    value = category_map[label]
    return (value,) if isinstance(value, str) else tuple(value)


def _check_mapping(labels: set[str], category_map: CategoryMap, source: str) -> None:
    missing = sorted(labels - category_map.keys())
    if missing:
        raise ValueError(f"{source} labels without common category: {missing}")


def paris_data_items(db_path: Path | str, category_map: CategoryMap) -> list[Item]:
    """Annonces du catalogue Paris Data.

    Position : première adresse de l'annonce. Les séances des annonces multi-lieux
    n'étant pas rattachées à une adresse, le moment de la journée est calculé sur
    toutes les séances de l'annonce.
    """
    with closing(sqlite3.connect(str(db_path))) as db:
        rows = db.execute("SELECT event_id, category FROM event_categories").fetchall()
        _check_mapping({category for _, category in rows}, category_map, "Paris Data")
        categories = defaultdict(set)
        for event_id, category in rows:
            categories[event_id].update(_common(category, category_map))
        hours = defaultdict(list)
        for event_id, start in db.execute("SELECT event_id, source_start FROM event_occurrences"):
            hours[event_id].append(source_local_hour(start))
        locations = {}
        for event_id, latitude, longitude in db.execute(
                "SELECT event_id, latitude, longitude FROM event_locations ORDER BY event_id, location_index"):
            locations.setdefault(event_id, (latitude, longitude))
        items = []
        for event_id, price_type in db.execute("SELECT event_id, price_type FROM events ORDER BY event_id"):
            latitude, longitude = locations.get(event_id, (None, None))
            items.append(Item(event_id, tuple(sorted(categories[event_id])), price_type,
                              latitude, longitude, time_profile(hours[event_id])))
        return items


def foursquare_items(db_path: Path | str, category_map: CategoryMap,
                     users: Collection[str] | None) -> list[Item]:
    """Lieux Foursquare. Le prix est inconnu (absent de Foursquare).

    Le moment de la journée vient uniquement des check-ins de `users`. Passer les
    utilisateurs d'entraînement : aucune visite de validation ou de test n'entre
    alors dans les features. `None` utilise tous les check-ins (démo seulement).
    """
    with closing(sqlite3.connect(str(db_path))) as db:
        venues = db.execute(
            "SELECT venue_id, category, latitude, longitude FROM venues ORDER BY venue_id").fetchall()
        _check_mapping({category for _, category, _, _ in venues}, category_map, "Foursquare")
        allowed = None if users is None else set(users)
        hours = defaultdict(list)
        for user_id, venue_id, utc_time, offset in db.execute(
                "SELECT user_id, venue_id, utc_time, timezone_offset FROM checkins"):
            if allowed is None or user_id in allowed:
                hours[venue_id].append(foursquare_local_hour(utc_time, offset))
        return [Item(venue_id, _common(category, category_map), "unknown", latitude, longitude,
                     time_profile(hours[venue_id]))
                for venue_id, category, latitude, longitude in venues]
