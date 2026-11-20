"""Vecteur de features φ(i) partagé par les lieux Foursquare et les annonces Paris Data.

Le même encodage sert à l'entraînement hors ligne (Foursquare) et au service
(catalogue Paris Data). Les deux sources sont d'abord ramenées à la taxonomie
commune : on ne cherche pas à apparier les lieux un à un.

Blocs, dans cet ordre :
  cat    catégories communes, multi-hot normalisé (somme 1) ;
  price  free / conditional_free / paid / unknown, one-hot. Un prix inconnu a sa
         propre dimension : il n'est jamais confondu avec la gratuité ;
  time   part des visites ou séances par tranche horaire locale
         (nuit 0-6 h, matin 6-12 h, après-midi 12-18 h, soirée 18-24 h) ;
  zone   cellule d'une grille posée sur le rectangle Paris des imports, one-hot ;
  text   embedding texte optionnel (normalisé L2), absent par défaut ;
  pop    popularité optionnelle : log(1 + visites) / log(1 + maximum), calculée
         sur les visites d'entraînement. Les annonces Paris Data n'ont pas de
         visites : leur popularité est inconnue et vaut 0 dans ce bloc.
Une information manquante donne un bloc nul.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Iterable, Sequence

import numpy as np

PARIS_BBOX = (48.815, 2.224, 48.903, 2.470)  # S, W, N, E : même rectangle que les imports
PRICE_TYPES = ("free", "conditional_free", "paid", "unknown")
TIME_BUCKETS = ("night", "morning", "afternoon", "evening")  # index = heure locale // 6


@dataclass(frozen=True)
class Item:
    """Activité candidate, déjà ramenée à la taxonomie commune."""

    item_id: str
    categories: tuple[str, ...] = ()
    price_type: str = "unknown"
    latitude: float | None = None
    longitude: float | None = None
    time_profile: tuple[float, ...] | None = None
    text_embedding: tuple[float, ...] | None = None
    popularity: float | None = None


def time_bucket(hour: int) -> int:
    if not 0 <= hour <= 23:
        raise ValueError(f"Invalid hour: {hour}")
    return hour // 6


def time_profile(hours: Iterable[int]) -> tuple[float, ...] | None:
    """Part de chaque tranche horaire ; None si aucune heure (moment inconnu)."""
    counts = [0] * len(TIME_BUCKETS)
    for hour in hours:
        counts[time_bucket(hour)] += 1
    total = sum(counts)
    return tuple(count / total for count in counts) if total else None


def foursquare_local_hour(utc_time: str, offset_minutes: int) -> int:
    """Heure locale d'un check-in importé (`utc_time` ISO en Z + décalage en minutes)."""
    instant = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%SZ")
    return (instant + timedelta(minutes=offset_minutes)).hour


def source_local_hour(iso_text: str) -> int:
    """Heure locale d'une séance Paris Data, lue avec le décalage écrit dans la source."""
    return datetime.fromisoformat(iso_text.replace("Z", "+00:00")).hour


class FeatureSpace:
    """Encodeur φ : Item -> vecteur numpy de taille `dim`."""

    def __init__(self, categories: Sequence[str], grid: tuple[int, int] = (3, 3),
                 bbox: tuple[float, float, float, float] = PARIS_BBOX, text_dim: int = 0,
                 popularity: bool = False):
        categories = tuple(categories)
        if not categories or len(set(categories)) != len(categories):
            raise ValueError("Categories must be non-empty and unique")
        rows, cols = grid
        if rows < 1 or cols < 1 or text_dim < 0:
            raise ValueError("Invalid grid or text dimension")
        south, west, north, east = bbox
        if not (south < north and west < east):
            raise ValueError("Invalid bbox")
        self.categories = categories
        self.grid = (rows, cols)
        self.bbox = bbox
        self.text_dim = text_dim
        self.popularity = popularity
        self._category_index = {name: i for i, name in enumerate(categories)}
        sizes = {"cat": len(categories), "price": len(PRICE_TYPES), "time": len(TIME_BUCKETS),
                 "zone": rows * cols, "text": text_dim, "pop": int(popularity)}
        self.blocks: dict[str, slice] = {}
        start = 0
        for name, size in sizes.items():
            self.blocks[name] = slice(start, start + size)
            start += size
        self.dim = start

    def names(self) -> list[str]:
        """Nom de chaque dimension, pour lire un profil w (ex. `cat:musee`)."""
        rows, cols = self.grid
        return ([f"cat:{c}" for c in self.categories]
                + [f"price:{p}" for p in PRICE_TYPES]
                + [f"time:{t}" for t in TIME_BUCKETS]
                + [f"zone:{r}-{c}" for r in range(rows) for c in range(cols)]
                + [f"text:{k}" for k in range(self.text_dim)]
                + (["pop:log_visits"] if self.popularity else []))

    def zone(self, latitude: float | None, longitude: float | None) -> int | None:
        """Indice de cellule (ligne 0 = nord, colonne 0 = ouest), None sans coordonnées."""
        if latitude is None or longitude is None:
            return None
        south, west, north, east = self.bbox
        if not (south <= latitude <= north and west <= longitude <= east):
            raise ValueError(f"Coordinates outside bbox: {latitude}, {longitude}")
        rows, cols = self.grid
        row = min(int((north - latitude) / (north - south) * rows), rows - 1)
        col = min(int((longitude - west) / (east - west) * cols), cols - 1)
        return row * cols + col

    def encode(self, item: Item) -> np.ndarray:
        x = np.zeros(self.dim)
        unknown = sorted(set(item.categories) - self._category_index.keys())
        if unknown:
            raise ValueError(f"{item.item_id}: categories outside the taxonomy: {unknown}")
        categories = set(item.categories)
        for name in categories:
            x[self.blocks["cat"].start + self._category_index[name]] = 1 / len(categories)
        if item.price_type not in PRICE_TYPES:
            raise ValueError(f"{item.item_id}: unknown price type {item.price_type!r}")
        x[self.blocks["price"].start + PRICE_TYPES.index(item.price_type)] = 1
        if item.time_profile is not None:
            profile = np.asarray(item.time_profile, dtype=float)
            if (profile.shape != (len(TIME_BUCKETS),) or (profile < 0).any()
                    or not math.isclose(profile.sum(), 1, abs_tol=1e-6)):
                raise ValueError(f"{item.item_id}: time profile must be {len(TIME_BUCKETS)} shares summing to 1")
            x[self.blocks["time"]] = profile
        zone = self.zone(item.latitude, item.longitude)
        if zone is not None:
            x[self.blocks["zone"].start + zone] = 1
        if self.text_dim and item.text_embedding is not None:
            vector = np.asarray(item.text_embedding, dtype=float)
            if vector.shape != (self.text_dim,):
                raise ValueError(f"{item.item_id}: text embedding must have {self.text_dim} values")
            norm = np.linalg.norm(vector)
            if norm > 0:
                x[self.blocks["text"]] = vector / norm
        if self.popularity and item.popularity is not None:
            if not 0 <= item.popularity <= 1:
                raise ValueError(f"{item.item_id}: popularity must be in [0, 1]")
            x[self.blocks["pop"].start] = item.popularity
        return x

    def encode_many(self, items: Sequence[Item]) -> np.ndarray:
        """Matrice (n, dim), une activité par ligne."""
        if not items:
            return np.zeros((0, self.dim))
        return np.vstack([self.encode(item) for item in items])
