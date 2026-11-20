"""Lecture d'un modèle versionné SANS torch (numpy seulement) : pour l'API et la démo.

Un dossier models/<version>/ contient item_vectors.npy (vecteurs normalisés z(i)),
prior_w0.npy (terme global), time_vectors.npy (une ligne par tranche horaire),
venue_ids.json et metadata.json. models/latest.txt pointe vers la dernière version.
"""
import json
from pathlib import Path

import numpy as np

BUCKETS = ("night", "morning", "afternoon", "evening")


class ModelBundle:
    def __init__(self, path):
        path = Path(path)
        self.path = path
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        self.venue_ids = json.loads((path / "venue_ids.json").read_text(encoding="utf-8"))
        self.item_vectors = np.load(path / "item_vectors.npy")
        self.w0 = np.load(path / "prior_w0.npy")
        self.time_vectors = np.load(path / "time_vectors.npy")
        self.index = {v: i for i, v in enumerate(self.venue_ids)}

    @property
    def version(self):
        return self.metadata["version"]

    def scores(self, time_bucket=None, profile=None):
        """Score de tous les lieux. profile : vecteur w_u de session (défaut : prior w0)."""
        w = self.w0 if profile is None else profile
        if time_bucket is not None:
            w = w + self.time_vectors[BUCKETS.index(time_bucket)]
        return self.item_vectors @ w


def latest_version(models_dir):
    return (Path(models_dir) / (Path(models_dir) / "latest.txt").read_text(encoding="utf-8").strip())


def load_latest(models_dir):
    return ModelBundle(latest_version(models_dir))
