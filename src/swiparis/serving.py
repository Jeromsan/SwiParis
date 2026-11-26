"""Prior versionné et fonction `predict` (rôle MLE-Ops : artefacts du modèle et inférence).

Un modèle enregistré est un dossier `models/<version>/` :
    prior.json      copie exacte du prior appris par `swiparis.ml.prior` (w0 + noms des features)
    w0.npy          les mêmes poids, au format numpy
    metadata.json   version, SHA-256, hyperparamètres, métriques, entrées, commit git, bibliothèques
`models/latest_prior.txt` pointe vers la dernière version (le fichier `latest.txt` reste celui de
`swiparis.registry`, un autre format de modèle).

La version est déterministe : `prior-<date de création du prior>-<8 premiers caractères du SHA-256>`.
L'espace de features est reconstruit depuis le prior lui-même : l'API doit utiliser `model.space`
et non un `FeatureSpace` construit à part, sinon l'ordre des dimensions peut différer.

    python -m swiparis.serving register --prior models/prior_w0_phi.json --models-dir models
    python -m swiparis.serving show --models-dir models

    from swiparis.serving import load_model
    model = load_model("models")                       # dernière version, SHA-256 vérifié
    phi = model.space.encode_many(items)               # items : liste de swiparis.ml.features.Item
    ranked = model.predict(phi, ids, answers=[(chosen_id, rejected_id), ...], k=10)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np

from swiparis.ml.features import FeatureSpace
from swiparis.ml.model import SessionProfile
from swiparis.ml.selection import next_pair as _next_pair
from swiparis.ml.selection import rank as _rank

MODEL_KIND = "bradley_terry_linear_phi"
LATEST = "latest_prior.txt"
REQUIRED = ("model", "created_utc", "feature_names", "grid", "w0")


def sha256_file(path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def space_from_prior(prior: dict) -> FeatureSpace:
    """FeatureSpace identique à celui de l'entraînement, reconstruit depuis les noms de features."""
    missing = [key for key in REQUIRED if key not in prior]
    if missing:
        raise ValueError(f"Prior is missing keys: {missing}")
    if prior["model"] != MODEL_KIND:
        raise ValueError(f"Unsupported prior model {prior['model']!r}")
    names = list(prior["feature_names"])
    space = FeatureSpace(
        [n[4:] for n in names if n.startswith("cat:")],
        grid=tuple(prior["grid"]),
        text_dim=sum(n.startswith("text:") for n in names),
        popularity="pop:log_visits" in names)
    if space.names() != names:
        raise ValueError("Feature names do not match the space rebuilt from the prior")
    w0 = np.asarray(prior["w0"], dtype=float)
    if w0.shape != (space.dim,) or not np.isfinite(w0).all():
        raise ValueError(f"w0 must contain {space.dim} finite values")
    return space


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def register_prior(prior_path, models_dir) -> Path:
    """Copie le prior dans `models/<version>/` avec ses métadonnées. Idempotent."""
    prior_path, models_dir = Path(prior_path), Path(models_dir)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    space = space_from_prior(prior)
    digest = sha256_file(prior_path)
    version = f"prior-{prior['created_utc'][:10].replace('-', '')}-{digest[:8]}"
    target = models_dir / version
    if target.exists():
        if sha256_file(target / "prior.json") != digest:
            raise ValueError(f"{target} exists with different content")
    else:
        target.mkdir(parents=True)
        (target / "prior.json").write_bytes(prior_path.read_bytes().replace(b"\r\n", b"\n"))
        np.save(target / "w0.npy", np.asarray(prior["w0"], dtype=float))
        metadata = {
            "version": version, "kind": MODEL_KIND, "created_utc": prior["created_utc"],
            "prior_sha256": digest, "git_commit": _git_commit(),
            "feature_dim": space.dim, "categories": list(space.categories), "grid": list(space.grid),
            "popularity_feature": space.popularity, "trained_blocks": prior.get("trained_blocks"),
            "lambda": prior.get("lambda"), "pop_scale": prior.get("pop_scale"),
            "metrics": prior.get("metrics"), "inputs": prior.get("inputs"),
            "libraries": {"numpy": np.__version__, "python": platform.python_version()},
            "note": prior.get("note"),
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    (models_dir / LATEST).write_text(version + "\n", encoding="utf-8")
    return target


class ServingModel:
    """Prior chargé et vérifié, avec l'espace de features et les fonctions d'inférence."""

    def __init__(self, path: Path, metadata: dict, prior: dict):
        self.path = Path(path)
        self.metadata = metadata
        self.version = metadata["version"]
        self.space = space_from_prior(prior)
        self.w0 = np.asarray(prior["w0"], dtype=float)

    def new_profile(self, **options) -> SessionProfile:
        """Profil de session neuf, initialisé sur w0 (options : mode, eta, lam, prior_var)."""
        return SessionProfile(self.w0, **options)

    def encode(self, items) -> np.ndarray:
        return self.space.encode_many(list(items))

    def profile_from(self, phi, ids: Sequence[str], answers: Iterable[tuple[str, str]] = (),
                     **options) -> SessionProfile:
        """Rejoue les réponses (identifiant choisi, identifiant rejeté) ; les passages sont absents."""
        phi = self._check(phi, ids)
        index = {item_id: pos for pos, item_id in enumerate(ids)}
        profile = self.new_profile(**options)
        for chosen, rejected in answers:
            if chosen in index and rejected in index:
                profile.update(phi[index[chosen]], phi[index[rejected]])
        return profile

    def predict(self, phi, ids: Sequence[str], answers: Iterable[tuple[str, str]] = (), k: int = 10,
                exclude: Iterable[str] = (), diversity: float = 0.3, profile: SessionProfile | None = None,
                **options) -> list[tuple[str, float]]:
        """Top-k (identifiant, score), après re-ranking pour la diversité.

        phi, ids : activités ÉLIGIBLES (contraintes déjà appliquées). answers : réponses de la session.
        exclude : activités à ne pas recommander (par exemple les sorties explicitement rejetées).
        """
        phi = self._check(phi, ids)
        profile = profile or self.profile_from(phi, ids, answers, **options)
        excluded = set(exclude)
        keep = [i for i, item_id in enumerate(ids) if item_id not in excluded]
        if not keep:
            return []
        return _rank(profile, phi[keep], [ids[i] for i in keep], k=k, diversity=diversity)

    def next_pair(self, phi, ids: Sequence[str], answers: Iterable[tuple[str, str]] = (),
                  shown_pairs: Iterable[Iterable[str]] = (), rng=None, **options):
        """Prochaine paire à comparer (sélection active), ou None."""
        phi = self._check(phi, ids)
        profile = self.profile_from(phi, ids, answers, **options)
        return _next_pair(profile, phi, list(ids), shown_pairs=shown_pairs, rng=rng)

    def _check(self, phi, ids):
        phi = np.asarray(phi, dtype=float)
        if phi.ndim != 2 or phi.shape[1] != self.space.dim:
            raise ValueError(f"phi must have shape (n, {self.space.dim}); encode items with model.space")
        if len(ids) != len(phi):
            raise ValueError("phi and ids must have the same length")
        return phi


def load_model(location="models") -> ServingModel:
    """Charge `models/<version>/` ou, pour un dossier `models/`, la version de `latest_prior.txt`."""
    location = Path(location)
    if not (location / "metadata.json").exists():
        pointer = location / LATEST
        if not pointer.exists():
            raise FileNotFoundError(f"No registered prior in {location} (run: python -m swiparis.serving register)")
        location = location / pointer.read_text(encoding="utf-8").strip()
    metadata = json.loads((location / "metadata.json").read_text(encoding="utf-8"))
    if sha256_file(location / "prior.json") != metadata["prior_sha256"]:
        raise ValueError(f"{location}/prior.json does not match its recorded SHA-256")
    prior = json.loads((location / "prior.json").read_text(encoding="utf-8"))
    return ServingModel(location, metadata, prior)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="enregistrer un prior dans models/<version>/")
    reg.add_argument("--prior", type=Path, default=Path("models/prior_w0_phi.json"))
    reg.add_argument("--models-dir", type=Path, default=Path("models"))
    show = sub.add_parser("show", help="afficher la version chargée")
    show.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    if args.command == "register":
        print(register_prior(args.prior, args.models_dir))
    else:
        model = load_model(args.models_dir)
        print(json.dumps({"version": model.version, "dim": model.space.dim, "categories": len(model.space.categories),
                          "sha256_ok": True, "metrics": model.metadata.get("metrics")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
