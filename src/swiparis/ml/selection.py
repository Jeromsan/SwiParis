"""Choix de la prochaine paire (fonction d'acquisition) et classement final avec diversité.

Les filtres d'éligibilité (créneau, arrondissement, budget) sont appliqués avant,
côté API : `phi` et `ids` ne contiennent que les activités éligibles.

Acquisition d'une paire (i, j) :
    A(i, j) = H(P(i ≻ j)) + β (1 − cos(φ(i), φ(j)))
H est l'entropie binaire (en bits) de la prédiction. L'entropie seule préfère
deux activités presque identiques (P ≈ 0,5 parce que Δ ≈ 0), qui n'apprennent
rien sur le profil : le terme de distance cosinus corrige ce biais.

Classement : score wᵀφ(i), puis re-ranking MMR (Maximal Marginal Relevance) :
à chaque rang, l'activité qui maximise
    (1 − d) · pertinence(i) − d · max_{j déjà classé} cos(φ(i), φ(j))
avec la pertinence ramenée dans [0, 1]. d = 0 revient à trier par score.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from swiparis.ml.model import SessionProfile


def binary_entropy(p):
    """Entropie en bits : 1 pour p = 0,5 ; 0 pour p = 0 ou 1."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def cosine_similarity_matrix(phi) -> np.ndarray:
    """Similarités cosinus entre lignes ; un vecteur nul a une similarité 0 avec tout."""
    phi = np.asarray(phi, dtype=float)
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    unit = np.divide(phi, norms, out=np.zeros_like(phi), where=norms > 0)
    return unit @ unit.T


def acquisition_matrix(profile: SessionProfile, phi, beta: float = 0.5) -> np.ndarray:
    """A[i, j] pour i < j ; -inf ailleurs (une paire non ordonnée est comptée une fois)."""
    phi = np.asarray(phi, dtype=float)
    n = len(phi)
    i, j = np.triu_indices(n, k=1)
    values = np.full((n, n), -np.inf)
    if n > 1:
        uncertainty = binary_entropy(profile.prob(phi[i], phi[j]))
        values[i, j] = uncertainty + beta * (1 - cosine_similarity_matrix(phi)[i, j])
    return values


def _pool(scores: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Moitié des activités les mieux classées + tirage uniforme parmi les autres."""
    n = len(scores)
    if n <= size:
        return np.arange(n)
    order = np.argsort(-scores, kind="stable")
    top = order[: size // 2]
    explore = rng.choice(order[size // 2:], size=size - len(top), replace=False)
    return np.sort(np.concatenate([top, explore]))


def _best_pair(profile, phi, ids, candidates, shown, beta):
    values = acquisition_matrix(profile, phi[candidates], beta)
    for flat in np.argsort(-values, axis=None, kind="stable"):
        a, b = divmod(int(flat), len(candidates))
        if not np.isfinite(values[a, b]):
            return None
        ia, ib = candidates[a], candidates[b]
        if frozenset((ids[ia], ids[ib])) in shown or np.allclose(phi[ia], phi[ib]):
            continue
        return ia, ib
    return None


def next_pair(profile: SessionProfile, phi, ids: Sequence[str],
              shown_pairs: Iterable[Iterable[str]] = (), beta: float = 0.5,
              pool_size: int = 40, rng=None) -> tuple[str, str] | None:
    """Prochaine paire à montrer, ou None s'il ne reste aucune paire informative.

    La recherche se fait dans un pool de `pool_size` activités (coût en pool_size²).
    Sont exclues les paires déjà montrées et celles dont les deux vecteurs sont
    identiques (Δ = 0 : le choix n'apprend rien). L'ordre gauche/droite est tiré
    au hasard pour ne pas créer de biais de position. `rng` : graine ou Generator.
    """
    phi = np.asarray(phi, dtype=float)
    ids = list(ids)
    if len(ids) != len(phi):
        raise ValueError("phi and ids must have the same length")
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")
    rng = np.random.default_rng(rng)
    shown = {frozenset(pair) for pair in shown_pairs}
    pool = _pool(profile.scores(phi), pool_size, rng)
    found = _best_pair(profile, phi, ids, pool, shown, beta)
    if found is None and len(pool) < len(ids):
        found = _best_pair(profile, phi, ids, np.arange(len(ids)), shown, beta)
    if found is None:
        return None
    first, second = found if rng.random() < 0.5 else found[::-1]
    return ids[first], ids[second]


def mmr_rerank(scores, phi, k: int, diversity: float = 0.3) -> list[int]:
    """Indices des k activités retenues, dans l'ordre d'affichage."""
    if not 0 <= diversity <= 1:
        raise ValueError("diversity must be in [0, 1]")
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = min(k, n)
    if k <= 0:
        return []
    span = scores.max() - scores.min()
    relevance = (scores - scores.min()) / span if span > 0 else np.zeros(n)
    similarity = cosine_similarity_matrix(phi)
    chosen: list[int] = []
    max_similarity = np.zeros(n)
    available = np.ones(n, dtype=bool)
    for _ in range(k):
        value = (1 - diversity) * relevance - diversity * max_similarity
        value[~available] = -np.inf
        best = int(np.argmax(value))  # premier maximum : égalités départagées par l'ordre d'entrée
        chosen.append(best)
        available[best] = False
        max_similarity = np.maximum(max_similarity, similarity[best])
    return chosen


def rank(profile: SessionProfile, phi, ids: Sequence[str], k: int = 10,
         diversity: float = 0.3) -> list[tuple[str, float]]:
    """Top-k (identifiant, score du profil), après re-ranking pour la diversité."""
    phi = np.asarray(phi, dtype=float)
    ids = list(ids)
    if len(ids) != len(phi):
        raise ValueError("phi and ids must have the same length")
    scores = profile.scores(phi)
    return [(ids[i], float(scores[i])) for i in mmr_rerank(scores, phi, k, diversity)]


def intra_list_diversity(phi) -> float:
    """Distance cosinus moyenne entre les activités d'une liste (métrique de l'innovation)."""
    phi = np.asarray(phi, dtype=float)
    n = len(phi)
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    return float(np.mean(1 - cosine_similarity_matrix(phi)[i, j]))
