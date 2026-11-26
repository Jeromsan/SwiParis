"""Politique de séance de l'application, partagée par l'API et par la simulation hors ligne.

Une seule implémentation pour les deux, pour que l'évaluation mesure exactement ce que
fait l'application :
- profil : prior global w₀ (ou mélange de personas), mis à jour après chaque choix ;
- paire suivante : `next_pair` dans un pool de 80 activités, avec un score de préférence
  qui suit les sorties choisies et rejetées (`preference_scores`) ; toutes les quatre
  paires, deux sorties déjà choisies sont comparées entre elles ;
- les sorties rejetées ne sont plus proposées ; les sorties choisies peuvent revenir ;
- classement : score du profil + préférence, re-ranking MMR (diversité 0,3), égalités
  départagées au hasard avec la graine de la séance.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable, Sequence

import numpy as np

from swiparis.ml.model import SessionProfile
from swiparis.ml.selection import mmr_rerank, next_pair

MAX_ANSWERS = 10
MAX_OFFERS = 15
# Pas η de la séance. L'évaluation hors ligne (Foursquare) ne départage pas les valeurs
# dans les conditions de l'application : sans signal de popularité (les annonces Paris
# Data n'en ont pas), η = 0,05, 0,2 et 0,5 donnent le même NDCG@10 (docs/modele.md,
# « Réglage de la séance »). On garde 0,5, qui personnalise le plus ; à trancher avec
# les données du pilote. Modifiable sans toucher au code : SWIPARIS_SESSION_ETA.
DEFAULT_ETA = 0.5
LAMBDA = 0.05
PAIR_POOL = 80
RECOMMENDATIONS = 3
DIVERSITY = 0.3
PREFERENCE_WEIGHT = 1.0
PAIR_PREFERENCE_WEIGHT = 0.75


def session_eta() -> float:
    """η lu dans SWIPARIS_SESSION_ETA (défaut DEFAULT_ETA)."""
    value = float(os.environ.get("SWIPARIS_SESSION_ETA", DEFAULT_ETA))
    if not 0 < value <= 2:
        raise ValueError("SWIPARIS_SESSION_ETA must be in (0, 2]")
    return value


def eta_arms() -> tuple[float, ...] | None:
    """Comparaison de réglages pendant le pilote : SWIPARIS_SESSION_ETA_AB="0.2,0.5".

    Chaque séance reçoit une des valeurs, tirée une fois pour toutes à partir de son
    identifiant (`eta_for_session`). Sans cette variable, toutes les séances ont le même η.
    """
    raw = os.environ.get("SWIPARIS_SESSION_ETA_AB", "").strip()
    if not raw:
        return None
    values = tuple(float(v) for v in raw.split(","))
    if len(values) < 2 or len(set(values)) != len(values) or any(not 0 < v <= 2 for v in values):
        raise ValueError("SWIPARIS_SESSION_ETA_AB needs at least two distinct values in (0, 2]")
    return values


def eta_for_session(session_id: str, arms: tuple[float, ...] | None, default: float) -> float:
    return arms[seed(f"eta:{session_id}") % len(arms)] if arms else default


def eta_version(arms: tuple[float, ...] | None, default: float) -> str:
    """Partie de la version du modèle qui décrit le réglage (gardée avec chaque séance)."""
    return "etaAB" + "_".join(f"{v:g}" for v in arms) if arms else f"eta{default:g}"


def eta_from_version(model_version: str, session_id: str) -> float | None:
    """η d'une séance du journal, retrouvé à partir de sa version enregistrée."""
    for token in model_version.split("+"):
        if token.startswith("etaAB"):
            return eta_for_session(session_id, tuple(float(v) for v in token[5:].split("_")), 0.0)
        if token.startswith("eta"):
            return float(token[3:])
    return None


def seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def replay(profile: SessionProfile, comparisons: Iterable[tuple[str, str, str]], index: dict, phi: np.ndarray):
    """Rejoue les réponses (gauche, droite, choix) ; renvoie paires montrées, choisies, rejetées, historique."""
    shown, selected, rejected, history = [], set(), set(), []
    for left, right, choice in comparisons:
        shown.append((left, right))
        if choice not in ("left", "right"):
            continue
        chosen, other = (left, right) if choice == "left" else (right, left)
        selected.add(chosen)
        rejected.add(other)
        history.append((chosen, other))
        if chosen in index and other in index:
            profile.update(phi[index[chosen]], phi[index[other]])
    return shown, selected, rejected, history


def preference_scores(ids: Sequence[str], phi: np.ndarray, history: list[tuple[str, str]],
                      space, history_ids: Sequence[str], history_phi: np.ndarray) -> np.ndarray:
    """Préférence tirée des choix récents : catégories aimées ou rejetées, ressemblance aux sorties aimées.

    Chaque choix plus ancien d'un cran pèse 0,78 fois moins ; une sortie rejetée retire
    0,8 fois le poids de sa catégorie. Règle fixée à la main (pas apprise).
    """
    if not history or not len(ids):
        return np.zeros(len(ids), dtype=float)
    index = {item_id: pos for pos, item_id in enumerate(history_ids)}
    cat = space.blocks["cat"]
    cat_evidence = np.zeros(cat.stop - cat.start, dtype=float)
    total_weight = 0.0
    liked_rows, liked_weights = [], []
    for age, (chosen, rejected) in enumerate(reversed(history)):
        weight = 0.78 ** age
        total_weight += weight
        if chosen in index:
            liked_rows.append(history_phi[index[chosen]])
            liked_weights.append(weight)
            cat_evidence += weight * history_phi[index[chosen], cat]
        if rejected in index:
            cat_evidence -= 0.8 * weight * history_phi[index[rejected], cat]
    category_fit = phi[:, cat] @ cat_evidence / max(total_weight, 1e-12)
    if not liked_rows:
        return category_fit
    candidate_norms = np.linalg.norm(phi, axis=1, keepdims=True)
    candidate_unit = np.divide(phi, candidate_norms, out=np.zeros_like(phi), where=candidate_norms > 0)
    liked = np.asarray(liked_rows, dtype=float)
    liked_norms = np.linalg.norm(liked, axis=1, keepdims=True)
    liked_unit = np.divide(liked, liked_norms, out=np.zeros_like(liked), where=liked_norms > 0)
    similarities = np.clip(candidate_unit @ liked_unit.T, 0, 1)
    recency = np.asarray(liked_weights, dtype=float)
    return category_fit + np.max(similarities * recency[None, :], axis=1)


def choose_pair(profile, ids: Sequence[str], phi: np.ndarray, shown, offered: int,
                history, rejected, space, pair_seed: int):
    """Paire suivante, comme l'API : sans sortie rejetée, avec préférence et ancres."""
    pair_indices = [i for i, item_id in enumerate(ids) if item_id not in rejected]
    pair_ids = [ids[i] for i in pair_indices]
    pair_phi = phi[pair_indices]
    preferences = preference_scores(pair_ids, pair_phi, history, space, ids, phi)
    liked_ids = list(dict.fromkeys(chosen for chosen, _ in history if chosen in set(pair_ids)))
    chosen_pair = None
    # Toutes les quatre paires, deux sorties déjà choisies sont comparées entre elles.
    if offered >= 3 and offered % 4 == 3 and len(liked_ids) >= 2:
        liked_set = set(liked_ids)
        favorite = [i for i, item_id in enumerate(pair_ids) if item_id in liked_set]
        chosen_pair = next_pair(profile, pair_phi[favorite], [pair_ids[i] for i in favorite],
                                shown_pairs=shown, pool_size=max(2, len(favorite)), rng=pair_seed)
    if chosen_pair is None:
        chosen_pair = next_pair(profile, pair_phi, pair_ids, shown_pairs=shown, pool_size=PAIR_POOL,
                                rng=pair_seed, preference_scores=preferences, anchor_ids=liked_ids,
                                preference_weight=PAIR_PREFERENCE_WEIGHT)
    return chosen_pair, len(pair_ids)


def recommend(profile, ids: Sequence[str], phi: np.ndarray, history, rejected, space,
              rank_seed: int, k: int = RECOMMENDATIONS, pool: int | None = None) -> list[tuple[str, float]]:
    """Top-k sans les sorties rejetées : score du profil + préférence, puis MMR.

    Les égalités de score sont départagées au hasard (graine de la séance) : beaucoup
    d'annonces ont le même φ. `pool` (simulation sur un grand catalogue) limite le MMR aux
    `pool` meilleures activités ; l'API classe toutes les activités éligibles.
    """
    keep = [i for i, item_id in enumerate(ids) if item_id not in rejected]
    if not keep:
        return []
    kept_ids, kept_phi = [ids[i] for i in keep], phi[keep]
    scores = profile.scores(kept_phi) + PREFERENCE_WEIGHT * preference_scores(
        kept_ids, kept_phi, history, space, ids, phi)
    order = np.random.default_rng(rank_seed).permutation(len(kept_ids))
    if pool is not None and len(order) > pool:
        order = order[np.argsort(-scores[order], kind="stable")[:pool]]
    picked = mmr_rerank(scores[order], kept_phi[order], k, DIVERSITY)
    return [(kept_ids[order[i]], float(scores[order[i]])) for i in picked]
