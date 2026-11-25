"""Modèle de choix par paires (Bradley-Terry linéaire) et profil de session.

    P(a ≻ b | u) = σ(w_uᵀ (φ(a) − φ(b)))        score(i) = w_uᵀ φ(i)

Après chaque choix, avec Δ = φ(choisi) − φ(rejeté), deux mises à jour possibles :

- "sgd" : un pas de gradient sur la perte logistique −log σ(wᵀΔ), avec un rappel
  vers le prior w₀ appris hors ligne :
      w ← w + η (1 − σ(wᵀΔ)) Δ − λ (w − w₀)
  Avec w₀ = 0, c'est exactement la règle w ← w + η(1 − σ(wᵀΔ))Δ − λw.
  Rappeler vers w₀ plutôt que vers 0 évite d'effacer le prior au fil de la session.

- "bayes" : régression logistique bayésienne en ligne (approximation de Laplace,
  un pas de Newton par choix). Le profil est une gaussienne N(μ, Σ), de prior
  N(w₀, σ₀² I). Σ mesure l'incertitude sur le profil :
      p = σ(μᵀΔ),  c = p (1 − p)
      Σ ← Σ − c ΣΔΔᵀΣ / (1 + c ΔᵀΣΔ)
      μ ← μ + (1 − p) ΣΔ          (avec le Σ mis à jour)
  La probabilité prédite tient compte de cette incertitude (approximation probit) :
      P(a ≻ b) ≈ σ(μᵀΔ / √(1 + π ΔᵀΣΔ / 8))
"""
from __future__ import annotations

import numpy as np

MODES = ("sgd", "bayes")


def sigmoid(x):
    """σ(x) numériquement stable (pas de dépassement pour |x| grand)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


class SessionProfile:
    """Profil w_u d'un utilisateur pendant une session de comparaisons."""

    def __init__(self, prior, mode: str = "sgd", eta: float = 0.5, lam: float = 0.05,
                 prior_var: float = 1.0):
        prior = np.asarray(prior, dtype=float)
        if prior.ndim != 1 or prior.size == 0:
            raise ValueError("The prior w0 must be a non-empty vector")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if eta <= 0 or not 0 <= lam < 1 or prior_var <= 0:
            raise ValueError("Expected eta > 0, 0 <= lam < 1 and prior_var > 0")
        self.mode = mode
        self.eta = eta
        self.lam = lam
        self.prior_var = prior_var
        self.prior = prior.copy()
        self.w = prior.copy()  # μ en mode bayésien
        self.cov = prior_var * np.eye(prior.size) if mode == "bayes" else None
        self.n_updates = 0  # réponses gauche/droite intégrées (les passages ne comptent pas)

    @property
    def dim(self) -> int:
        return self.w.size

    def scores(self, phi) -> np.ndarray:
        """Score wᵀφ(i) de chaque activité (une par ligne)."""
        return np.asarray(phi, dtype=float) @ self.w

    def margin_variance(self, delta) -> np.ndarray:
        """ΔᵀΣΔ pour chaque ligne de `delta` (zéro en mode sgd)."""
        delta = np.atleast_2d(np.asarray(delta, dtype=float))
        if self.cov is None:
            return np.zeros(len(delta))
        return np.einsum("ij,jk,ik->i", delta, self.cov, delta)

    def prob(self, phi_a, phi_b):
        """P(a ≻ b). Accepte deux vecteurs, ou deux matrices avec une paire par ligne."""
        delta = np.asarray(phi_a, dtype=float) - np.asarray(phi_b, dtype=float)
        margin = delta @ self.w
        if self.cov is not None:
            variance = self.margin_variance(delta).reshape(np.shape(margin))
            margin = margin / np.sqrt(1 + np.pi * variance / 8)
        return sigmoid(margin)

    def update(self, phi_chosen, phi_rejected) -> float:
        """Intègre un choix ; renvoie P(choisi ≻ rejeté) prédite avant la mise à jour."""
        delta = np.asarray(phi_chosen, dtype=float) - np.asarray(phi_rejected, dtype=float)
        if delta.shape != self.w.shape:
            raise ValueError(f"Expected feature vectors of size {self.dim}")
        p = float(sigmoid(self.w @ delta))
        if self.mode == "sgd":
            self.w = self.w + self.eta * (1 - p) * delta - self.lam * (self.w - self.prior)
        else:
            c = p * (1 - p)
            s_delta = self.cov @ delta
            cov = self.cov - c * np.outer(s_delta, s_delta) / (1 + c * delta @ s_delta)
            self.cov = (cov + cov.T) / 2  # garde Σ symétrique malgré les arrondis
            self.w = self.w + (1 - p) * (self.cov @ delta)
        self.n_updates += 1
        return p

    def answer(self, phi_left, phi_right, choice: str) -> float | None:
        """Applique une réponse du journal : `left`, `right` ou `skip`.

        Un passage ne dit rien sur la préférence : le profil ne bouge pas et la
        méthode renvoie None. Sinon, renvoie la probabilité prédite du choix fait.
        """
        if choice == "left":
            return self.update(phi_left, phi_right)
        if choice == "right":
            return self.update(phi_right, phi_left)
        if choice == "skip":
            return None
        raise ValueError("choice must be left, right or skip")

    def to_dict(self) -> dict:
        """État sérialisable en JSON (pour garder la session côté API ou la journaliser)."""
        return {"mode": self.mode, "eta": self.eta, "lam": self.lam, "prior_var": self.prior_var,
                "prior": self.prior.tolist(), "w": self.w.tolist(),
                "cov": None if self.cov is None else self.cov.tolist(), "n_updates": self.n_updates}

    @classmethod
    def from_dict(cls, state: dict) -> "SessionProfile":
        profile = cls(state["prior"], state["mode"], state["eta"], state["lam"], state["prior_var"])
        profile.w = np.asarray(state["w"], dtype=float)
        if state["cov"] is not None:
            profile.cov = np.asarray(state["cov"], dtype=float)
        profile.n_updates = state["n_updates"]
        return profile
