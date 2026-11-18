# Cœur ML : features, profil de session, paires et classement

Code dans `src/swiparis/ml/`, tests dans `tests/test_ml_core.py`. Seule
dépendance : `numpy`.

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m unittest tests.test_ml_core -v
```

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `features.py` | `Item` (activité ramenée à la taxonomie commune) et `FeatureSpace`, qui calcule φ(i) |
| `catalog.py` | Construit les `Item` depuis les bases SQLite Foursquare et Paris Data |
| `model.py` | `SessionProfile` : score, P(a ≻ b) et mise à jour après chaque choix |
| `selection.py` | `next_pair` (fonction d'acquisition) et `rank` (score puis re-ranking pour la diversité) |

## Features φ(i)

Même encodage pour Foursquare (entraînement) et Paris Data (démo), une fois les
catégories ramenées à la taxonomie commune :

- catégories communes : multi-hot normalisé ;
- prix : `free`, `conditional_free`, `paid`, `unknown`. Le prix inconnu a sa
  propre dimension, il n'est jamais traité comme gratuit. Foursquare n'a pas de
  prix : tous ses lieux sont `unknown`, donc le poids du prix ne s'apprend
  qu'en session ;
- moment de la journée : part des visites (Foursquare) ou des séances
  (Paris Data) la nuit (0-6 h), le matin, l'après-midi et le soir ;
- zone : cellule d'une grille 3 × 3 sur le rectangle Paris des imports ;
- embedding texte : prévu (`text_dim`), désactivé par défaut.

Fuite de données : `foursquare_items(..., users=...)` calcule le moment de la
journée avec les check-ins des seuls utilisateurs donnés. Passer les
utilisateurs d'entraînement.

## Modèle

Bradley-Terry linéaire : P(a ≻ b | u) = σ(wᵤᵀ(φ(a) − φ(b))), score(i) = wᵤᵀφ(i).

Après un choix, avec Δ = φ(choisi) − φ(rejeté) :

- mode `sgd` (par défaut) : w ← w + η(1 − σ(wᵀΔ))Δ − λ(w − w₀). Le rappel se
  fait vers le prior w₀ et pas vers 0, sinon le prior appris hors ligne
  s'efface au fil de la session. Avec w₀ = 0, on retrouve la règle −λw ;
- mode `bayes` : régression logistique bayésienne en ligne (Laplace). Le profil
  est N(μ, Σ). Σ diminue dans la direction des choix déjà faits, et la
  probabilité prédite en tient compte : σ(μᵀΔ / √(1 + πΔᵀΣΔ/8)).

## Choix de la paire suivante

A(i, j) = H(P(i ≻ j)) + β(1 − cos(φ(i), φ(j))), avec H l'entropie binaire.
L'entropie seule choisirait deux activités presque identiques (P ≈ 0,5 car
Δ ≈ 0), qui n'apprennent rien. Le terme cosinus corrige ça. La recherche se fait
dans un pool (moitié des mieux classées, moitié au hasard). Les paires déjà
montrées et les paires de vecteurs identiques sont exclues. Le côté
gauche/droite est tiré au hasard.

## Classement

Les filtres (créneau, arrondissement, budget) s'appliquent avant, côté API.
Ensuite : score wᵀφ(i), puis re-ranking MMR avec un paramètre `diversity`
(0 = simple tri par score). `intra_list_diversity` mesure la diversité d'une
liste, pour l'ablation.

## Valeurs par défaut (à régler sur la validation)

| Paramètre | Défaut |
| --- | --- |
| η (`eta`) | 0,5 |
| λ (`lam`) | 0,05 |
| σ₀² (`prior_var`, mode bayes) | 1,0 |
| β (`beta`) | 0,5 |
| `pool_size` | 40 |
| `diversity` | 0,3 |

Mesure sur 1 778 activités aléatoires : moins de 2 ms par paire et moins de
10 ms pour un top 10, donc loin de la limite de 2 s.

## État d'avancement (18/11/2026, Merwane)

Fait :

- φ(i) commun à Foursquare et Paris Data, avec le chargement des deux bases SQLite.
- Modèle Bradley-Terry et mise à jour de session (modes `sgd` et `bayes`).
- Choix de la paire suivante (entropie + distance cosinus) et classement avec
  re-ranking pour la diversité.
- 20 tests unitaires, dont les vérifications de la semaine 2 (inverser les
  choix change l'activité classée première, une paire déjà montrée est exclue).

Reste à faire :

- Vérifier les profils exemples : 3 utilisateurs et leurs recommandations pour
  le rapport de semaine 3.
- Écrire la spécification de l'algorithme (1 à 2 pages) avec Yvan.
- Régler η, λ, β et `diversity` sur la validation, avec Martin.
- Comparer avec le prototype Node.js s'il est retrouvé (il n'est pas dans le repo).

En attente de :

- **Jeromsan** : la taxonomie commune (libellé source → catégorie commune), les
  bases `foursquare.sqlite` et `paris_data.sqlite`, et l'ajout de `numpy` aux
  dépendances du `pyproject.toml`.
- **Martin** : le prior w₀ entraîné avec le même `FeatureSpace` (même ordre des
  dimensions, voir `FeatureSpace.names()`).
- **Yvan** : valider dans la spécification le rappel vers w₀, −λ(w − w₀), au
  lieu de −λw ; fournir la liste des utilisateurs d'entraînement, utilisée par
  `foursquare_items` pour éviter les fuites.
- **Amine** : `next_pair` et `rank` attendent des activités déjà filtrées
  (créneau, arrondissement, budget). L'état d'une session se sauvegarde avec
  `SessionProfile.to_dict()` et se recharge avec `SessionProfile.from_dict()`.

À caler en réunion : les signatures prévues étaient `next_pair(session, contraintes)`
et `rank(session, contraintes)`. Ici, elles prennent le profil, la matrice φ et
les identifiants des activités éligibles. L'API peut les envelopper avec ses filtres.
