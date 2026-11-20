# Cœur ML : features, profil de session, paires et classement

Code dans `src/swiparis/ml/`, tests dans `tests/test_ml_core.py`. Seule
dépendance : `numpy`, déclarée en option pour ne pas alourdir la démo de données.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sans numpy, les tests ML sont signalés « skipped » au lieu d'échouer. Le test sur
les vraies données tourne seulement si `data/processed/common.sqlite` est présent.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `features.py` | `Item` (activité dans la taxonomie commune) et `FeatureSpace`, qui calcule φ(i) |
| `items.py` | Lit `common.sqlite` et précalcule φ pour toutes les activités (`encode_common`) |
| `model.py` | `SessionProfile` : score, P(a ≻ b), mise à jour après chaque réponse |
| `selection.py` | `next_pair` (acquisition), `session_finished` (arrêt), `rank` (classement avec diversité) |

## Features φ(i)

Même encodage pour Foursquare (entraînement) et Paris Data (démo), 27 dimensions :

- 10 catégories communes de `common.sqlite`, triées par code, en multi-hot
  normalisé (une annonce Paris Data peut en avoir plusieurs) ;
- prix : `free`, `conditional_free`, `paid`, `unknown`. Le prix inconnu a sa
  propre dimension, il n'est jamais traité comme gratuit. Tous les lieux
  Foursquare sont `unknown` : le poids du prix ne s'apprend qu'en session ;
- moment de la journée : part des visites (Foursquare) ou des séances
  (Paris Data) la nuit (0-6 h), le matin, l'après-midi et le soir ;
- zone : cellule d'une grille 3 × 3 sur le rectangle Paris des imports ;
- embedding texte : prévu (`text_dim`), désactivé par défaut.

`FeatureSpace.names()` donne le nom de chaque dimension. **Le w₀ entraîné doit
être sauvegardé avec cette liste**, pour vérifier au chargement que l'ordre est
le même.

Fuite de données : `encode_common(..., users=...)` calcule le moment de la
journée des lieux Foursquare avec les visites des seuls utilisateurs donnés.
Pour l'entraînement et l'évaluation, passer les utilisateurs d'entraînement ;
`users=None` (toutes les visites) est réservé à la démo.

## Modèle

Bradley-Terry linéaire : P(a ≻ b | u) = σ(wᵤᵀ(φ(a) − φ(b))), score(i) = wᵤᵀφ(i).

Après une réponse gauche ou droite, avec Δ = φ(choisi) − φ(rejeté) :

- mode `sgd` (par défaut) : w ← w + η(1 − σ(wᵀΔ))Δ − λ(w − w₀). Le rappel se
  fait vers le prior w₀ et pas vers 0, sinon le prior appris hors ligne
  s'efface au fil de la session. Avec w₀ = 0, on retrouve la règle −λw ;
- mode `bayes` : régression logistique bayésienne en ligne (Laplace). Le profil
  est N(μ, Σ). Σ diminue dans la direction des choix déjà faits, et la
  probabilité prédite en tient compte : σ(μᵀΔ / √(1 + πΔᵀΣΔ/8)).

Un passage (`skip`) ne modifie pas le profil.

## Choix de la paire suivante

A(i, j) = H(P(i ≻ j)) + β(1 − cos(φ(i), φ(j))), avec H l'entropie binaire.
L'entropie seule choisirait deux activités presque identiques (P ≈ 0,5 car
Δ ≈ 0), qui n'apprennent rien. Le terme cosinus corrige ça. La recherche se fait
dans un pool : moitié des activités les mieux classées (égalités départagées au
hasard), moitié tirée au hasard. Les paires déjà montrées et les paires de
vecteurs identiques sont exclues. Le côté gauche/droite est tiré par le journal
(`LogStore.offer_comparison`), pas par le modèle.

Arrêt (guide de l'équipe) : `session_finished` devient vrai après 5 réponses
gauche/droite ou 8 paires proposées (passages et paires sans réponse compris).

## Classement

Score wᵀφ(i), puis re-ranking MMR avec un paramètre `diversity` (0 = simple tri
par score). `rank(..., exclude=...)` retire les activités déjà proposées en
comparaison : le journal refuse de les recommander. `intra_list_diversity`
mesure la diversité d'une liste, pour l'ablation.

## Brancher dans l'API

```python
from swiparis.catalog import Catalog
from swiparis.logging_store import LogStore
from swiparis.ml import SessionProfile, encode_common, next_pair, rank, session_finished

DB = "data/processed/common.sqlite"
catalog = Catalog(DB)
store = LogStore("data/pilot/interactions.sqlite", catalog)
encoded = encode_common(DB, users=None, sources=["paris_data"])   # une fois au démarrage
# Par session : profil (à garder avec to_dict/from_dict) et paires proposées
profile = SessionProfile(w0)                                     # w0 = zéros tant que Martin n'a pas livré
eligible = [i["item_id"] for i in catalog.list_items(constraints, limit=None)["items"]]
phi = encoded.rows(eligible)
if not session_finished(profile.n_updates, len(offered)):
    pair = next_pair(profile, phi, eligible, shown_pairs=offered)  # puis store.offer_comparison(...)
# Après store.answer(...) : profile.answer(encoded.vector(left), encoded.vector(right), choice)
top = rank(profile, phi, eligible, k=10, exclude={i for p in offered for i in p})
# Puis store.show_recommendations(...) avec les identifiants de top
```

## Valeurs par défaut (à régler sur la validation)

| Paramètre | Défaut |
| --- | --- |
| η (`eta`) | 0,5 |
| λ (`lam`) | 0,05 |
| σ₀² (`prior_var`, mode bayes) | 1,0 |
| β (`beta`) | 0,5 |
| `pool_size` | 40 |
| `diversity` | 0,3 |

Sur le vrai catalogue (1 778 annonces) : encodage de toute la base (8 332
activités) en 0,2 s, et une session complète (5 paires + top 10) en 10 à 12 ms.
On est très loin de la limite de 2 s.

## Simulation sur le vrai catalogue (20/11/2026, sans prior)

Utilisateur simulé qui préfère une catégorie et répond au hasard quand aucune des
deux activités ne l'a. On mesure la part de son top 10 dans cette catégorie,
après 5 réponses, sur 10 à 20 sessions. Ce n'est pas l'évaluation officielle
(pas de vrais utilisateurs, pas de w₀).

- Catégories fréquentes : le modèle apprend bien. Culture 0,73 à 0,89 et
  spectacle 0,74 à 0,92, contre 0,38 au hasard.
- Catégories rares (nature 4 % du catalogue, restauration 2 %) : à peine mieux
  que le hasard. En 5 paires, elles apparaissent rarement. C'est la raison d'être
  du prior w₀ appris sur Foursquare.
- Ajouter un terme de « nouveauté » à l'acquisition n'a rien changé (moyenne
  0,35 à 0,38) : non retenu.
- Tirer le pool catégorie par catégorie aide les catégories rares (nature 0,06 →
  0,16, sport 0,13 → 0,41) mais dégrade les fréquentes (culture 0,84 → 0,51),
  pour la même moyenne. À trancher dans l'ablation avec le vrai w₀.

## État d'avancement (20/11/2026, Merwane)

Fait :

- φ(i) commun Foursquare / Paris Data, lu directement depuis `common.sqlite`.
- Modèle Bradley-Terry et mise à jour de session (modes `sgd` et `bayes`),
  passages gérés.
- Choix de la paire suivante, arrêt après 5 réponses ou 8 paires, classement
  avec diversité qui exclut les activités déjà proposées.
- numpy déclaré en option `[ml]`. 37 tests passent (dont 24 pour le cœur ML,
  avec un test sur la vraie base).

Reste à faire :

- Les 3 profils exemples du rapport de semaine 3, avec le vrai w₀ et les splits.
- La spécification de l'algorithme (1 à 2 pages) avec Yvan.
- Régler η, λ, β, `diversity` et le type de pool sur la validation, avec Martin.

En attente de :

- **Martin** : le prior w₀ entraîné avec ce `FeatureSpace` (27 dimensions, ordre
  de `FeatureSpace.names()`), sauvegardé avec la liste des noms.
- **Yvan** : les splits (utilisateurs train/val/test et règle de découpage) ;
  valider dans la spécification le rappel vers w₀ au lieu de −λw.
- **Amine** : brancher selon la section « Brancher dans l'API ».
