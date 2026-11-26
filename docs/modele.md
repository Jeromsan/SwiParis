# Cœur ML : features, profil de session, paires, prior et évaluation

Code dans `src/swiparis/ml/`, tests dans `tests/test_ml_*.py`. Seule dépendance :
`numpy`, déclarée en option pour ne pas alourdir la démo de données.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ml]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sans numpy, les tests ML sont marqués « skipped » au lieu d'échouer (les tests de
Martin demandent `requirements-ml.txt`). Les tests sur les vraies données ne
tournent que si `data/processed/common.sqlite` est présent.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `features.py` | `Item` et `FeatureSpace`, qui calcule φ(i) |
| `items.py` | Lit `common.sqlite`, précalcule φ (`encode_common`), visites et popularité |
| `model.py` | `SessionProfile` : score, P(a ≻ b), mise à jour après chaque réponse |
| `selection.py` | `next_pair` (acquisition), `session_finished` (arrêt), `rank` (diversité) |
| `splits.py` | Exporte le découpage unique d'Yvan au format de `swiparis.pairs` |
| `prior.py` | Apprend le prior w₀ dans l'espace φ (`models/prior_w0_phi.json`) |
| `simulation.py` | Sessions simulées : scores pour `evaluation.py`, baselines, ablation |
| `examples.py` | Les 3 profils exemples (`docs/profils_exemples.md`) |
| `evaluation.py` | Découpage, métriques et bootstrap (Yvan) |

## Features φ(i)

Même encodage pour Foursquare (entraînement) et Paris Data (démo), 28 dimensions
(`FeatureSpace.names()`) :

- 10 catégories communes, triées par code, en multi-hot normalisé ;
- prix : `free`, `conditional_free`, `paid`, `unknown`. Le prix inconnu n'est
  jamais traité comme gratuit. Tous les lieux Foursquare sont `unknown` ;
- moment de la journée : part des visites (Foursquare) ou des séances (Paris
  Data) la nuit, le matin, l'après-midi et le soir ;
- zone : cellule d'une grille 3 × 3 sur le rectangle Paris des imports ;
- popularité : log(1 + visites) / log(1 + maximum), sur les visites des
  utilisateurs d'entraînement. Les annonces Paris Data n'ont pas de visites :
  le bloc vaut 0 pour elles, il ne joue donc pas dans la démo.

Contre les fuites, le moment de la journée et la popularité ne sont calculés
qu'avec les visites des utilisateurs d'entraînement (`users=` et
`popularity_users=` de `encode_common`).

## Modèle de session

Bradley-Terry linéaire : P(a ≻ b | u) = σ(wᵤᵀ(φ(a) − φ(b))), score(i) = wᵤᵀφ(i).
Après une réponse gauche ou droite, avec Δ = φ(choisi) − φ(rejeté) :

- mode `sgd` (référence) : w ← w + η(1 − σ(wᵀΔ))Δ − λ(w − w₀) ;
- mode `bayes` : régression logistique bayésienne en ligne (Laplace).

Un passage (`skip`) ne modifie pas le profil. L'évaluation hors ligne garde son
budget de 5 réponses/8 offres afin de préserver le protocole et les résultats
publiés. La démo interactive utilise 10 réponses/15 offres, avec la mise à jour
SGD actuelle (η = 0,5 par défaut, réglable avec `SWIPARIS_SESSION_ETA`, λ = 0,05). Les choix sont rejoués depuis le journal à
chaque requête. Les sorties choisies restent éligibles au classement final et
les sorties rejetées sont écartées. Cette politique est identifiée par le
suffixe `session-v3` dans la version exposée par l'API. Un score de préférence
complémentaire combine les catégories plébiscitées/écartées et la similarité
des caractéristiques avec les sorties choisies, avec une importance qui
décroît avec leur ancienneté.

## Choix des paires et classement

A(i, j) = H(P(i ≻ j)) + β(1 − cos(φ(i), φ(j))) + un bonus de préférence,
dans un pool de candidats (moitié les mieux classées, moitié échantillonnée).
Les sorties choisies sont ajoutées comme ancres à ce pool. Une comparaison sur
quatre peut opposer deux sorties déjà choisies. La bibliothèque utilise un pool
de 40 par défaut ; l'API sert 80 candidats avec un tirage reproductible pour la
séance. Les paires déjà montrées et les vecteurs identiques sont exclus. Le côté
gauche/droite est tiré par le journal (`LogStore.offer_comparison`).

Classement : score wᵀφ(i) combiné au score de préférence récent, puis re-ranking
MMR (`diversity`). Les trois premiers résultats sont affichés. Les activités
explicitement rejetées sont écartées ; celles choisies restent éligibles et sont
signalées dans les résultats. Le formulaire propose toute la journée par défaut,
ou un moment de la journée / un horaire précis, et vérifie qu'au moins six
sorties correspondent aux contraintes avant de créer une séance.

## Prior w₀

`prior.py` apprend w₀ avec le même modèle que la session (perte logistique sur les
paires de Martin, utilisateurs d'entraînement seulement), résolu exactement par
Newton. λ est choisi sur la perte de validation. Deux corrections, mesurées :

- **Moment de la journée exclu de w₀.** Dans une paire d'entraînement, le lieu
  choisi a toujours un moment connu, puisque sa propre visite le calcule (100 %
  contre 79 % en validation). Avec ce bloc, w₀ ordonnait 60 % des paires
  d'entraînement mais 50 % en validation. Ce bloc reste à 0 dans w₀ ; la session
  l'apprend.
- **Popularité sans la visite de l'utilisateur.** Pour la même raison, la
  popularité du lieu choisi d'une paire d'entraînement est recalculée sans les
  visites de cet utilisateur.

Les négatifs de Martin sont tirés selon la popularité : sur ces paires, être
populaire distingue peu le choisi du rejeté. Le poids de popularité de w₀ est donc
multiplié par un facteur choisi sur le NDCG@10 de validation (×2 retenu).
Effet de bord : comme les rejetés sont souvent des restaurants proches et
populaires, w₀ pénalise la restauration (−0,37) pour tout le monde. C'est un
artefact des négatifs, pas un vrai goût.

## Protocole d'évaluation

- Découpage unique (`splits.py`) : celui d'Yvan, graine 42, 490 / 163 / 163
  utilisateurs. Avant, `swiparis.pairs` tirait son propre découpage : 93 des 163
  utilisateurs test servaient à entraîner le w₀ de Martin. Les paires sont
  maintenant générées avec `--splits`.
- Pour chaque utilisateur évalué : ses 5 premiers lieux distincts forment le
  profil, les suivants servent seulement à mesurer.
- Les réponses aux paires sont **simulées** : l'utilisateur choisit l'activité la
  plus proche de ses lieux de profil (cosinus sur catégorie, prix, moment, zone).
- Méthodes : session avec sélection active, session avec paires tirées au hasard
  (ablation, même budget), w₀ seul, popularité, hasard.
- Réglages choisis sur la validation : λ = 1e-4, popularité ×2, η = 0,05
  (meilleur NDCG@10). Le test a été lancé une fois, avec ces réglages.

## Résultats

Test, 162 utilisateurs évalués (script officiel `swiparis.ml.evaluation` pour
active/uniforme ; mêmes fonctions pour les autres méthodes) :

| Méthode | NDCG@10 [IC 95 %] | Recall@20 [IC 95 %] | AUC |
| --- | --- | --- | --- |
| Session, sélection active | 0,086 [0,067 ; 0,107] | 0,077 [0,062 ; 0,092] | 0,662 |
| Session, paires au hasard | 0,088 [0,068 ; 0,110] | 0,083 [0,068 ; 0,099] | 0,660 |
| w₀ seul | 0,083 [0,066 ; 0,105] | 0,083 [0,067 ; 0,099] | 0,655 |
| Popularité | 0,067 [0,050 ; 0,084] | 0,081 [0,066 ; 0,097] | 0,654 |
| Hasard | 0,003 [0,001 ; 0,007] | 0,003 [0,001 ; 0,006] | 0,491 |

Écarts appariés sur le test (moyenne, IC 95 % bootstrap) :

| Comparaison | NDCG@10 | Recall@20 |
| --- | --- | --- |
| active − popularité | +0,019 [+0,009 ; +0,030] | −0,004 [−0,013 ; +0,005] |
| w₀ seul − popularité | +0,016 [+0,009 ; +0,024] | +0,002 [−0,007 ; +0,012] |
| active − paires au hasard | −0,002 [−0,009 ; +0,005] | −0,006 [−0,012 ; −0,000] |
| active − w₀ seul | +0,003 [−0,005 ; +0,010] | −0,006 [−0,012 ; −0,001] |

Compromis précision / personnalisation selon η (validation, sélection active) :

| η | NDCG@10 | Top 10 commun entre deux utilisateurs (Jaccard) | Top 10 dans les catégories du profil | Couverture du catalogue |
| --- | --- | --- | --- | --- |
| 0,05 | 0,097 | 0,64 | 0,43 | 0,3 % |
| 0,2 | 0,085 | 0,32 | 0,44 | 1,5 % |
| 0,5 | 0,058 | 0,06 | 0,64 | 5,8 % |
| 1,0 | 0,022 | 0,02 | 0,81 | 9,5 % |

Fichiers : `docs/resultats_ml/` (rapports JSON) et `docs/profils_exemples.md`.

## Lecture honnête

- Le modèle bat le hasard de très loin, et la popularité sur le NDCG@10 (écart
  apparié significatif). Il fait jeu égal avec elle sur le Recall@20.
- Ce gain vient du prior w₀ (popularité + catégorie + zone), pas de la session :
  avec η = 0,05, les mises à jour bougent peu le profil, et les utilisateurs
  reçoivent presque la même liste (64 % en commun). Les profils exemples le
  montrent : un utilisateur qui choisit des restaurants reçoit les lieux de
  culture les plus célèbres.
- La sélection active ne fait pas mieux que les paires au hasard dans ce
  protocole. Pas de preuve de l'innovation pour l'instant (Q4).
- Pourquoi : φ est grossier (des centaines de lieux ont le même φ), les réponses
  sont simulées, et les visites futures Foursquare vont surtout vers les lieux les
  plus fréquentés. Personnaliser fait baisser le NDCG dans ce protocole.

## Réglage de la séance (mesure de la configuration réelle de l'application)

Le tableau ci-dessus mesure 5 réponses et le modèle seul. L'application fait jusqu'à 10
réponses, affiche 3 recommandations et ajoute les règles de préférence. Cette politique est
maintenant dans un seul module, `swiparis.ml.policy`, utilisé par l'API **et** par la mesure
hors ligne `swiparis.ml.app_eval` (même protocole de profil et de réponses simulées).
Résultats : `docs/resultats_ml/app_eval/`.

Validation, 163 utilisateurs, catalogue Foursquare (avec popularité) :

| Profil | η | NDCG@10 | Hit@3 | Activités communes entre deux top 3 |
| --- | --- | --- | --- | --- |
| global | 0,05 | 0,066 | 0,196 | 0,50 |
| global | 0,2 | 0,057 | 0,202 | 0,25 |
| global | 0,5 | 0,030 | 0,080 | 0,10 |
| personas | 0,05 | 0,102 | 0,288 | 1,40 |
| personas | 0,2 | 0,091 | 0,252 | 0,86 |
| personas | 0,5 | 0,076 | 0,215 | 0,54 |
| w₀ seul, sans question | – | 0,098 | 0,270 | – |
| popularité | – | 0,091 | 0,245 | – |

Test (lu une fois) : personas, η = 0,05 : NDCG@10 0,079 ; écart avec la configuration
d'origine (global, η = 0,5) +0,052 [+0,033 ; +0,072], Hit@3 +0,099 [+0,031 ; +0,173].

**Mais ces écarts viennent de la popularité, que l'application n'a pas.**
- À poids de popularité égal, personas et profil global font jeu égal (validation :
  −0,002 [−0,017 ; +0,014]). Les personas ont un poids de popularité ×16 contre ×2 pour w₀.
- Les annonces Paris Data n'ont aucune visite : leur bloc popularité vaut 0. En refaisant
  la mesure sans signal de popularité (`--no-popularity`, comme dans l'application), toutes
  les configurations tombent autour de NDCG@10 = 0,005 et ne se distinguent plus
  (η = 0,05 contre 0,5 : +0,002 [−0,002 ; +0,006] ; personas contre global à η = 0,05 :
  +0,001 [−0,005 ; +0,006]).
- Conclusion : l'évaluation Foursquare mesure surtout la popularité. Elle ne permet pas de
  choisir η ni de prouver un gain des personas **pour l'application**. Le « problème de
  η = 0,5 » signalé avant venait de là.

Décision dans le code : η = 0,5 (choix d'Amine, celui qui personnalise le plus), réglable
sans toucher au code avec `SWIPARIS_SESSION_ETA` ; personas activées (`SWIPARIS_PERSONAS`,
`off` pour les retirer), présentées comme une explication des choix, sans gain de précision
prouvé. Les règles de préférence sont gardées (les retirer ne change rien de mesurable avec
les personas). Ce qui tranchera : les choix réels du pilote (taux d'ouverture et de sorties
gardées par réglage).

## Décisions à prendre en équipe

1. **η de la démo.** Hors ligne, rien ne départage 0,05, 0,2 et 0,5 dans les conditions de
   l'application (voir ci-dessus). Le code garde 0,5 ; changer de valeur se fait avec
   `SWIPARIS_SESSION_ETA`, sans relancer le test. Pour trancher avec de vrais utilisateurs :
   lancer le pilote avec `SWIPARIS_SESSION_ETA_AB="0.2,0.5"`, puis
   `python -m swiparis.ml.pilot_eta` (part des recommandations ouvertes et gardées par réglage,
   écart avec intervalle de confiance). Ne pas conclure tant que l'intervalle contient 0.
2. **Restauration et vie nocturne** (question de Martin). Les retirer de
   l'entraînement enlèverait aussi l'artefact du w₀ sur la restauration. Il faut
   alors relancer `pairs`, `prior` puis l'évaluation.

## Reproduire

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m swiparis.ml.splits                                    # data/processed/splits.csv
python -m swiparis.pairs --db data/processed/common.sqlite --output data/processed/pairs.csv `
  --manifest outputs/pairs_manifest.json --seed 42 --splits data/processed/splits.csv
python -m swiparis.ml.prior --force                             # models/prior_w0_phi.json
python -m swiparis.ml.simulation --split val --eta 0.05         # réglages sur la validation
python -m swiparis.ml.simulation --split test --eta 0.05 --scores outputs/scores_test.csv
python -m swiparis.ml.evaluation --scores outputs/scores_test.csv --output outputs/evaluation_test.json
python -m swiparis.ml.examples                                  # docs/profils_exemples.md
```

## Brancher dans l'API

```python
from swiparis.catalog import Catalog
from swiparis.logging_store import LogStore
from swiparis.ml import SessionProfile, encode_common, next_pair, rank, session_finished
from swiparis.ml.prior import load_prior
from swiparis.ml.splits import load_splits

DB = "data/processed/common.sqlite"
train = {u for u, s in load_splits("data/processed/splits.csv").items() if s == "train"}
catalog = Catalog(DB)
store = LogStore("data/pilot/interactions.sqlite", catalog)
encoded = encode_common(DB, users=train, sources=["paris_data"], popularity_users=train)  # au démarrage
w0 = load_prior("models/prior_w0_phi.json", encoded.space)        # vérifie l'ordre des features
profile = SessionProfile(w0)                                      # par session (to_dict/from_dict)
eligible = [i["item_id"] for i in catalog.list_items(constraints, limit=None)["items"]]
phi = encoded.rows(eligible)
if not session_finished(profile.n_updates, len(offered)):
    pair = next_pair(profile, phi, eligible, shown_pairs=offered)  # puis store.offer_comparison(...)
# Après store.answer(...) : profile.answer(encoded.vector(left), encoded.vector(right), choice)
top = rank(profile, phi, eligible, k=10, exclude={i for p in offered for i in p})
```

## État d'avancement (20/11/2026, Merwane)

Fait : φ commun, modèle de session, sélection des paires, arrêt 5/8, classement
avec diversité, branchement sur `common.sqlite` et le journal ; découpage unique ;
prior w₀ dans l'espace φ ; évaluation test complète avec ablation ; 3 profils
exemples. 62 tests passent.

Reste : décisions d'équipe ci-dessus ; mettre à jour `docs/evaluation.md`
(Yvan : il annonce encore qu'il n'y a ni w₀ ni résultats) ; brancher dans l'API
(Amine).
