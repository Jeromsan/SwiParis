# Branche `feature/personas-merwane` : ce qui a changé et comment reprendre

Auteur : Merwane (Core ML), 26/11/2026. Base : `feature/data-collection` au commit `050a44f`
(merge de la PR #10 d'Amine). 5 commits, 56 fichiers. Ce document suffit pour reprendre le
travail : il dit ce qui a changé, pourquoi, où, comment le lancer et ce qu'il reste à faire.

## 1. En bref

- **Personas** apprises sur les historiques Foursquare (6 groupes d'habitudes), utilisées par
  l'API : chaque séance part d'un mélange de personas, mis à jour après chaque choix, et l'appli
  affiche « Tes choix ressemblent le plus à : … ». Dès 2 sorties gardées, « Ma sélection » sert de
  point de départ.
- **« Garder » fonctionne vraiment** : sorties enregistrées par l'API, page « Ma sélection »
  (retirer, annuler, export agenda `.ics` et tableau `.csv`), retrouvée avec l'adresse de la page.
- **« Voir l'annonce » en un clic** (l'API enregistre l'ouverture puis redirige).
- **Correctifs** : filtre de créneau (les expositions étaient exclues), horaires Paris Data en
  avance d'une heure en hiver, φ de l'API différent de celui de l'entraînement, égalités de score
  départagées par ordre alphabétique, sortie choisie 4 fois remplacée par une « jumelle ».
- **Politique de séance dans un seul module** (`swiparis/ml/policy.py`), utilisé par l'API et par
  la mesure hors ligne : l'évaluation mesure enfin ce que fait l'appli.
- **η** reste à 0,5 ; réglable sans code ; outil pour le comparer sur le pilote. La simulation
  Foursquare ne départage pas les valeurs (l'écart venait de la popularité, absente des annonces
  Paris Data).
- **Tests** : 101 (dont les 2 tests torch), recette de l'interface : 48 vérifications.

## 2. Reprendre en 5 minutes

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[systems,ml]"
# data/processed/ (common.sqlite, pairs.csv, splits.csv) est dans le dépôt
export PYTHONPATH=$PWD/src            # voir « Pièges » : nécessaire sur certains Mac (Python 3.13)
.venv/bin/python -m unittest discover -s tests                                     # OK (skipped=2 sans torch)
.venv/bin/python -m uvicorn swiparis.api:app --host 127.0.0.1 --port 8000          # terminal 1
.venv/bin/python -m streamlit run src/swiparis/streamlit_app.py                    # terminal 2
.venv/bin/python scripts/check_ui_flow.py                                          # terminal 3, API lancée : 48/48
```

`/health` doit renvoyer une version du type
`prior-20261120-ac6b83d6+session-v4+eta0.5+personas6`.

## 3. Carte des changements

| Zone | Fichier | Ce qui change | Pourquoi |
| --- | --- | --- | --- |
| Personas | `src/swiparis/ml/personas.py` (nouveau) | Habitudes, k-moyennes, profil w_k par persona, `PersonaProfile`, entraînement, évaluation | Personnaliser à partir d'habitudes observées |
| Personas | `models/personas.json` (nouveau) | Modèle entraîné : 6 personas, noms, descriptions, réglages | Chargé par l'API |
| Politique | `src/swiparis/ml/policy.py` (nouveau) | Logique d'Amine déplacée sans changer son comportement : rejouer les réponses, `preference_scores`, choix de la paire, classement ; η, bras de comparaison, départage | Une seule implémentation pour l'API et la simulation |
| Mesure | `src/swiparis/ml/app_eval.py` (nouveau) | Simulation de la vraie politique (10 choix, top 3, préférences), options `--no-preferences`, `--no-popularity` | Mesurer l'appli réelle |
| Pilote | `src/swiparis/ml/pilot_eta.py` (nouveau) | Rapport : parts de recommandations ouvertes et gardées par valeur de η, écart avec IC 95 % | Trancher η avec de vrais utilisateurs |
| Simulation | `src/swiparis/ml/simulation.py` | `simulate_session` accepte un profil déjà construit et des limites de réponses | Réutilisé par les personas |
| API | `src/swiparis/api.py` | φ calculé comme à l'entraînement (`common_items`), politique via `policy`, personas, η par variable, routes « Ma sélection » et `/go`, `lifespan` | Voir sections 4 et 5 |
| Catalogue | `src/swiparis/catalog.py` | Règle de créneau (séance ≤ 3 h entière, ouverture longue ≥ 1 h), heure locale de Paris relue (`occurrence_utc`), filtre `item_ids` | Expos exclues ; décalage d'1 h |
| Journal | `src/swiparis/logging_store.py`, `logs_schema.sql` | Table `saved_items` (schéma v2, migration automatique), garder / retirer / annuler / lister | « Garder » réel |
| Interface | `src/swiparis/streamlit_app.py` | Mise en page d'Amine gardée ; « Garder » réel (aussi pendant les choix, plusieurs à la fois), page « Ma sélection », encadré persona, dates bornées, horaires lisibles, libellés, erreurs en français, lien `/go` | Boutons qui ne faisaient rien |
| Dépendances | `pyproject.toml` | `httpx` dans l'extra `systems` | Tests de l'API après l'installation du README |
| Nettoyage | `src/swiparis/ml/catalog.py` supprimé ; imports inutiles retirés (`serving.py`, `bprmf.py`, 2 tests) | Code mort | |
| Tests | `tests/test_api.py`, `test_policy.py`, `test_ml_personas.py`, `test_catalog_slots.py`, `common_fixture.py` (nouveaux) | API complète, politique, personas, créneaux | 101 tests |
| Scripts | `scripts/check_ui_flow.py`, `scripts/demo_personas.py` (nouveaux) | Recette de l'interface ; démo « mêmes choix, historiques différents » | |
| Docs | `docs/personas.md` (nouveau), `docs/modele.md`, `docs/team_handoff.md`, `README.md`, `.env.example` | Méthode, résultats, décisions, routes, variables | |
| Résultats | `docs/resultats_ml/personas_*.json`, `docs/resultats_ml/app_eval/*.json` | Rapports des mesures | Reproductibles |

## 4. Une séance, étape par étape (code actuel)

1. **Catalogue** (`Catalog.list_items`) : activités Paris Data compatibles avec le créneau, le
   quartier, la catégorie et le budget. Heures comparées avec `occurrence_utc(source_start)`.
2. **φ** : 28 nombres par activité (catégorie, prix, moment, zone, popularité), calculés une fois
   au démarrage (`api._encode_catalog`). Les annonces Paris Data ont une popularité de 0.
3. **Profil de départ** (`api.session_state`) : `PersonaProfile` = mélange des 6 personas. Point de
   départ : parts des personas, ou « Ma sélection » si le carnet (`owner_id`) a ≥ 2 sorties gardées
   (`personas.habit_from_phi` puis `PersonaModel.start`). Sans `models/personas.json` : `SessionProfile(w₀)`.
4. **Choix de la paire** (`policy.choose_pair`) : sans les sorties rejetées ; préférence des choix
   récents ; toutes les 4 paires, deux sorties déjà choisies face à face. Graine = séance + rang.
5. **Après chaque réponse** (`policy.replay`, rejoué depuis le journal à chaque requête) : π des
   personas mis à jour (π_k ∝ π_k σ(w_kᵀΔ)^τ), puis pas de gradient η vers le choix. Une sortie qui
   perd une comparaison devient « rejetée » et ne revient plus.
6. **Arrêt** : 10 réponses ou 15 paires (`policy.MAX_ANSWERS`, `MAX_OFFERS`).
7. **Recommandations** (`policy.recommend`) : score du profil + préférence ; égalités départagées
   d'abord en faveur des sorties choisies (`CHOSEN_TIEBREAK`), puis au hasard (graine de séance) ;
   MMR (diversité 0,3) ; 3 sorties. Une sortie choisie peut revenir (marquée « Tu avais choisi… »).
8. **Réponse** : liste, `persona.top` (2 personas, pourcentages), `persona.from_selection`,
   `session_eta`.

## 5. API : ce qui a changé

| Route | Changement |
| --- | --- |
| `GET /next_pair` | Paramètre facultatif `owner_id` ; réponse avec `session_eta` |
| `POST /recommendations` | Champ facultatif `owner_id` ; réponse avec `persona` et `session_eta` |
| `POST /api/saved` | Nouveau : garder 1 à 50 activités d'une séance (`owner_id`, `session_id`, `items: [{item_id, impression_id?}]`), sans doublon ; journalise aussi `save` une fois par recommandation |
| `POST /api/saved/remove`, `/api/saved/restore` | Nouveaux : retirer, annuler un retrait (`owner_id`, `item_ids`) |
| `GET /api/saved?owner_id=…` | Nouveau : sélection avec la fiche complète de chaque activité |
| `GET /go/{impression_id}` | Nouveau : enregistre `open` puis redirige vers l'annonce (303) |

Variables d'environnement (à définir avant de lancer l'API) :

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `SWIPARIS_SESSION_ETA` | `0.5` | Pas η de la séance |
| `SWIPARIS_SESSION_ETA_AB` | vide | Ex. `0.2,0.5` : chaque séance reçoit une valeur (tirée de son id), pour le pilote |
| `SWIPARIS_PERSONAS` | `models/personas.json` | Fichier des personas ; `off` pour les désactiver |
| `SWIPARIS_PUBLIC_API_URL` | `SWIPARIS_API_URL` | Adresse de l'API vue par le navigateur (liens `/go`) |

La version de `/health` contient ces réglages et elle est enregistrée avec chaque séance : c'est
ainsi que `pilot_eta` retrouve le η de chaque séance.

## 6. Base de logs (`data/pilot/interactions.sqlite`, pas dans Git)

- Schéma **v2** : nouvelle table `saved_items` (une ligne par carnet et activité ; retirer =
  `removed_at_utc` rempli). Une base v1 est migrée automatiquement au démarrage.
- **Attention** : la version d'Amine avant cette branche refuse une base v2. Pour revenir à une
  ancienne version, mettre la base de côté (`mv … interactions-v2.sqlite`).
- Le carnet (`owner_id`) est un identifiant aléatoire dans l'adresse Streamlit (`?carnet=…`),
  sans donnée personnelle.

## 7. Personas

- **Données** : 490 utilisateurs Foursquare d'entraînement. Habitudes = racine carrée des parts de
  visites par catégorie, moment de la journée (heure locale de la visite) et zone. k-moyennes.
- **Profil w_k** : même modèle et mêmes blocs que w₀, appris sur les paires des utilisateurs de la
  persona, rappelé vers le w global. Réglages choisis sur la validation : k = 6, λ = 0,01,
  popularité ×16, τ = 0,5, β = 3. Le test n'a servi qu'une fois.
- **Les 6 personas** : Bars · le soir · centre (19 %) ; Restaurants · le soir · ouest (12 %) ;
  Culture · l'après-midi · centre (16 %) ; Restaurants · l'après-midi · nord-ouest (11 %) ;
  Visites & Nature · l'après-midi · centre (20 %) ; Concerts & Culture · le soir · centre (22 %).
- **Résultat solide** : elles prédisent mieux les choix d'utilisateurs jamais vus, rattachés par
  leur historique (test, 5 358 paires : 62,4 % contre 57,7 %).
- **Ce qu'elles ne font pas** : pas de gain mesurable sur la qualité du top 10 ; à popularité
  égale, jeu égal avec le profil global. Elles servent à expliquer les recommandations et à partir
  de « Ma sélection ». À présenter ainsi.
- **Réentraîner** (après un nouveau `pairs.csv` ou un nouveau w₀) :
  `python -m swiparis.ml.personas train`, puis `evaluate --split val`, `set --temperature … --beta …`,
  `evaluate --split test --final`. Détails : `docs/personas.md`.

## 8. Mesures et décisions

| Mesure (validation, catalogue Foursquare) | NDCG@10 |
| --- | --- |
| Appli d'origine (global, η = 0,5) | 0,030 |
| Global, η = 0,05 | 0,066 |
| Personas, η = 0,05 | 0,102 |
| w₀ seul, sans question | 0,098 |
| **Toutes les configurations, sans signal de popularité** | ≈ 0,005, aucune différence |

- Les écarts ne viennent que de la popularité. Les annonces Paris Data n'en ont pas. Donc
  **aucune mesure hors ligne ne permet de choisir η** pour l'appli. Décision : η = 0,5 (choix d'Amine).
- Retirer les règles de préférence d'Amine ne change rien de mesurable avec les personas : gardées.
- **Pilote famille du 26/11** (10 séances, 5 à η = 0,2, 5 à η = 0,5) : 33 % des recommandations
  gardées dans les deux cas, aucune annonce ouverte ; échantillon petit et non représentatif, pas de
  conclusion. Fichier local `data/pilot/pilote-famille-26-11.sqlite` (pas dans Git).
- Tous les chiffres et intervalles : `docs/modele.md` (« Réglage de la séance ») et
  `docs/resultats_ml/app_eval/`.

## 9. Tests et scripts

| Commande | Ce que ça vérifie |
| --- | --- |
| `python -m unittest discover -s tests` | 101 tests (API complète, créneaux, politique, personas, journal…) ; 2 sautés sans torch |
| `python scripts/check_ui_flow.py` (API lancée) | L'interface comme un utilisateur : 48 vérifications (garder, retirer, annuler, export, persistance, liens) |
| `python scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models` | Parcours complet de l'API (script d'Amine) |
| `python scripts/demo_personas.py` | Mêmes 5 choix, historiques différents : listes différentes |
| `python -m swiparis.ml.app_eval --split val --eta 0.5 --profile personas` | Mesure de la vraie politique (ajouter `--no-popularity` pour les conditions de l'appli) |
| `python -m swiparis.ml.pilot_eta` | Rapport du pilote (ajouter `--include-test` pour les séances de test) |

Les 2 tests torch (`test_bprmf`, `test_train`) passent avec torch 2.15 (vérifié le 26/11). Pour
les lancer : `pip install torch matplotlib pandas`.

## 10. Pièges connus

- **`ModuleNotFoundError: No module named 'swiparis'` sur Mac (Python 3.13)** : le fichier `.pth`
  de l'installation éditable est ignoré. Faire `export PYTHONPATH=$PWD/src` dans chaque terminal.
- **`address already in use`** : une API tourne déjà sur le port 8000. `kill $(lsof -ti :8000)`.
- **« L'API a été relancée avec un autre réglage »** dans l'appli : la version a changé (η,
  personas). Recharger la page ou cliquer « Nouvelle séance ».
- **Données** : `data/processed/` (base, paires, découpage) est dans Git ; `data/pilot/` (réponses
  des participants) et `data/raw/` restent locales.
- **Garder la simulation et l'API alignées** : toute modification de la politique se fait dans
  `swiparis/ml/policy.py`, jamais directement dans `api.py`.

## 11. Ce qui reste à faire

1. Pilote plus large avec `SWIPARIS_SESSION_ETA_AB="0.2,0.5"`, puis `pilot_eta`. Ne conclure que si
   l'intervalle ne contient plus 0.
2. Si la PR est fusionnée : nouveau tag (ex. `v1-final-2`) plutôt que déplacer `v1-final`.
3. Pistes : un φ plus fin (beaucoup d'annonces ont le même φ, d'où les « jumelles ») ; une mesure
   sur des données Paris Data réelles quand le pilote en aura assez.

## 12. Consignes pour une IA qui reprend ce dépôt

Lire dans cet ordre : ce fichier → `src/swiparis/ml/policy.py` → `src/swiparis/api.py` →
`src/swiparis/ml/personas.py` → `docs/modele.md` → `docs/personas.md` → `tests/test_api.py`.
À ne pas casser :
- l'API et la simulation passent par `policy.py` (même comportement) ;
- une sortie rejetée n'est jamais reproposée (le journal le refuse aussi) ;
- le test n'est lu qu'une fois par configuration ; les réglages se choisissent sur la validation ;
- la version de `/health` décrit les réglages et sert au rapport du pilote ;
- lancer `python -m unittest discover -s tests` et `scripts/check_ui_flow.py` avant chaque commit ;
- ne jamais présenter des séances faites par l'équipe comme des données d'utilisateurs.
