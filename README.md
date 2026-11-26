# SwiParis

Application interactive de recommandation de sorties et loisirs à Paris. En dix choix entre paires d'activités, SwiParis adapte les comparaisons aux sorties déjà choisies et à leurs ressemblances, puis affiche les trois meilleures recommandations sous contraintes de disponibilité, d'arrondissement et de budget. La disponibilité par défaut couvre toute la journée pour permettre un vrai choix parmi les sorties annoncées.

> **Reprise de la branche `feature/personas-merwane`** : tout ce qui a changé (personas, « Ma sélection », correctifs, η, tests) et comment continuer est expliqué dans [docs/CHANGEMENTS_PERSONAS.md](docs/CHANGEMENTS_PERSONAS.md).

## Équipe

| Membre | Rôle | Responsabilités principales |
| --- | --- | --- |
| **Yvan Feugang** | Project Lead | Protocole d'évaluation, splits 60/20/20, contrôle des fuites, ablation |
| **Jeromsan Judes Ramesh** | Data Engineer | Collecte Foursquare & Paris Data, taxonomie commune, base SQLite et logs |
| **Merwane Kheloufi** | Core ML Engineer | Représentations d'items, modèle Bradley-Terry, sélection active, re-ranking MMR |
| **Martin Duverneuil** | ML Operations Engineer | Génération de paires, baselines, entraînement, versioning du prior (`w0`) et serving |
| **Amine Zaidi** | Systems Engineer | Backend FastAPI, interface Streamlit, intégration du modèle et démo locale |

## Données

Les données préparées sont versionnées dans le dépôt : `data/processed/common.sqlite` (base commune, 16 Mo), `data/processed/pairs.csv` et `data/processed/splits.csv` (paires et découpage pour l'entraînement et l'évaluation). Un clone suffit donc pour lancer l'appli, les tests et l'entraînement. Les données brutes (`data/raw/`) et le journal du pilote (`data/pilot/`, réponses des participants) ne sont pas versionnés. Le catalogue partagé comprend 816 utilisateurs, 31 694 visites Foursquare, 6 554 lieux et 1 778 annonces Paris Data, avec une taxonomie commune à dix catégories dans `config/common_taxonomy.json`.

Pour régénérer les bases, voir `docs/team_handoff.md`, `docs/foursquare.md` et `docs/paris_data.md`.

## Installation et lancement

Prérequis : Python 3.10 ou plus récent. Depuis PowerShell, à la racine du dépôt :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[systems,ml]"
```

Dans une première fenêtre PowerShell, lancer l'API :

```powershell
.\.venv\Scripts\python.exe -m uvicorn swiparis.api:app --host 127.0.0.1 --port 8000
```

Dans une deuxième fenêtre, lancer l'interface :

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/swiparis/streamlit_app.py
```

Sur macOS ou Linux, les mêmes étapes dans un terminal :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[systems,ml]"
.venv/bin/python -m uvicorn swiparis.api:app --host 127.0.0.1 --port 8000      # fenêtre 1
.venv/bin/python -m streamlit run src/swiparis/streamlit_app.py                # fenêtre 2
```

Ouvrir `http://localhost:8501`. La démo demande le consentement du pilote, vérifie qu'au moins six sorties correspondent aux filtres, journalise les comparaisons dans une base SQLite séparée, puis affiche les trois meilleures recommandations et leur carte. Les réponses récentes favorisent les catégories et sorties similaires aux choix précédents ; une comparaison sur quatre peut départager deux sorties déjà choisies. Les sorties explicitement écartées ne sont plus proposées. Sans clé Google Maps, une carte OpenStreetMap interactive s'affiche avec des liens Google Maps sur les repères. Pour utiliser la carte Google intégrée, fournir une clé navigateur restreinte via `GOOGLE_MAPS_API_KEY` dans l'environnement du processus API.

L'API charge le prior Bradley–Terry versionné dans `models/` et expose la version du prior et de la politique de session dans `/health`. Elle réapplique les choix au profil de la séance, utilise un signal de préférence récent pour les paires et le classement, garde les sorties choisies comme candidates et écarte celles explicitement rejetées. Si aucun prior versionné n'est présent, l'API utilise le prior de fréquence par catégorie. Voir [le guide du serving](docs/serving.md), [le guide du modèle](docs/modele.md) et [le guide d'équipe](docs/team_handoff.md).

### Sélection, personas et réglages

- **Ma sélection** (menu de gauche) : « Garder » enregistre vraiment la sortie, pendant les choix ou sur les recommandations (on peut en cocher plusieurs). La page liste les sorties gardées par jour ; on peut en retirer (confirmation pour un lot, puis « Annuler ») et les exporter vers un agenda (`.ics`) ou un tableau (`.csv`). La sélection est liée à l'adresse de la page (`?carnet=…`) : en revenant sur la même adresse, on la retrouve.
- **Voir l'annonce** : un clic, l'ouverture est enregistrée puis l'annonce s'ouvre (`/go`).
- **Personas** : après les choix, l'appli affiche les deux personas auxquelles les réponses ressemblent (groupes d'habitudes trouvés dans les données Foursquare, voir [docs/personas.md](docs/personas.md)).
- **Personas et « Ma sélection »** : dès qu'une personne a gardé au moins deux sorties, sa sélection sert de point de départ aux personas ; deux personnes qui font les mêmes choix mais n'ont pas gardé les mêmes sorties n'ont donc pas le même profil.
- **Réglages sans toucher au code** : `SWIPARIS_SESSION_ETA` (pas de mise à jour de la séance, défaut 0,5) et `SWIPARIS_PERSONAS` (`off` pour les désactiver), à définir avant de lancer l'API. Pourquoi 0,5 : voir « Réglage de la séance » dans [docs/modele.md](docs/modele.md).
- **Choisir η avec le pilote** : lancer l'API avec `SWIPARIS_SESSION_ETA_AB="0.2,0.5"` ; chaque séance reçoit une des deux valeurs. Ensuite, `.venv/bin/python -m swiparis.ml.pilot_eta` compare les réglages sur les vraies réponses (part des recommandations ouvertes et gardées, avec intervalle de confiance).
- **Recette de l'interface** (API lancée) : `.venv/bin/python scripts/check_ui_flow.py` clique comme un utilisateur (48 vérifications).

## Vérification du parcours API

Avec la base partagée disponible localement :

```powershell
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
```

Pour la préparation des données et le rapport de répartition des visites, voir [le guide d'équipe](docs/team_handoff.md) et [le résumé des données](docs/data_summary.md).

## Ré-entraînement du modèle

Pour relancer le pipeline MLOps, installez les dépendances avec `.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt`, puis suivez [le guide du pipeline](docs/ml_pipeline.md) pour générer les paires avec `swiparis.pairs` et réentraîner le modèle. Pour enregistrer le prior utilisé par le serving, exécutez `.\.venv\Scripts\python.exe -m swiparis.serving register --prior models/prior_w0_phi.json --models-dir models` ; voir aussi [le guide du serving](docs/serving.md).
