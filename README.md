# SwiParis

Système de recommandation de sorties et de loisirs à Paris. L'utilisateur indique ses disponibilités, son arrondissement et son budget, puis choisit entre des paires d'activités. L'application adapte les comparaisons et propose trois sorties correspondant à ses préférences.

## Membres du groupe

| Membre | Rôle |
| --- | --- |
| Yvan Feugang | Évaluation et protocole expérimental |
| Jeromsan Judes Ramesh | Collecte et préparation des données |
| Merwane Kheloufi | Modèle de recommandation et sélection des paires |
| Martin Duverneuil | Entraînement, baselines et versionnement du modèle |
| Amine Zaidi | API FastAPI et interface Streamlit |

## Objectif

- Filtrer les sorties selon la date, l'heure, l'arrondissement, la catégorie et le budget.
- Apprendre les préférences de la séance à partir de choix entre deux activités.
- Choisir les comparaisons suivantes selon les réponses déjà données.
- Classer les activités éligibles et afficher trois recommandations avec une carte.

Le moteur utilise un modèle de préférence Bradley–Terry et un profil mis à jour pendant la séance. Le prior entraîné est déjà enregistré dans le dossier models : aucun entraînement n'est nécessaire pour lancer la démo.

## Données

| Source | Contenu |
| --- | ---: |
| Foursquare | 816 utilisateurs, 31 694 visites et 6 554 lieux |
| Paris Data | 1 778 annonces de sorties |

Les sources sont réunies dans une taxonomie de dix catégories. Les données préparées sont versionnées dans le dépôt : `data/processed/common.sqlite` (base commune, 16 Mo), `data/processed/pairs.csv` et `data/processed/splits.csv` (paires et découpage pour l'entraînement et l'évaluation). Un clone suffit pour lancer l'application, les tests et l'entraînement. Les données brutes (`data/raw/`) et le journal du pilote (`data/pilot/`, réponses des participants) ne sont pas versionnés.

Le catalogue fourni est un instantané des annonces du 18 novembre au 18 décembre 2026. Pour obtenir des annonces plus récentes, il faut réimporter Paris Data. La préparation est décrite dans [le guide de l'équipe](docs/team_handoff.md), [Foursquare](docs/foursquare.md) et [Paris Data](docs/paris_data.md).

## Installation

Prérequis : Python 3.10 ou plus récent. Sous Windows PowerShell :

~~~powershell
git clone https://github.com/Jeromsan/SwiParis.git
cd SwiParis
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[systems,ml]" tzdata
~~~

La base est déjà dans `data/processed/`. Sur macOS ou Linux, créer le venv avec `python3 -m venv .venv` et installer avec `.venv/bin/python -m pip install -e ".[systems,ml]"`. Si Python indique `No module named 'swiparis'` (vu sur Mac avec Python 3.13), faire `export PYTHONPATH=$PWD/src` dans chaque terminal.

## Lancer l'application

Ouvrir deux terminaux à la racine du projet.

Terminal 1, API :

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn swiparis.api:app --host 127.0.0.1 --port 8000
~~~

Terminal 2, interface :

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run src/swiparis/streamlit_app.py
~~~

Ouvrir http://localhost:8501. L'interface demande le consentement avant d'enregistrer les choix dans une base SQLite locale séparée. La carte OpenStreetMap fonctionne sans clé ; une clé Google Maps est facultative.

## Tester

Installer les dépendances des modules d'entraînement, puis lancer les tests :

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

Vérifier aussi le parcours complet avec la base et le modèle enregistrés :

~~~powershell
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
~~~

## Sélection, personas et réglages

- **Ma sélection** (menu de gauche) : « Garder » enregistre vraiment la sortie, pendant les choix ou sur les recommandations (on peut en cocher plusieurs). La page liste les sorties gardées par jour ; on peut en retirer (confirmation pour un lot, puis « Annuler ») et les exporter vers un agenda (`.ics`) ou un tableau (`.csv`). La sélection est liée à l'adresse de la page (`?carnet=…`).
- **Voir l'annonce** : un clic ; l'ouverture est enregistrée puis l'annonce s'ouvre (`/go`).
- **Personas** : après les choix, l'application affiche les deux personas auxquelles les réponses ressemblent (groupes d'habitudes trouvés dans les données Foursquare, voir [docs/personas.md](docs/personas.md)). Dès deux sorties gardées, « Ma sélection » sert de point de départ.
- **Réglages sans toucher au code** : `SWIPARIS_SESSION_ETA` (pas de mise à jour de la séance, défaut 0,5) et `SWIPARIS_PERSONAS` (`off` pour les désactiver), avant de lancer l'API. Pourquoi 0,5 : « Réglage de la séance » dans [docs/modele.md](docs/modele.md).
- **Choisir η avec le pilote** : lancer l'API avec `SWIPARIS_SESSION_ETA_AB="0.2,0.5"` ; chaque séance reçoit une des deux valeurs. Ensuite, `python -m swiparis.ml.pilot_eta` compare les réglages sur les vraies réponses (recommandations ouvertes et gardées, avec intervalle de confiance).

## Architecture

~~~text
SwiParis/
├── config/                 # Taxonomie commune
├── data/processed/         # Base commune, paires et découpage (versionnés)
├── data/pilot/             # Journal du pilote, local et non versionné
├── docs/                   # Données, modèle, évaluation et personas
├── models/                 # Prior entraîné et personas, versionnés
├── scripts/                # Vérification du parcours API
├── src/swiparis/
│   ├── api.py              # API FastAPI
│   ├── streamlit_app.py    # Interface utilisateur
│   ├── ml/                 # Profil, personas, politique de séance (policy.py), évaluation
│   └── serving.py          # Chargement du modèle
├── tests/                  # Tests automatisés
├── pyproject.toml
└── README.md
~~~

## Dépôt

GitHub : https://github.com/Jeromsan/SwiParis
