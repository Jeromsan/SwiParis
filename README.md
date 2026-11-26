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

Les sources sont réunies dans une taxonomie de dix catégories. La base préparée n'est pas versionnée dans Git. Depuis un clone, récupérer common.sqlite dans le fichier swiparis-demo.zip fourni par l'équipe et le placer dans data/processed/common.sqlite. Ne pas créer une base SQLite vide.

Le catalogue fourni est un instantané des annonces du 18 novembre au 18 décembre 2026. Pour obtenir des annonces plus récentes, il faut réimporter Paris Data. La préparation est décrite dans [le guide de l'équipe](docs/team_handoff.md), [Foursquare](docs/foursquare.md) et [Paris Data](docs/paris_data.md).

## Installation

Prérequis : Python 3.10 ou plus récent. Sous Windows PowerShell :

~~~powershell
git clone https://github.com/Jeromsan/SwiParis.git
cd SwiParis
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[systems,ml]" tzdata
New-Item -ItemType Directory -Force data/processed
Copy-Item "CHEMIN_VERS_LE_ZIP_EXTRAIT\common.sqlite" data/processed/common.sqlite
~~~

Adapter le chemin de common.sqlite à son emplacement réel. Sur macOS ou Linux, créer le venv avec python3 -m venv .venv, installer avec .venv/bin/python -m pip install -e ".[systems,ml]", puis copier la base au même emplacement.

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

Vérifier le parcours complet avec la vraie base et le modèle enregistré :

~~~powershell
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
~~~

Ce parcours teste la création d'une séance, dix comparaisons, les recommandations et leur journalisation. Le réentraînement est décrit dans [le pipeline ML](docs/ml_pipeline.md) et le chargement du modèle dans [le guide du serving](docs/serving.md).

## Architecture

~~~text
SwiParis/
├── config/                 # Taxonomie commune
├── data/processed/         # Base SQLite locale, non versionnée
├── docs/                   # Données, modèle et évaluation
├── models/                 # Prior entraîné et versionné
├── scripts/                # Vérification du parcours API
├── src/swiparis/
│   ├── api.py              # API FastAPI
│   ├── streamlit_app.py    # Interface utilisateur
│   ├── ml/                 # Profil, sélection et classement
│   └── serving.py          # Chargement du modèle
├── tests/                  # Tests automatisés
├── pyproject.toml
└── README.md
~~~

## Dépôt

GitHub : https://github.com/Jeromsan/SwiParis
