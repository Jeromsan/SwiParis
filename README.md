# SwiParis

Application interactive de recommandation de sorties et loisirs à Paris. En cinq choix rapides entre paires d'activités, SwiParis apprend le profil de la session et recommande jusqu'à dix sorties adaptées, sous contraintes de créneau, d'arrondissement et de budget.

## Équipe

| Membre | Rôle | Responsabilités principales |
| --- | --- | --- |
| **Yvan Feugang** | Project Lead | Protocole d'évaluation, splits 60/20/20, contrôle des fuites, ablation |
| **Jeromsan Judes Ramesh** | Data Engineer | Collecte Foursquare & Paris Data, taxonomie commune, base SQLite et logs |
| **Merwane Kheloufi** | Core ML Engineer | Représentations d'items, modèle Bradley-Terry, sélection active, re-ranking MMR |
| **Martin Duverneuil** | ML Operations Engineer | Génération de paires, baselines, entraînement, versioning du prior (`w0`) et serving |
| **Amine Zaidi** | Systems Engineer | Backend FastAPI, interface Streamlit, intégration du modèle et démo locale |

## Données

Les bases brutes et préparées ne sont pas versionnées sur Git. Récupérer `swiparis-demo.zip` auprès de l'équipe et copier `common.sqlite` dans `data/processed/common.sqlite`. Le catalogue partagé comprend 816 utilisateurs, 31 694 visites Foursquare, 6 554 lieux et 1 778 annonces Paris Data, avec une taxonomie commune à dix catégories dans `config/common_taxonomy.json`.

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

Ouvrir `http://localhost:8501`. La démo demande le consentement du pilote, filtre les annonces Paris Data, journalise les comparaisons dans une base SQLite séparée et affiche les recommandations et la carte. Sans clé Google Maps, les fiches et liens source restent accessibles. Pour afficher la carte, fournir une clé navigateur restreinte via la variable d'environnement `GOOGLE_MAPS_API_KEY`.

L'API charge le prior Bradley–Terry versionné dans `models/` et expose la version dans `/health`. Si aucun prior versionné n'est présent, elle utilise le prior de fréquence par catégorie. Voir [le guide du serving](docs/serving.md), [le guide du modèle](docs/modele.md) et [le guide d'équipe](docs/team_handoff.md).

## Vérification du parcours API

Avec la base partagée disponible localement :

```powershell
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
```

Pour la préparation des données et le rapport de répartition des visites, voir [le guide d'équipe](docs/team_handoff.md) et [le résumé des données](docs/data_summary.md).

## Ré-entraînement du modèle

Pour relancer le pipeline MLOps, installez les dépendances avec `.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt`, puis suivez [le guide du pipeline](docs/ml_pipeline.md) pour générer les paires avec `swiparis.pairs` et réentraîner le modèle. Pour enregistrer le prior utilisé par le serving, exécutez `.\.venv\Scripts\python.exe -m swiparis.serving register --prior models/prior_w0_phi.json --models-dir models` ; voir aussi [le guide du serving](docs/serving.md).
