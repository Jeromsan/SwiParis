# SwiParis

Application interactive de recommandation de sorties et loisirs à Paris. En dix choix entre paires d'activités, SwiParis adapte les comparaisons aux sorties déjà choisies et à leurs ressemblances, puis affiche les trois meilleures recommandations sous contraintes de disponibilité, d'arrondissement et de budget. La disponibilité par défaut couvre toute la journée pour permettre un vrai choix parmi les sorties annoncées.

## Équipe

| Membre | Rôle | Responsabilités principales |
| --- | --- | --- |
| **Yvan Feugang** | Project Lead | Protocole d'évaluation, splits 60/20/20, contrôle des fuites, ablation |
| **Jeromsan Judes Ramesh** | Data Engineer | Collecte Foursquare & Paris Data, taxonomie commune, base SQLite et logs |
| **Merwane Kheloufi** | Core ML Engineer | Représentations d'items, modèle Bradley-Terry, sélection active, re-ranking MMR |
| **Martin Duverneuil** | ML Operations Engineer | Génération de paires, baselines, entraînement, versioning du prior (`w0`) et serving |
| **Amine Zaidi** | Systems Engineer | Backend FastAPI, interface Streamlit, intégration du modèle et démo locale |

## Données

Les bases brutes et préparées ne sont pas versionnées sur Git. Le zip local de la version de référence inclut `data/processed/common.sqlite` pour un lancement direct. Depuis un clone Git, récupérer `swiparis-demo.zip` auprès de l'équipe et copier `common.sqlite` dans `data/processed/common.sqlite`. Le catalogue partagé comprend 816 utilisateurs, 31 694 visites Foursquare, 6 554 lieux et 1 778 annonces Paris Data, avec une taxonomie commune à dix catégories dans `config/common_taxonomy.json`.

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

Ouvrir `http://localhost:8501`. La démo demande le consentement du pilote, vérifie qu'au moins six sorties correspondent aux filtres, journalise les comparaisons dans une base SQLite séparée, puis affiche les trois recommandations et leur carte. Les réponses récentes favorisent les catégories et sorties similaires aux choix précédents ; une comparaison sur quatre peut départager deux sorties déjà choisies. Les sorties explicitement écartées ne sont plus proposées. Sans clé Google Maps, une carte OpenStreetMap interactive s'affiche avec des liens Google Maps sur les repères. Pour utiliser la carte Google intégrée, fournir une clé navigateur restreinte via `GOOGLE_MAPS_API_KEY` dans l'environnement du processus API.

L'API charge le prior Bradley–Terry versionné dans `models/` et expose la version du prior et de la politique de session dans `/health`. Elle réapplique les choix au profil de la séance, utilise un signal de préférence récent pour les paires et le classement, garde les sorties choisies comme candidates et écarte celles explicitement rejetées. Si aucun prior versionné n'est présent, l'API utilise le prior de fréquence par catégorie. Voir [le guide du serving](docs/serving.md), [le guide du modèle](docs/modele.md) et [le guide d'équipe](docs/team_handoff.md).

## Vérification du parcours API

Avec la base partagée disponible localement :

```powershell
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
```

Pour la préparation des données et le rapport de répartition des visites, voir [le guide d'équipe](docs/team_handoff.md) et [le résumé des données](docs/data_summary.md).
