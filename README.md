# SwiParis

Application interactive de recommandation de sorties et loisirs à Paris.
En 5 choix rapides entre paires d'activités (sous contraintes de créneau, d'arrondissement et de budget), SwiParis apprend le profil de la session et recommande jusqu'à 10 sorties adaptées.

## Équipe

| Membre | Rôle | Responsabilités principales |
| --- | --- | --- |
| **Yvan Feugang** | Project Lead | Protocole d'évaluation, splits 60/20/20, contrôle des fuites, ablation |
| **Jeromsan Judes Ramesh** | Data Engineer | Collecte Foursquare & Paris Data, taxonomie commune, base SQLite et logs |
| **Merwane Kheloufi** | Core ML Engineer | Représentations d'items, modèle Bradley-Terry, sélection active, re-ranking MMR |
| **Martin Duverneuil** | ML Operations Engineer | Génération de paires, baselines, entraînement, versioning du prior (`w0`) et serving |
| **Amine Zaidi** | Systems Engineer | Backend FastAPI, interface Streamlit, intégration du modèle et démo locale |

## Données

Les bases brutes et préparées ne sont pas versionnées sur Git.
La base partagée `data/processed/common.sqlite` (816 utilisateurs, 31 694 visites Foursquare, 6 554 lieux, 1 778 annonces Paris Data, taxonomie commune à 10 catégories dans `config/common_taxonomy.json`) est distribuée dans l'archive `swiparis-demo.zip` auprès de l'équipe.

1. Récupérer `swiparis-demo.zip` auprès de l'équipe.
2. Copier `common.sqlite` dans `data/processed/common.sqlite`.

Pour régénérer les bases depuis les fichiers bruts, voir `docs/team_handoff.md`, `docs/foursquare.md` et `docs/paris_data.md`.

## Installation rapide (démo, < 10 min)

Prérequis : **Python 3.10+** (testé sous Python 3.12).

```powershell
git clone [https://github.com/Jeromsan/SwiParis.git](https://github.com/Jeromsan/SwiParis.git)
cd SwiParis
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -r requirements.txt