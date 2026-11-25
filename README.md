# SwiParis

SwiParis est un projet étudiant de recommandation de sorties à Paris.

L'application propose des activités deux par deux. L'utilisateur choisit celle
qu'il préfère pour découvrir progressivement des sorties adaptées à ses goûts :
culture, promenades, sport, restaurants et vie nocturne.

L'objectif est de faciliter le choix d'une sortie et de faire découvrir de
nouveaux lieux grâce à une expérience simple et personnalisée.

## Lancer la démo

Python 3.10+ requis. Le catalogue préparé est partagé dans `swiparis-demo.zip`
(à récupérer auprès de l'équipe, les données ne sont pas sur Git). Extraire le
ZIP et, dans son dossier, lancer PowerShell :

```powershell
py -3 start.py
```

Ouvrir **http://127.0.0.1:8000**. Sans clé Google, la liste reste utilisable.
Cette commande lance l'ancienne démo légère du catalogue.

## Parcours comparatif (FastAPI + Streamlit)

Après avoir préparé le catalogue local, installer le groupe système optionnel :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[systems]"
```

Dans une première fenêtre PowerShell, démarrer l'API :

```powershell
# Facultatif : clé navigateur Google Maps restreinte à localhost.
$env:GOOGLE_MAPS_API_KEY = 'votre-cle'
.\.venv\Scripts\swiparis-api.exe
```

Dans une deuxième fenêtre :

```powershell
.\.venv\Scripts\swiparis-streamlit.exe
```

Ouvrir **http://localhost:8501**. La démo demande le consentement du pilote,
filtre les annonces Paris Data, enregistre les comparaisons dans une base SQLite
séparée et affiche les recommandations et la carte. Sans clé Google, les fiches
et leurs liens source restent accessibles. L'API charge le prior Bradley–Terry
versionné dans `models/` et expose sa version dans `/health`. Si aucun prior
versionné n'est trouvé, elle utilise le prior de fréquence par catégorie.
Voir [le guide du serving](docs/serving.md) et [le guide de l'équipe](docs/team_handoff.md).

Pour vérifier le parcours API de bout en bout sur le catalogue partagé :

```powershell
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
```

Développement, préparation des données, clé Google et API de logs :
[guide pour l'équipe](docs/team_handoff.md). Répartition des visites :
[rapport commun](docs/data_summary.md).
