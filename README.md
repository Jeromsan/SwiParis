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
La démo présente le catalogue ; elle n'inclut pas encore de modèle entraîné.

Développement, préparation des données, clé Google et API de logs :
[guide pour l'équipe](docs/team_handoff.md). Répartition des visites :
[rapport commun](docs/data_summary.md).
