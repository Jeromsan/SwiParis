# SwiParis — collecte Foursquare

Importer TIST2015 vers SQLite, sans dépendance d'exécution externe. Python 3.10+
requis. Depuis ce dossier, dans PowerShell :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sur cette machine, si `py -3` est introuvable, remplacer la première commande par :
`& 'C:\Program Files\Odoo 19.0.20260106\python\python.exe' -m venv .venv`.

Placer les deux sources décompressées dans `data/raw/foursquare/`.
Commencer par l'inventaire des catégories réellement présentes :

```powershell
.\.venv\Scripts\swiparis-import.exe --checkins data/raw/foursquare/dataset_TIST2015_Checkins.txt --pois data/raw/foursquare/dataset_TIST2015_POIs.txt --output data/processed/inventory_unused.sqlite --report outputs/foursquare_inventory.json --inventory-only
```

La liste livrée `config/foursquare_leisure_categories.json` contient 153 catégories
de loisirs réellement observées. Elle peut être modifiée à partir de l'inventaire.
Pour importer sur de nouveaux chemins de sortie :

```powershell
.\.venv\Scripts\swiparis-import.exe --checkins data/raw/foursquare/dataset_TIST2015_Checkins.txt --pois data/raw/foursquare/dataset_TIST2015_POIs.txt --output data/processed/foursquare.sqlite --report outputs/foursquare_report.json --categories config/foursquare_leisure_categories.json
```

Import réel effectué le 18/11/2026 : **31 694 visites, 6 554 lieux, 816 utilisateurs**,
du 03/04/2012 au 16/09/2013. Entrée : 33 263 633 lignes de visites (2 horodatages
corrompus rejetés). Paris : 89 871 visites ; loisirs : 41 457 ; dédoublonnées :
41 440. Le seuil reste 10. Les sorties existantes sont refusées : choisir d'autres
noms pour relancer. [Règles et format](docs/foursquare.md).

Sorties : `data/processed/foursquare.sqlite` (`venues`, `checkins`) et
`outputs/foursquare_report.json` ; catégories observées : `outputs/paris_categories_observed.json`.
Vérifications : trois tests exécutés avec succès (rectangle et pays, doublons,
seuil 9/10, ordre UTC, rejets, catégories inconnues, inventaire et intégrité
SQLite). Exemple : 25 visites valides, 21 après filtres, 19 après dédoublonnage,
10 conservées pour un utilisateur et un lieu. Sorties synthétiques exécutées :
`outputs/example/out.sqlite` et `outputs/example/report.json` (ignorées par Git).
Python trouvé dans l'installation
Odoo pour ces tests ; `py -3` reste à configurer sur cette machine. Base réelle
également contrôlée : intégrité OK, zéro doublon, zéro référence invalide,
géographie conforme, 10 à 283 visites par utilisateur.
