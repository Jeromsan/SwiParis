# Catalogue Paris Data

Source : [Que Faire à Paris ? — Évènements et activités](https://opendata.paris.fr/explore/dataset/que-faire-a-paris-/).
Import hors ligne du fichier JSON fourni le 18/11/2026 : 3 241 objets, 17 456 979
octets. Les noms des champs ont été vérifiés dans les fichiers réels ; aucun
schéma Foursquare n'est appliqué à ces annonces. La base Foursquare reste séparée.

## Relancer dans PowerShell

Depuis le dossier `SwiParis`, avec Python 3.10+ :

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\swiparis-paris-import.exe --source data/raw/paris_data/que-faire-a-paris-.json --output data/processed/paris_data.sqlite --report outputs/paris_data_report.json --as-of 2026-11-18T12:30:00+01:00 --days 30
```

Sur cette machine, `py -3` n'est pas configuré ; l'interpréteur utilisé est
`C:\Program Files\Odoo 19.0.20260106\python\python.exe`. On peut l'utiliser pour
créer le venv, ou lancer le module sans installation et sans dépendance externe :

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:\Program Files\Odoo 19.0.20260106\python\python.exe' -m swiparis.import_paris_data --source data/raw/paris_data/que-faire-a-paris-.json --output data/processed/paris_data.sqlite --report outputs/paris_data_report.json --as-of 2026-11-18T12:30:00+01:00
```

Les chemins existants sont refusés : choisir d'autres noms pour répéter l'import.
Sans `--as-of`, l'instant UTC courant est utilisé. Un import avec une autre date
produit un autre snapshot. La fenêtre est de 30 × 24 heures, pas 30 journées
civiles : ici `[2026-11-18T11:30:00Z, 2026-12-18T11:30:00Z)`.

## Règles de nettoyage

- Texte : suppression des balises HTML, scripts et styles, décodage des entités,
  espaces normalisés, Unicode NFC. Titres, descriptions, chapeaux et textes de
  prix sont conservés ; les descriptions absentes sont signalées.
- Identité : `id` est la clé de l'annonce ; `event_id` source est conservé
  séparément. La clé du catalogue porte le préfixe `paris_data:`.
- Géographie : uniquement des lieux physiques, avec coordonnées finies non
  nulles, une adresse Paris (code 75001–75020/75116 ou ville Paris) et le même
  rectangle que Foursquare : latitude [48.815, 48.903], longitude [2.224, 2.470].
  Le rectangle n'est pas une frontière administrative. Les coordonnées seules
  ne suffisent pas à retenir une adresse de banlieue. L'arrondissement provient
  uniquement d'un code postal reconnu, sinon il reste inconnu.
- Multi-lieux : tous les lieux parisiens valides sont conservés et dédoublonnés.
  Les séances n'étant pas associées à un lieu dans l'export, une annonce ayant
  plusieurs lieux reçoit `multi_location_schedule_unassigned`. Une future route
  ne devra pas supposer que chaque séance existe à chaque adresse.
- Calendrier : le champ `occurrences` fournit des couples ISO
  `début_fin` séparés par `;`. Les couples invalides, inversés ou dépassant 24 h
  sont écartés, les couples répétés dédoublonnés. S'il manque, `date_start` et
  `date_end` ne servent de séance que pour un intervalle positif <=24 h. Une
  période d'exposition sur plusieurs semaines ne devient jamais une ouverture
  continue. Aucune récurrence ni heure n'est déduite du texte libre.
- Fenêtre : une séance est retenue si `début < fin_fenêtre` et
  `fin > début_fenêtre`, sans tronquer son horaire réel. Une annonce sans séance
  retenue est exclue. Les horaires source et leurs conversions UTC sont stockés.
- Fuseaux : les décalages ISO explicites de la source sont préservés. Le fichier
  contient des incohérences apparentes entre certaines séances, les enveloppes
  `date_start/date_end` et les textes horaires (notamment des offsets +02:00 en
  hiver). Aucune correction arbitraire n'est appliquée. L'import ne certifie pas
  leur conformité à Europe/Paris ; la description horaire et le lien restent
  accessibles, et un avertissement figure au rapport. Les horaires doivent être
  confirmés sur les annonces pour une démonstration en conditions réelles.
- Catégories : les valeurs réelles de `qfap_tags` sont séparées par `;` puis
  conservées dans `event_tags`. La table de correspondance versionnée
  `config/paris_data_categories.json` produit des catégories d'activité. Les
  tags thématiques (par exemple Enfants) ne sont pas transformés arbitrairement
  en type d'activité ; faute de correspondance, la catégorie vaut `autre`.
- Prix : `gratuit` -> `free`, `payant` -> `paid`, `gratuit sous condition` ->
  `conditional_free`. Seule la gratuité inconditionnelle donne un montant connu
  de 0 €. Un tarif payant ou conditionnel reste numériquement inconnu : les
  fourchettes et réductions restent dans `price_detail`, nettoyé. Un montant
  inconnu ne doit pas être considéré comme 0 dans les futurs filtres de budget.
- Doublons : pour un ID ou une URL canonique répétés, la version la plus récemment
  mise à jour gagne (première ligne en cas d'égalité). Puis sont fusionnés les
  doublons de contenu ayant les mêmes titre, description, lieux, séances dans la
  fenêtre et prix. Des séances différentes restent des annonces différentes.
  Chaque ligne écartée conserve son motif et, le cas échéant, l'ID retenu.

## Versionnement et sorties

L'original reste intact. Une copie figée est créée dans
`data/raw/paris_data/snapshots/<SHA-256>.json`, avec vérification si elle existe.
Les données brutes, les bases et les rapports sont ignorés par Git.

Le snapshot est identifié à partir de l'instant de référence et des empreintes
de la source, du mapping, du schéma, de la fenêtre et de la version de nettoyage.
Le manifeste conserve ces paramètres. Le schéma SQL est versionné dans
`src/swiparis/paris_data_schema.sql` (`PRAGMA user_version=1`).

| Sortie | Contenu |
| --- | --- |
| `data/processed/paris_data.sqlite` | Catalogue versionné |
| `data/processed/paris_data.jsonl` | Même catalogue nettoyé, un objet par ligne |
| `outputs/paris_data_report.json` | Manifeste, volumes, qualités, motifs et lignes rejetées |

Tables : `snapshots`, `events`, `event_locations`, `event_occurrences`,
`event_categories`, `event_tags`, `import_rejections`. Les séances et les lieux
référencent l'activité ; les catégories et tags sont multivalués. Les URLs
d'origine, de réservation et d'image, ainsi que les crédits d'image, sont gardés.
Le fichier fourni est assez petit pour être chargé en mémoire (17,5 Mo).

## Résultats du snapshot du 18/11/2026

| Étape | Annonces |
| --- | ---: |
| Entrée | 3 241 |
| ID, titre et URL valides et uniques | 3 241 |
| Au moins un lieu Paris exploitable | 2 970 |
| Au moins une séance exploitable | 2 373 |
| Séance dans la fenêtre de 30 jours | 1 780 |
| Après dédoublonnage de contenu | **1 778** |

Sortie : **7 909 séances, 1 816 lignes de lieux**, 1 003 annonces gratuites,
717 payantes et 58 gratuites sous conditions. Rejets : 271 sans lieu exploitable
à Paris, 597 sans calendrier exploitable, 593 hors fenêtre, 2 doublons de contenu.
Les motifs sont successifs : une ligne n'est rejetée qu'une fois.

Qualité signalée : 775 montants numériques inconnus, 78 annonces sans type
d'activité mappé, 31 annonces multi-lieux sans correspondance séance/lieu,
9 descriptions absentes et 3 annonces avec arrondissement inconnu. Ces comptes
peuvent se chevaucher et ne sont pas des rejets supplémentaires.

Sur les annonces parvenues à l'étape calendrier : 59 séances répétées supprimées,
478 intervalles dépassant 24 h non utilisés comme séances, et 1 859 séances
invalides (fin non postérieure au début) écartées. Ces comptes portent sur les
séances, et non sur les annonces : une annonce peut rester si elle possède
d'autres séances valides. Certaines fins à minuit sont indiquées le même jour que
le début ; l'import ne devine ni une fin le lendemain ni une durée manquante.

Tests : HTML, dates UTC, bornes de fenêtre, absence de calendrier inventé,
géographie, multi-lieux, prix conditionnels, doublons, préférence pour la dernière
version, copie figée, reproductibilité et changement de version avec le mapping.
Le catalogue réel est également vérifié : intégrité SQLite, clés étrangères,
aucune séance hors fenêtre, aucune activité sans lieu/séance, géographie conforme,
absence de balises HTML résiduelles, concordance SQLite/JSONL/rapport et empreinte
de la copie brute identique à l'original. Le README reste une présentation courte
du projet ; les instructions de collecte sont dans ce document.
La génération des paires et le modèle attendent les consignes suivantes.
