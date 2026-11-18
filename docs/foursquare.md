# Contrat de collecte

Version ciblée : **Global-scale Check-in Dataset, TIST2015**, section 3 de
https://sites.google.com/site/yangdingqi/home/foursquare-dataset (consultée le 18/11/2026).
Cette version est distincte du jeu avec réseaux sociaux, section 5.
Le README original lié sur Google Drive ne rendait pas son texte accessible ;
le format a été recoupé avec sa transcription :
https://huggingface.co/datasets/Davidham3/FourSquare_Global/blame/main/README.md.
La page de prévisualisation montre également les horodatages anglais avec `+0000`.

Fichiers UTF-8, TSV sans en-tête :

| Fichier | Colonnes, dans l'ordre |
| --- | --- |
| dataset_TIST2015_Checkins.txt | user_id, venue_id, UTC time, timezone offset (minutes) |
| dataset_TIST2015_POIs.txt | venue_id, latitude, longitude, category name, country code |

Les fichiers réels ont été fournis localement le 18/11/2026 (ignorés par Git).
Les premières lignes confirment les cinq colonnes POI et les quatre colonnes
check-in, avec horodatages UTC anglais `+0000`. Chaque ligne est validée à l'import ;
une source sans ligne valide provoque un arrêt. Les autres lignes invalides sont
comptées, avec dix exemples de numéros de ligne et motifs par fichier.
Les empreintes SHA-256 portent sur les octets sources, y compris les lignes rejetées.

Paris : pays `FR` et rectangle fermé WGS84 latitude [48.815, 48.903], longitude
[2.224, 2.470]. C'est une approximation reproductible de Paris, incluant des zones
limitrophes, pas un découpage administratif. `--bbox S W N E` permet de le changer.
Les catégories sont inventoriées sur les lieux de ce rectangle. Créer ensuite une
liste JSON de libellés de loisirs choisis **exactement** dans cet inventaire
(musées, salles de spectacle, parcs, etc. selon les valeurs observées). Les valeurs
inconnues sont refusées. Aucun filtre par sous-chaîne ni taxonomie supposée.

Le relevé des métadonnées réelles compte 16 132 lieux dans ce périmètre et
372 catégories. La sélection livrée dans `config/foursquare_leisure_categories.json`
contient 153 libellés observés : culture et spectacles, nature et promenades,
activités sportives, jeux, bars, cafés et restaurants (7 604 lieux candidats).
Elle exclut notamment transports, hébergements, bureaux, domiciles et commerces.
C'est une décision de périmètre éditoriale, modifiable dans le JSON. Les deux
libellés source `Caf\u001a` et `Caf\u001a\u001a` sont conservés littéralement,
avec échappement JSON, sans tenter de réparer l'encodage d'origine.

Ordre des étapes : validation, jointure aux lieux connus, géographie, catégories,
dédoublonnage `(user_id, venue_id, utc_time)`, utilisateurs avec au moins 10 visites.
Deux visites au même lieu à des instants différents restent distinctes. En cas de
doublon de métadonnées, la première ligne valide d'un identifiant lieu gagne ;
leur nombre est rapporté. Pour des doublons de visite avec offsets divergents,
le plus petit offset est conservé ; l'instant est déjà UTC et n'est pas décalé.

Le rapport distingue `venues` (catalogue de l'étape) et `visited_venues` (lieux
effectivement visités). Les volumes d'utilisateurs et lieux en entrée concernent
les lignes valides : les lignes malformées n'ont pas d'identifiants fiables.
Après dédoublonnage, seuls les lieux visités sont comptés ; la sortie élimine les
lieux devenus orphelins. La période est calculée avant filtrage et en sortie.
Les alertes de taille sont heuristiques : moins de 100 utilisateurs ou 1 000
visites par défaut, configurables, sans réduction automatique du seuil de 10.

SQLite conserve les lieux et les visites UTC ISO 8601, avec clé primaire et index
temporel. Pour tout découpage futur, lire explicitement :
`SELECT * FROM checkins ORDER BY user_id, utc_time, venue_id`.
Une table SQL n'offre aucun ordre implicite ; venue_id départage les égalités.

La lecture est en flux et les insertions par lots (10 000). Les calculs globaux et
les identifiants distincts sont stockés dans un SQLite temporaire sur disque,
supprimé en fin d'exécution. Prévoir de l'espace pour l'ensemble des visites
valides, les index et les tris, donc plusieurs fois la taille brute. Les chemins
de sortie existants sont refusés. L'import vérifie l'intégrité SQLite et les clés
étrangères avant de publier la base finale.

Première exécution complète locale du 18/11/2026 : 33 263 633 lignes de visites
observées, dont 2 horodatages corrompus (lignes 24 918 445 et 26 001 832) rejetés.
Le volume observé diffère du volume annoncé sur la page du dataset ; le rapport
utilise uniquement les comptes réels et les SHA-256 des fichiers fournis.
Après Paris : 89 871 visites, 16 132 lieux, 5 858 utilisateurs ; après loisirs :
41 457 visites, 7 604 lieux, 4 705 utilisateurs ; 17 doublons retirés ; après seuil :
31 694 visites, 6 554 lieux et 816 utilisateurs. Période retenue :
2012-04-03T18:08:01Z à 2013-09-16T22:34:39Z. Aucune alerte de taille.

Le premier calcul combiné des distincts s'est avéré lent à l'échelle mondiale.
Les comptes sont désormais séparés pour utiliser l'index des lieux ; lorsque
tous les lieux référencés existent (vérifié par anti-jointure), l'étape « lieux
connus » réutilise les volumes globaux. Le cache SQLite principal est borné à
128 Mio. Cette exécution a été finalisée depuis le chargement temporaire déjà
validé, avec revérification des sources, puis suppression de la base de travail.
