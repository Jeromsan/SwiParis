# Données et démo — guide de l'équipe

## Livrables et responsabilités

- Une seule taxonomie : `config/common_taxonomy.json`, version 1, **10 catégories**,
  correspondance explicite pour les **153 libellés Foursquare** et 28 tags d'activité
  Paris Data. Les tags de public/thème non mappés restent sans interprétation.
- Base partagée : `data/processed/common.sqlite`. Aucune modification des deux
  bases sources. Les données ne sont pas envoyées sur Git.
- Rapport détaillé : `outputs/common_report.json`. Tableau à intégrer au rapport :
  [data_summary.md](data_summary.md).
- Schéma versionné des logs : `src/swiparis/logs_schema.sql`. Base pilote séparée :
  `data/pilot/interactions.sqlite`, ignorée par Git.
- Carte Google de démonstration : lieux Paris Data uniquement, sans collecte
  Places, sans notes Google ajoutées au modèle, sans stockage de données Google.
- Le modèle, les splits, les métriques et la sélection active restent aux autres
  membres. L'ordre d'affichage du catalogue de démo est alphabétique, pas personnalisé.
- Yelp et Kaggle ne sont pas traités dans cette livraison.

## Installation depuis le dépôt

Python **3.10+**, testé avec **3.12.3 sur Windows**. La préparation du catalogue
et la démo historique restent sans dépendance d'exécution externe. Les outils de
construction sont figés dans `pyproject.toml` : `setuptools==69.2.0` et
`wheel==0.43.0`. L'API FastAPI et l'interface Streamlit sont optionnelles et
installées séparément avec `.[systems]`.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sur le poste de Jeromsan, remplacer `py -3` par
`& 'C:\Program Files\Odoo 19.0.20260106\python\python.exe'` si nécessaire.
Les environnements virtuels ne sont pas partagés. Pour éviter toute politique
d'activation PowerShell, les commandes utilisent directement l'exécutable du venv.

## Préparer les données une fois

Avec `data/processed/foursquare.sqlite` et `data/processed/paris_data.sqlite`
déjà produits par les imports :

```powershell
.\.venv\Scripts\swiparis-prepare.exe
```

Cette commande génère `common.sqlite`, `outputs/common_report.json` et le rapport
Markdown. Pour régénérer ces seules sorties : ajouter `--force`. Les chemins
d'entrée et de sortie identiques sont refusés. Les imports bruts sont documentés
dans [foursquare.md](foursquare.md) et [paris_data.md](paris_data.md).

Arrêter la démo avant `--force`, puis la relancer : une session de logs garde sa
version de catalogue et ne doit pas traverser un remplacement des données.

La préparation prend quelques secondes. **L'import brut mondial Foursquare est
plus long et ne fait pas partie de la préparation rapide d'une machine de démo.**
Partager le ZIP décrit plus bas pour éviter de faire réimporter 2,4 Go à chacun.

## Contrat SQLite pour Core / Ops / PL

| Objet | Utilisation |
| --- | --- |
| `items` | Tous les lieux historiques et toutes les annonces, avec `source` |
| `item_categories` | Catégories communes, potentiellement multiples pour Paris Data |
| `categories` | Codes stables et libellés d'affichage |
| `visits` | Visites Foursquare uniquement, sans paires ni labels négatifs inventés |
| `training_visits` | Vue visites + catégorie ; une catégorie par lieu Foursquare |
| `demo_items` | Vue des seules annonces Paris Data |
| `locations` | Coordonnées et arrondissement, plusieurs lieux possibles |
| `occurrences` | Séances UTC Paris Data ; aucune disponibilité inventée pour Foursquare |
| `metadata` | Manifeste, empreintes des sources, version du catalogue et taxonomie |

Identifiants : `foursquare:<venue_id>` et `paris_data:<id>`. Les utilisateurs sont
préfixés `foursquare:` ; `source_user_id` garde l'ID original. **Ne jamais joindre
des objets Foursquare et Paris Data par ID**, même s'ils ont le même suffixe.
La taxonomie aligne les types d'activités, pas les identités des lieux.

```python
import sqlite3
with sqlite3.connect("data/processed/common.sqlite") as db:
    histories = db.execute("""
        SELECT user_id, item_id, utc_time, category
        FROM training_visits ORDER BY user_id, utc_time, item_id
    """).fetchall()
```

Les lieux Foursquare n'ont ni titre descriptif d'événement, ni description riche :
`title` reprend le libellé de catégorie, avec le flag `category_only_text`.
Ne pas interpréter ce champ comme un nom de lieu réel. Les sources et leurs
limites restent distinctes même dans une base commune.

Les musées, expositions, cinémas et lectures sont regroupés dans `culture` ; les
concerts et théâtres dans `spectacle`. La restauration inclut cafés, desserts et
dégustations ; les bars/pubs vont dans `vie_nocturne`. Ce sont des choix éditoriaux
explicites, modifiables dans le JSON. Chaque changement modifie la version des
données et nécessite une nouvelle préparation/validation. Les tags source sont
conservés dans `source_category_label`.

Paris Data est multivalué : le rapport donne les affectations de catégories,
ainsi qu'un nombre fractionnaire d'annonces (1/n par catégorie). Les parts de
visites Foursquare, elles, sont additives. Les ateliers ont zéro visite dans nos
historiques : cela reste visible, sans fabriquer de labels pour couvrir le manque.

## Lancer la liste et la carte

```powershell
# Facultatif, clé navigateur uniquement : ne pas committer la valeur.
$env:GOOGLE_MAPS_API_KEY = 'votre-cle'
.\.venv\Scripts\swiparis-demo.exe
```

Ouvrir http://127.0.0.1:8000. Le serveur écoute uniquement sur localhost.
La page affiche les dates du snapshot. Le filtre temporel demande qu'une séance
**tienne entièrement** dans le créneau choisi. Avec un arrondissement, les
annonces multi-lieux sans association séance/adresse sont exclues. Un budget
exclut les prix numériques inconnus et les gratuités conditionnelles.

Sans clé ou sans réseau, la liste et les liens Google Maps restent disponibles.
La carte Google ne charge qu'après clic sur « Afficher la carte Google » et
montre les lieux de la page courante, jusqu'à 60 annonces. Les horaires source
suspects et les tarifs inconnus restent signalés : voir le rapport Paris Data.

Activer **Maps JavaScript API**, utiliser une clé navigateur restreinte aux
référents `http://127.0.0.1:8000/*` et `http://localhost:8000/*`, et à cette API.
Une clé navigateur est visible dans le navigateur par conception ; elle n'est
pas une clé de serveur secrète. Elle est lue depuis l'environnement, jamais
copiée dans les fichiers ou le ZIP. `.env.example` est un exemple, pas un fichier
chargé automatiquement. Instructions officielles :
[clé et restrictions](https://developers.google.com/maps/documentation/javascript/get-api-key),
[chargement de Maps JS](https://developers.google.com/maps/documentation/javascript/load-maps-js-api).
`GOOGLE_MAPS_MAP_ID` est facultatif ; par défaut le Map ID de démonstration Google
est utilisé. Aucun appel à Places API, scraping ou enregistrement de fiches
Google n'est nécessaire pour afficher les coordonnées déjà connues.

## API de logs pour Systems / Core

Le service FastAPI est `src/swiparis/api.py`. Installer les dépendances système
puis préparer `data/processed/common.sqlite` comme ci-dessus :

```powershell
py -3 -m pip install -e ".[systems]"
$env:GOOGLE_MAPS_API_KEY = 'votre-cle-navigateur' # facultatif
swiparis-api
```

Dans une autre fenêtre PowerShell :

```powershell
swiparis-streamlit
```

Ouvrir http://localhost:8501. Le serveur API écoute seulement sur
`127.0.0.1:8000`. Le parcours demande un consentement explicite avant de créer
une session pilote, filtre les annonces Paris Data par créneau, arrondissement,
catégorie et budget, propose les comparaisons, puis affiche jusqu'à dix
recommandations et leur carte. Un prix inconnu ne satisfait jamais un filtre de
budget. La clé Google est lue dans l'environnement et n'est ni stockée ni
envoyée au catalogue ; sans clé, la liste reste disponible.

`GET /next_pair?session_id=...` calcule la prochaine paire sans écrire dans la
base. Le front-end l'enregistre ensuite via `POST /api/comparisons` avec un
`event_id` stable, ce qui persiste les deux items et leur ordre gauche/droite.
`POST /log_interaction` enregistre `left`, `right` ou `skip`. `POST
/recommendations` classe les activités non montrées et enregistre les impressions.
`POST /api/item_action` enregistre `open` et `save`. `POST /api/sessions`
nécessite le consentement et fixe les contraintes de la séance.

Le scorer initial utilise la fréquence lissée des catégories Foursquare comme
prior et met à jour le profil Bradley-Terry après chaque choix. Pour charger un
prior compatible exporté au format NumPy, définir `SWIPARIS_PRIOR_W0` vers un
vecteur fini de même dimension que `FeatureSpace` et, facultativement,
`SWIPARIS_MODEL_VERSION`. Le modèle MLOps présent dans la branche exporte des
vecteurs Foursquare projetés en 32 dimensions, mais pas encore la projection
nécessaire pour encoder Paris Data ; il ne peut donc pas être branché directement
sur ce service. L'intégration de ce bundle attend un export commun convenu avec
Martin et Merwane.

La réponse HTTP contient `X-Process-Time-Ms` pour mesurer la durée de traitement
sur le catalogue réel. Il faut mesurer le parcours avec la base distribuée avant
de rapporter la cible de latence d'environ deux secondes.

`GET /health` donne la version chargée. `GET /api/catalog` fournit le catalogue
avec `items`, `total`, `categories` et `catalog_version`.
Paramètres : `start_utc`, `end_utc` (ISO avec offset, tous deux ou aucun),
`arrondissement` (1–20), `budget_eur`, `category`, `limit` (1–500), `offset`.

Toutes les écritures sont JSON, avec un `event_id` unique **par événement réel**.
Réutiliser le même `event_id` et le même corps lors d'une retransmission : réponse
`replayed=true`, sans insertion supplémentaire. Un ID réutilisé avec un autre
contenu, une autre opération ou une seconde réponse à une comparaison retourne
HTTP 409. Données invalides : 400 ; route inconnue : 404.

| Route POST | Champs en plus de `event_id` |
| --- | --- |
| `/api/sessions` | `consent: true`, `constraints: {}`, `catalog_version`, `model_version`, `data_origin` (`pilot` ou `test`) |
| `/api/comparisons` | `session_id`, `item_a`, `item_b` |
| `/log_interaction` | `session_id`, `comparison_id`, `choice` (`left`, `right`, `skip`) |
| `/api/recommendations_shown` | `session_id`, `item_ids` ordonnés, 1–10 IDs uniques |
| `/api/item_action` | `session_id`, `impression_id`, `action` (`open` ou `save`) |

`constraints` accepte les mêmes contraintes que le catalogue, sans pagination.
Le modèle choisit la paire ; le logger randomise seulement gauche/droite **une
fois** et renvoie l'ordre à afficher. Utiliser cet ordre, y compris lors d'une
retransmission. Les réponses renvoient l'ID choisi, ou null pour un passage.
L'arrêt après 5 réponses/8 offres et l'acquisition sont gérés par le service
avec `swiparis.ml.selection.next_pair`.
Les recommandations doivent être éligibles et absentes des comparaisons offertes.

Exemple PowerShell, avec le serveur lancé dans une autre fenêtre :

```powershell
$base = 'http://127.0.0.1:8000'
function Post-Json($route, $body) {
    Invoke-RestMethod -Method Post -Uri ($base + $route) -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 8)
}
$catalog = Invoke-RestMethod "$base/api/catalog?limit=5"
$session = Post-Json '/api/sessions' @{
    event_id = [guid]::NewGuid().ToString(); consent = $true; constraints = @{}
    catalog_version = $catalog.catalog_version; model_version = 'modele-votre-version'; data_origin = 'pilot'
}
$pair = Post-Json '/api/comparisons' @{
    event_id = [guid]::NewGuid().ToString(); session_id = $session.session_id
    item_a = $catalog.items[0].item_id; item_b = $catalog.items[1].item_id
}
# Afficher pair.left_item_id / pair.right_item_id, puis enregistrer le vrai choix.
```

Les horodatages sont attribués en UTC par le serveur. Les versions et contraintes
sont fixées dans `sessions` et récupérables par jointure. Aucun nom, email ou IP
n'est stocké. La liste/carte seule ne crée pas de session ni de feedback.
Un élément affiché mais non ouvert/sauvegardé reste sans label négatif.
Les actions appartiennent à une impression et une session existantes.

Les autres services peuvent appeler directement la même couche, sans ce serveur :

```python
from swiparis.catalog import Catalog
from swiparis.logging_store import LogStore
catalog = Catalog("data/processed/common.sqlite")
store = LogStore("data/pilot/interactions.sqlite", catalog)
# La route FastAPI appelle directement store.answer(corps_json).
```

`swiparis-demo` reste l'adaptateur léger pour la liste/carte et le smoke test.
`swiparis-api` est le service FastAPI pour les comparaisons et recommandations.
Les logs de tests sont temporaires et supprimés ; les logs pilotes ne sont jamais
réinjectés automatiquement dans l'entraînement/test. Le catalogue est lu seul.

## Paquet hors ligne et validation rapide

Construire le wheel avec les outils figés, puis le ZIP :

```powershell
.\.venv\Scripts\python.exe -m pip wheel --no-deps . --wheel-dir dist
.\.venv\Scripts\swiparis-bundle.exe --wheel dist/swiparis-0.2.0-py3-none-any.whl
```

`dist/swiparis-demo.zip` contient uniquement le wheel, la base commune, les
scripts de démarrage et un manifeste SHA-256. Il exclut les fichiers bruts,
les logs, `.env`, la clé Google et l'environnement virtuel. Le partager avec
l'équipe en dehors de Git. Il n'est pas publié ni envoyé automatiquement.

Sur le poste destinataire, avec Python installé, extraire le ZIP puis :

```powershell
py -3 start.py --check-only   # Empreintes, installation hors ligne, tests HTTP + SQLite
py -3 start.py                # Même préparation, puis serveur local
```

Avec un interpréteur non reconnu par `py`, utiliser
`& 'C:\chemin\python.exe' start.py`. Ce lanceur Python évite le blocage des
scripts `.ps1` constaté sur le poste de Jeromsan, sans modifier la politique
d'exécution de Windows. Il fonctionne également via `python3 start.py` sur les
autres systèmes disposant du module venv.

Validation automatisée dans un venv neuf, sans PYTHONPATH ni packages utilisateur :

```powershell
.\.venv\Scripts\python.exe scripts/check_clean_install.py --wheel dist/swiparis-0.2.0-py3-none-any.whl
```

Résultat : `outputs/clean_install_report.json` avec durée mesurée, preuve du chemin
d'import dans le venv et résultats du parcours HTTP. Choisir un nouveau
`--work-dir`/`--report` pour répéter le contrôle. Les tests n'utilisent pas la clé
Google et ne la valident pas. **Un venv neuf sur ce poste ne remplace pas le test
final par un coéquipier sur un autre ordinateur.** Le téléchargement/installation
de Python et l'import mondial brut sont des prérequis, pas inclus dans les 10 min.

### Vérification réalisée le 19/11/2026

Les **13 tests passent**. Le wheel 0.2.0 a été installé sans téléchargement dans
un venv neuf, sans PYTHONPATH ni packages utilisateur, sur le poste Windows :
**56,762 secondes** au total, dont 2,163 secondes pour le parcours HTTP réel.
L'import du paquet provient bien du `site-packages` du nouvel environnement.
Le parcours vérifie les sessions, les choix, les retransmissions, les conflits,
les recommandations affichées, les ouvertures/sauvegardes et l'intégrité SQLite.
Rapport machine : `outputs/clean_install_report.json`.

Le **ZIP final avec lanceur Python** a ensuite été extrait et testé dans un
autre environnement vierge du même poste : **21,366 secondes**, hors extraction,
avec installation hors ligne, empreintes vérifiées et parcours HTTP complet.
Preuve : `outputs/bundle-test-python/bundle_check.json` ; import depuis le
`site-packages` du venv du ZIP. Le premier essai du lanceur PowerShell avait été
bloqué par la politique d'exécution Windows ; le lanceur livré est `start.py`.

La carte Google réelle reste à vérifier avec une clé autorisée ; la page HTML,
la configuration de carte et les routes de catalogue/logs sont vérifiées. Le dernier contrôle à faire
par l'équipe est de lancer le ZIP sur un autre ordinateur disposant de Python.
