# Prior versionné et `predict` (`swiparis.serving`)

Ne dépend que de numpy et de `swiparis.ml` (pas de torch). Remplace l'usage du prior « par fréquence
de catégorie » de l'API par le prior entraîné (`models/prior_w0_phi.json`, Merwane).

## Enregistrer le prior (une fois, dans le dépôt, après la fusion de la PR de Merwane)

```powershell
.\.venv\Scripts\python.exe -m swiparis.serving register --prior models/prior_w0_phi.json --models-dir models
.\.venv\Scripts\python.exe -m swiparis.serving show --models-dir models
git add models/prior-* models/latest_prior.txt
```

Crée `models/prior-20261120-ac6b83d6/` (nom déterministe : date de création du prior + 8 premiers caractères
de son SHA-256) avec `prior.json`, `w0.npy` et `metadata.json` (version, SHA-256, lambda, poids de popularité,
métriques, hash des paires, commit git, versions des bibliothèques). La commande est idempotente.
`models/latest_prior.txt` désigne la dernière version (le `latest.txt` de `swiparis.registry` reste à part).

## Utiliser le modèle

```python
from swiparis.serving import load_model
model = load_model("models")          # dernière version ; refuse un prior dont le SHA-256 a changé
phi = model.encode(items)             # items : liste de swiparis.ml.features.Item
top10 = model.predict(phi, ids, answers=[(chosen_id, rejected_id), ...], exclude=shown_ids, k=10)
pair = model.next_pair(phi, ids, answers=answers, shown_pairs=shown, rng=0)
```

`phi` et `ids` sont les activités **éligibles** (contraintes déjà appliquées). L'espace de features vient du prior
lui-même (`model.space`, 28 dimensions dont la popularité) : ne pas en construire un autre.

## Brancher l'API (Amine)

```powershell
git apply integration/api_use_serving.patch     # modifie src/swiparis/api.py
.\.venv\Scripts\python.exe scripts/check_api_e2e.py --catalog data/processed/common.sqlite --models-dir models
```

Le patch fait charger `models/` à l'API (variable `SWIPARIS_MODELS_DIR`, défaut `models`). Sans prior enregistré, l'API
retombe sur l'ancien prior par catégorie. `check_api_e2e.py` produit la preuve pour Q5 : `model_version` dans `/health`,
dans la session et dans les recommandations, lignes écrites dans le journal, latences. Nécessite `fastapi` et `httpx`.

## Limite à connaître

Les annonces Paris Data n'ont pas de popularité (valeur 0) : le prior les classe d'après leur catégorie, leur prix et
leur zone. Sans réponse, son top 10 est surtout « culture » et « visite », parce que ce sont les catégories populaires
dans Foursquare.
