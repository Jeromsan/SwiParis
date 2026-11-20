# Paires, baselines et entraînement (MLOps)

Ces trois modules complètent la collecte : `pairs` (paires de préférence),
`baselines` (aléatoire, popularité) et `train` (prior hors ligne + versioning).
Dépendances : `requirements-ml.txt` (torch n'est requis que pour `train` et `bprmf`).
Base d'entrée : `data/processed/common.sqlite` de Jeromsan (catégories déjà communes, donc pas de `--taxonomy`).
L'ancien SQLite brut `foursquare.sqlite` (tables `venues`, `checkins`) est aussi reconnu.

## 1. Paires : `swiparis.pairs`

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
python -m swiparis.pairs --db data/processed/common.sqlite --output data/processed/pairs.csv `
  --manifest outputs/pairs_manifest.json --seed 42
```

- Une paire par visite : lieu visité = « choisi », lieu **jamais visité par l'utilisateur** = « rejeté ».
- Quatre stratégies de tirage (`negative_type`) : `weak_geo` (voisins géographiques),
  `weak_cat` (même catégorie commune), `weak_time` (lieux populaires à la même tranche
  horaire locale), `weak_pop` (popularité globale). Tirage pondéré par popularité^0,75.
- La popularité n'utilise **que les utilisateurs d'entraînement** (pas de fuite vers val/test).
- Ce sont des **négatifs faibles** : non visité ne veut pas dire pas aimé. À écrire tel quel dans le rapport.
- Split par utilisateur 60/20/20 (seed). Pour imposer le split de l'évaluation : `--splits splits.csv`
  (colonnes `user_id,split`).
- Sans `--taxonomy`, la catégorie Foursquare brute sert de catégorie commune. Les catégories absentes
  de la taxonomie sont listées dans le manifeste (`unmapped_categories`).
- Les sorties existantes sont refusées (`--force` pour écraser). Le manifeste contient seed, SHA-256
  des entrées et de la sortie, comptes par split et par stratégie.

Colonnes : `pair_id, user_id, split, visit_rank, n_visits, timestamp, time_bucket,
chosen_venue_id, rejected_venue_id, negative_type, chosen_category, rejected_category`.
`visit_rank` (0 = première visite de l'utilisateur) permet de séparer profil et cible.

## 2. Baselines : `swiparis.baselines`

```powershell
python -m swiparis.baselines --db data/processed/common.sqlite --pairs data/processed/pairs.csv --output outputs/baselines.json
```

Précision par paires (choisi mieux classé que rejeté) pour l'aléatoire (~0,5) et la popularité lissée
sur les utilisateurs d'entraînement. Ce n'est pas le NDCG@10 / Recall@20 : le classement complet
sur candidats éligibles reste dans le script d'évaluation.

**Limite importante :** les négatifs sont tirés proportionnellement à la popularité, donc la baseline
popularité peut obtenir moins de 0,5 sur val/test. Ne pas citer ce chiffre comme preuve qu'on la bat :
comparer sur NDCG@10 / Recall@20 avec les mêmes candidats. De même, le modèle entraîné ici est
**global** (un seul profil pour tout le monde) : il n'apprend que ce qui est commun à tous les
utilisateurs. La personnalisation vient de la mise à jour de session par-dessus w0.

## 3. Entraînement : `swiparis.train`

```powershell
python -m swiparis.train --db data/processed/common.sqlite --pairs data/processed/pairs.csv `
  --models-dir models
```

Modèle : `s(i | moment) = z(i) · (w_g + t[moment])`, `z(i) = normalize(W φ(i))`, perte logistique
par paires + L2. φ(i) = catégorie commune (one-hot) + latitude/longitude standardisées
(+ embeddings externes avec `--features fichier.npz` contenant `venue_ids` et `X`).
Early stopping sur la perte de validation. Le test n'est évalué qu'avec `--eval-test`
(une fois, hyperparamètres figés).

Sortie dans `models/<version>/` : `model.pt`, `item_vectors.npy` (z(i) précalculés),
`prior_w0.npy`, `time_vectors.npy`, `venue_ids.json`, `losses.csv`, `loss_curve.png`,
`metadata.json` (config, seed, métriques, SHA-256 des paires, commit git, versions des bibliothèques).
`models/latest.txt` pointe vers la dernière version.

Pour l'API (sans torch) : `from swiparis.registry import load_latest; m = load_latest("models");
m.scores(time_bucket="evening", profile=w_session)`.

## 4. BPR-MF : `swiparis.bprmf`

```powershell
python -m swiparis.bprmf --db data/processed/common.sqlite --pairs data/processed/pairs.csv `
  --output outputs/bprmf_report.json --save models/bprmf.npz
```

Filtrage collaboratif entraîné sur les visites des utilisateurs d'entraînement (perte BPR). Les
utilisateurs val/test sont nouveaux : leur vecteur est estimé par *fold-in* sur leurs premières
visites (50 % par défaut), item vectors figés. Le rapport donne la précision par paires, sur les
visites postérieures au profil, pour BPR-MF, la popularité et l'aléatoire. Pour le classement
complet : `BPRMF.load(path).scores(liste_de_venue_id_du_profil)` renvoie un score par lieu.
Même limite que ci-dessus (négatifs biaisés vers la popularité) : citer le NDCG@10 / Recall@20.

## 5. Catégories et exclusion de la restauration

La taxonomie est celle de Jeromsan (`config/common_taxonomy.json`, 10 catégories). Sur les visites
réelles, **restauration (40 %) et vie_nocturne (18 %)** font 58 % des 31 694 visites, et ces deux
catégories n'existent presque pas dans le catalogue de démo (27 annonces de restauration, 0 de vie
nocturne). `--drop-categories restauration,vie_nocturne` les exclut : il reste 13 268 visites, et
seulement 467 des 807 utilisateurs gardent au moins 10 visites (médiane : 11 visites par utilisateur).
Décision d'équipe à prendre : garder tout (plus de données, mais le modèle apprend surtout à classer
des restaurants et des bars) ou exclure (données plus pertinentes pour la démo, mais beaucoup plus petites).
