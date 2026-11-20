# Profils exemples (rapport de semaine 3)

Généré par `python -m swiparis.ml.examples` sur les utilisateurs **test**, après avoir figé
la configuration sur la validation (prior `models/prior_w0_phi.json`, η = 0.05). Règle de choix des
3 utilisateurs, fixée avant de voir leurs résultats : les 3 catégories dominantes les plus
fréquentes dans les profils test, et pour chacune l'utilisateur au plus petit identifiant.
Les réponses aux paires sont **simulées** à partir des 5 lieux du profil (protocole de
`swiparis.ml.simulation`) ; les visites suivantes ne servent qu'à vérifier les recommandations.
Les lieux Foursquare n'ont pas de nom dans la base : ils sont décrits par catégorie et zone.

## Utilisateur 1 : `foursquare:100923` (profil dominé par « restauration »)

**Profil (5 premiers lieux)** : restauration (centre) ; vie_nocturne (centre) ; restauration (sud) ; spectacle (est) ; vie_nocturne (centre)

**Paires proposées par la sélection active et réponses simulées** :

- visite (sud-ouest) vs nature (nord-ouest) : passe
- vie_nocturne (centre) vs restauration (nord-ouest) : ← gauche
- restauration (nord-ouest) vs loisirs (est) : ← gauche
- vie_nocturne (est) vs restauration (ouest) : droite →
- restauration (nord) vs nature (centre) : ← gauche
- vie_nocturne (sud-ouest) vs restauration (nord) : droite →

**Top 10 Foursquare après la session** :

| Rang | Catégorie | Zone | Rang de popularité | Visité plus tard |
|---|---|---|---|---|
| 1 | culture | centre | 1 / 6554 |  |
| 2 | culture | sud | 2 / 6554 |  |
| 3 | visite | ouest | 13 / 6554 |  |
| 4 | culture | sud | 3 / 6554 | oui |
| 5 | visite | centre | 4 / 6554 |  |
| 6 | visite | centre | 8 / 6554 |  |
| 7 | visite | centre | 14 / 6554 |  |
| 8 | visite | centre | 11 / 6554 |  |
| 9 | culture | centre | 6 / 6554 |  |
| 10 | visite | centre | 17 / 6554 | oui |

2 des 10 lieux recommandés font partie des 35 lieux qu'il a visités ensuite. NDCG@10 : session 0.158, w₀ seul 0.158, popularité 0.110.

**Ce que la démo lui proposerait (catalogue Paris Data, avec diversité)** :

1. Taxi Driver. Ciné-club de la bibliothèque Germaine Tillion (culture + visite)
2. Balades conviviales Paris & environs (visite)
3. À la découverte des quartiers parisiens : En route vers les usines Panhard et Levassor (13e) (visite)
4. Samuel Bak, fragments/espoir de Daniela Bak (culture + visite)
5. Pop Art Car 2.0 - Le défilé Renault ® (culture)

## Utilisateur 2 : `foursquare:130080` (profil dominé par « visite »)

**Profil (5 premiers lieux)** : loisirs (centre) ; restauration (nord) ; visite (nord) ; spectacle (centre) ; visite (centre)

**Paires proposées par la sélection active et réponses simulées** :

- restauration (centre) vs nature (est) : ← gauche
- nature (sud) vs sport (centre) : droite →
- visite (est) vs spectacle (centre) : droite →
- restauration (centre) vs nature (sud-ouest) : ← gauche
- vie_nocturne (centre) vs spectacle (nord) : ← gauche

**Top 10 Foursquare après la session** :

| Rang | Catégorie | Zone | Rang de popularité | Visité plus tard |
|---|---|---|---|---|
| 1 | culture | centre | 1 / 6554 |  |
| 2 | visite | centre | 4 / 6554 |  |
| 3 | visite | centre | 8 / 6554 |  |
| 4 | visite | centre | 11 / 6554 |  |
| 5 | culture | centre | 6 / 6554 |  |
| 6 | visite | centre | 14 / 6554 |  |
| 7 | visite | centre | 18 / 6554 |  |
| 8 | visite | centre | 21 / 6554 |  |
| 9 | visite | centre | 17 / 6554 |  |
| 10 | culture | sud | 3 / 6554 |  |

0 des 10 lieux recommandés font partie des 37 lieux qu'il a visités ensuite. NDCG@10 : session 0.000, w₀ seul 0.000, popularité 0.000.

**Ce que la démo lui proposerait (catalogue Paris Data, avec diversité)** :

1. Hommage à Hélène Mouchard-Zay (1940‑2026) (visite)
2. Les grands âges (culture)
3. La Course des Lumières (sport)
4. Rencontre avec Rob Grams : Le cinéma a-t-il un regard de classe ? (culture)
5. Quinzaine de l'accessibilité et de l'inclusion : exposition "ce corps qui échappe" de Hélène Mastrandréas (culture)

## Utilisateur 3 : `foursquare:123374` (profil dominé par « vie_nocturne »)

**Profil (5 premiers lieux)** : vie_nocturne (sud) ; sport (nord) ; nature (nord) ; restauration (nord) ; spectacle (nord-est)

**Paires proposées par la sélection active et réponses simulées** :

- restauration (sud-ouest) vs autre (est) : ← gauche
- restauration (sud) vs culture (centre) : ← gauche
- vie_nocturne (nord-ouest) vs nature (centre) : ← gauche
- restauration (ouest) vs vie_nocturne (est) : passe
- nature (centre) vs sport (sud-ouest) : droite →
- vie_nocturne (sud) vs spectacle (nord) : ← gauche

**Top 10 Foursquare après la session** :

| Rang | Catégorie | Zone | Rang de popularité | Visité plus tard |
|---|---|---|---|---|
| 1 | culture | sud | 2 / 6554 |  |
| 2 | culture | sud | 3 / 6554 |  |
| 3 | visite | ouest | 13 / 6554 |  |
| 4 | culture | centre | 1 / 6554 |  |
| 5 | visite | centre | 4 / 6554 |  |
| 6 | visite | sud | 34 / 6554 |  |
| 7 | culture | nord-ouest | 10 / 6554 |  |
| 8 | visite | centre | 8 / 6554 |  |
| 9 | culture | sud | 23 / 6554 |  |
| 10 | visite | ouest | 28 / 6554 |  |

0 des 10 lieux recommandés font partie des 18 lieux qu'il a visités ensuite. NDCG@10 : session 0.000, w₀ seul 0.000, popularité 0.000.

**Ce que la démo lui proposerait (catalogue Paris Data, avec diversité)** :

1. À la découverte des quartiers parisiens : En route vers les usines Panhard et Levassor (13e) (visite)
2. Conférence "Dans une composition, une fleur est une personne" (culture)
3. TaxiTram décembre 2026 (culture + visite)
4. Histoires d'A, un film emblématique de la lutte pour l'IVG (visite)
5. Tournicoti : les histoires du mercredi (culture)

## Lecture honnête

Sur l'ensemble des utilisateurs test, les écarts appariés (moyenne et IC 95 % bootstrap) sont :

- active-popularity : NDCG@10 +0.0191 [+0.0088, +0.0299], Recall@20 -0.0038 [-0.0130, +0.0051]
- prior_only-popularity : NDCG@10 +0.0164 [+0.0086, +0.0242], Recall@20 +0.0022 [-0.0068, +0.0115]
- active-uniform : NDCG@10 -0.0018 [-0.0085, +0.0049], Recall@20 -0.0061 [-0.0117, -0.0003]
- active-prior_only : NDCG@10 +0.0027 [-0.0048, +0.0101], Recall@20 -0.0060 [-0.0116, -0.0008]

Les recommandations restent des lieux très populaires (rang de popularité moyen du top 10 dans le premier centile) : le modèle classe d'abord par popularité, puis départage par catégorie et zone. Voir docs/modele.md pour l'analyse.
