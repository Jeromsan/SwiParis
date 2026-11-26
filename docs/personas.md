# Personas apprises sur les historiques

Code : `src/swiparis/ml/personas.py`. Tests : `tests/test_ml_personas.py`. Modèle : `models/personas.json`.
Résultats : `docs/resultats_ml/personas_*.json`. Exemple : `scripts/demo_personas.py`.

## Ce qu'est une persona ici

Un groupe d'utilisateurs Foursquare d'entraînement aux habitudes proches, trouvé dans les
données, pas un stéréotype inventé. Habitudes d'un utilisateur : répartition de ses visites
par catégorie, moment de la journée (heure locale de la visite) et zone de Paris. Les
groupes sont trouvés par k-moyennes sur les 490 utilisateurs d'entraînement.

Chaque persona a son propre profil w_k : même modèle et mêmes blocs que le prior global w₀
(catégorie, prix, zone, popularité), appris sur les paires de ses utilisateurs et rappelé vers
le w global. Le nombre de personas (k = 6) et ce rappel (λ = 0,01) sont choisis sur les paires
de validation ; le poids de popularité (×16) sur le NDCG@10 de validation, comme pour w₀.

| Persona | Part | Habitudes |
| --- | --- | --- |
| Bars · le soir · centre | 19 % | Bars 41 %, restaurants 41 % ; 52 % des visites le soir ; centre (80 %) |
| Restaurants · le soir · ouest | 12 % | Restaurants 58 %, bars 20 % ; soir (46 %) ; ouest (47 %) |
| Culture · l'après-midi · centre | 16 % | Restaurants 54 %, culture 16 %, visites 12 % ; après-midi (49 %) ; centre (46 %) |
| Restaurants · l'après-midi · nord-ouest | 11 % | Restaurants 68 % ; après-midi (51 %) ; nord-ouest (48 %) |
| Visites & Nature · l'après-midi · centre | 20 % | Restaurants 36 %, visites 34 %, culture 11 % ; après-midi (53 %) ; centre (77 %) |
| Concerts & Culture · le soir · centre | 22 % | Restaurants 45 %, bars 21 %, culture 12 %, concerts au-dessus de la moyenne ; soir (53 %) |

Le nom retient ce qui distingue la persona de l'ensemble des utilisateurs. Les restaurants
dominent partout : c'est la catégorie la plus visitée dans Foursquare.

## Pendant une séance

π_k = probabilité de ressembler à la persona k. Au départ : la part de chaque persona
(personne inconnue), ou la proximité de son historique connu, π_k ∝ part_k · exp(−β d²).
Après chaque choix : π_k ∝ π_k · σ(w_kᵀΔ)^τ. Le profil de départ est Σ π_k w_k ; les choix
l'affinent ensuite avec la même mise à jour que `SessionProfile`. τ = 0,5 et β = 3 sont
choisis sur la validation.

## Résultats (test lu une seule fois, 162 utilisateurs, 5 réponses simulées, η = 0,05)

Prédire les choix suivants d'une personne jamais vue, rattachée à une persona par son
historique connu (5 358 paires) : **62,4 % de bonnes réponses contre 57,7 %** pour le w
global (perte 0,643 contre 0,672). Même résultat en validation (63,0 % contre 57,0 %).

| Méthode | NDCG@10 | Recall@20 | Listes top 10 différentes |
| --- | --- | --- | --- |
| Séance, modèle global | 0,086 | 0,077 | 159 |
| Séance, personas, sans historique | 0,084 | 0,086 | 115 |
| Séance, personas, avec historique | 0,085 | 0,085 | 141 |
| Personas avec historique, sans réponse | 0,084 | 0,086 | 127 |
| w₀ global, sans réponse | 0,083 | 0,083 | 20 |
| Popularité | 0,067 | 0,081 | 13 |

Écarts appariés avec la séance globale (IC 95 %) : NDCG@10 −0,001 [−0,009 ; +0,007] avec
historique, −0,002 [−0,008 ; +0,003] sans ; Recall@20 +0,008 [−0,002 ; +0,018] et
+0,009 [−0,000 ; +0,018]. Personas avec historique contre w₀ seul, sans réponse : NDCG@10
+0,000 [−0,007 ; +0,007].

## Lecture honnête

- Les personas captent de vraies habitudes : elles prédisent mieux les choix d'utilisateurs
  jamais vus (+4,7 points), en validation comme en test.
- Avec l'historique, elles donnent des listes individuelles (127 listes différentes sur 162
  personnes au lieu de 20 avec w₀ seul) **sans perte de précision**. Même parcours,
  historiques différents : listes différentes (`scripts/demo_personas.py` : 6 activités
  communes sur 10 entre deux personas).
- Elles **n'améliorent pas** le classement du top 10 (NDCG@10) : les gains vus en
  validation (+0,015) ne se retrouvent pas en test. Ils venaient en partie du choix des
  réglages sur la validation. Le Recall@20 monte un peu, mais l'intervalle touche zéro.
- Pour une personne inconnue, les personas n'apportent rien de mesurable par rapport à la
  séance globale.
- Réglage de la séance : avec η = 0,5 (valeur par défaut de `SessionProfile`, utilisée par
  l'API), une séance simulée fait moins bien que w₀ seul en validation (NDCG@10 0,058 contre
  0,098) ; avec η = 0,05, 0,097 (`docs/resultats_ml/personas_val_eta0.5.json`).
- Limites : visites Foursquare de 2012-2013, non-visites qui ne sont pas des rejets, réponses
  simulées. Les choix du pilote restent nécessaires pour valider les personas.

## Dans l'application

L'API charge `models/personas.json` au démarrage (`SWIPARIS_PERSONAS`, `off` pour les
désactiver). Chaque séance part du mélange des personas, mis à jour après chaque choix, et
`/recommendations` renvoie les deux personas les plus proches avec leur pourcentage
(`persona.top`). Streamlit affiche « Tes choix ressemblent le plus à : … » avec une
explication. Une personne de l'application n'a pas d'historique Foursquare : la séance part
des parts de chaque persona (β ne sert pas).

Mesure de la configuration réelle (10 choix, top 3, règles de préférence,
`swiparis.ml.app_eval`) : les personas y semblent bien meilleures que le profil global
(+0,036 de NDCG@10 en validation), mais cet écart vient du poids de popularité (×16 contre
×2). À popularité égale, et sans signal de popularité comme pour les annonces Paris Data,
les deux font jeu égal (`docs/modele.md`, « Réglage de la séance »). Les personas servent
donc à **expliquer** les recommandations et à démarrer la séance, pas à les rendre plus
précises : c'est ainsi qu'il faut les présenter.

Le gain sur la prédiction des paires (62,4 % contre 57,7 %) ne dépend pas de ce poids : il
est mesuré avant le calibrage de la popularité, pour les personas comme pour le w global.

## Reproduire

```bash
python -m swiparis.ml.personas train
python -m swiparis.ml.personas evaluate --split val --eta 0.05      # choix de τ et β
python -m swiparis.ml.personas set --temperature 0.5 --beta 3
python -m swiparis.ml.personas evaluate --split test --final --eta 0.05
python scripts/demo_personas.py
```
