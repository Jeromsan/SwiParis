# Rapport de semaine 3 et protocole d'évaluation

État au 20 novembre 2026. Ce rapport sépare les éléments vérifiés, les outils
préparés et les résultats qui ne peuvent pas encore être revendiqués.

## Statut vérifié

- Le cœur Bradley-Terry, les profils de session, la sélection active et le
  classement diversifié sont implémentés dans `src/swiparis/ml/`. La
  spécification concise et les paramètres provisoires sont dans
  [specification-algorithme.md](./specification-algorithme.md); les notes
  d'implémentation plus détaillées restent dans [modele.md](./modele.md).
- Les tests du dépôt passent : `python -m unittest discover -s tests -v`
  (32 tests au 20/11/2026).
- La base locale `data/processed/common.sqlite` contient 816 utilisateurs,
  31 694 visites Foursquare, 6 554 activités Foursquare et 1 778 activités
  Paris Data. Chaque utilisateur a au moins dix visites dans cette base.
- Avec la graine 42 et cinq visites de profil, le découpage local donne
  490 / 163 / 163 utilisateurs; 162 des 163 utilisateurs test ont au moins un
  lieu distinct à évaluer après leur profil, soit 3 161 couples utilisateur-
  lieu futurs distincts au total.
- Cette base et les données brutes sont ignorées par Git. Ces volumes sont donc
  une observation de la machine de travail, pas des données reproductibles pour
  un clone du dépôt.
- Il n'y a pas de prior population w₀ validé, de choix de préférence par paire
  enregistrés, de score produit par utilisateur pour le jeu test, ni de
  résultats expérimentaux. Les métriques et comparaisons ci-dessous ne sont
  donc pas annoncées comme résultats.

## Découpage et prévention des fuites

Le protocole préparé dans `swiparis.ml.evaluation` utilise une graine fixe et
sépare les **utilisateurs** avant toute adaptation : 60 % entraînement, 20 %
validation et 20 % test. Avec 816 utilisateurs, l'arrondi donne 490 / 163 / 163.
Les utilisateurs de validation et de test ne contribuent pas au prior appris
sur l'entraînement.

Dans chaque utilisateur de validation ou de test, les cinq premiers lieux
distincts, ordonnés par première visite UTC, sont réservés au profil; les lieux
distincts visités ensuite forment la pertinence future. Les visites répétées
d'un même lieu ne le font pas passer artificiellement du profil à l'évaluation.
Les utilisateurs sans visite future distincte sont exclus des métriques et
comptés dans le rapport.

Contrôles obligatoires avant une évaluation :

1. Construire toute feature agrégée des visites Foursquare avec les utilisateurs
   d'entraînement uniquement (`foursquare_items(..., users=train_users)`).
2. Garder les utilisateurs de validation/test hors de l'estimation du prior et
   du réglage final des hyperparamètres. Ne consulter le jeu test qu'une fois le
   protocole arrêté.
3. Pour chaque utilisateur évalué, ne pas classer les lieux déjà présents dans
   ses cinq visites de profil; les scores fournis couvrent le reste du catalogue
   Foursquare.
4. Les horodatages servent au tri seulement; le split reste par utilisateur,
   pas par ligne de visite.

## Métriques et outil livré

`python -m swiparis.ml.evaluation --scores SCORES.csv --output evaluation.json`
calcule, pour chaque colonne de score `active_score` et `uniform_score`,
NDCG@10, Recall@20 et AUC, puis un bootstrap à 2 000 réplications par
utilisateur avec IC à 95 %. Il donne aussi l'IC bootstrap apparié de
`actif - uniforme`. Les paramètres de graine, de taille du profil et de
réplications sont configurables dans la commande (`--seed`, `--profile-items`,
`--bootstrap-samples`). Le CSV doit contenir une ligne par couple
utilisateur/candidat et les colonnes `user_id,item_id,active_score,uniform_score`.

L'AUC est explicitement une **approximation à feedback implicite** : les lieux
non visités sont considérés comme négatifs, alors qu'ils peuvent simplement ne
pas avoir été vus. NDCG et Recall mesurent la récupération des visites futures,
pas une préférence déclarée. Les intervalles bootstrap rééchantillonnent les
utilisateurs, non les visites, afin de préserver le regroupement des observations.

Ce script calcule les métriques à partir de scores fournis; il ne fabrique pas
de scores ni de préférences. Il ne peut pas encore être exécuté de bout en bout
sur le modèle actuel : les visites historiques ne disent pas lequel de deux
éléments un utilisateur aurait choisi, alors que `SessionProfile.update`
attend une préférence choisie/rejetée. Le prior w₀ et le générateur de scores
utilisateur pour les deux politiques manquent également.

## Ablation prévue

La comparaison active/uniforme est appariée : mêmes utilisateurs test, mêmes
lieux candidats, même budget de questions et mêmes visites futures; seule la
politique de sélection de la paire change. Le script rapporte les métriques
pour chaque politique et la différence appariée avec IC à 95 %. Pour une
ablation interprétable, les deux politiques doivent recevoir les mêmes réponses
de préférence pour les mêmes paires montrées. Les visites Foursquare seules ne
fournissent pas ces réponses; aucun résultat d'ablation n'est donc disponible.

## Pilote avec volontaires — protocole à valider

**Objectif exploratoire :** vérifier la compréhension de l'interface de choix
par paires, le fonctionnement de la personnalisation et la durée d'une session;
ne pas présenter un petit pilote comme une preuve d'efficacité statistique.

**Recrutement :** volontaires adultes, participation libre, aucun lien avec une
évaluation scolaire ou professionnelle. Commencer par un petit groupe
exploratoire (par exemple 5 à 10 personnes) et augmenter seulement si le
protocole est validé. Ne pas importer leurs identités dans les données de
développement.

**Consentement :** fournir avant la session une information claire sur l'objet,
les données collectées, leur usage, la durée de conservation et le contact
responsable. Obtenir un accord explicite avant de commencer; participation et
retrait sont possibles sans justification. Ne pas collecter de nom, adresse,
localisation continue ou donnée sensible. Utiliser un identifiant aléatoire,
conserver séparément et brièvement tout éventuel document de consentement, puis
supprimer les données brutes à l'échéance annoncée. Faire valider le formulaire
et la durée de conservation par l'encadrant responsable avant le recrutement.

**Déroulement proposé :** expliquer la comparaison sans suggérer de réponse;
laisser le volontaire choisir ou passer une paire; enregistrer uniquement
l'identifiant aléatoire, les paires présentées, le choix ou le passage, et un
temps de réponse arrondi. Terminer par quelques questions de compréhension et
de confort. Arrêter si la personne retire son accord ou si un incident de
confidentialité survient.

**Statut :** protocole de travail, pas encore autorisation, formulaire approuvé,
recrutement, consentement obtenu ou pilote réalisé.

## Actions encore bloquées

- **Métriques réelles du modèle :** il manque des choix par paires (ou un
  protocole simulé explicitement qualifié), un prior w₀ validé et les scores
  des deux politiques pour tous les candidats test.
- **Ablation expérimentale :** même blocage; un historique de visites ne permet
  pas de reconstruire les réponses aux comparaisons contrefactuelles.
- **Pilote :** impossible de recruter ou d'attester le consentement depuis le
  dépôt; il faut l'accord du responsable, le formulaire validé et des
  volontaires.
- **Reproductibilité des chiffres locaux :** les bases et fichiers bruts sont
  absents du dépôt et exclus de Git. Les publier ou les fournir doit respecter
  les conditions de la source et les règles du projet.
