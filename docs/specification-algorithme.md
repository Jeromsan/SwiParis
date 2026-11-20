# Spécification de l'algorithme de recommandation

## Représentation et modèle

Chaque activité \(i\) est décrite par un vecteur commun \(\phi(i)\) : catégories
normalisées, type de prix, répartition des horaires, zone géographique et, à
terme, représentation textuelle. Le même espace de caractéristiques doit être
utilisé pour apprendre le prior et personnaliser les recommandations. Une
information manquante reste inconnue; elle n'est pas remplacée par une valeur
positive (notamment le prix inconnu n'est pas gratuit).

Le profil utilisateur est un vecteur de poids \(w_u\). Pour deux activités
\(a,b\), le modèle de Bradley–Terry linéaire estime :

\[
P(a \succ b \mid u)=\sigma\!\left(w_u^\top(\phi(a)-\phi(b))\right),
\qquad \sigma(x)=\frac{1}{1+e^{-x}}.
\]

Le score individuel d'une activité est \(s_u(i)=w_u^\top\phi(i)\). Pour une
réponse observée (choix \(c\), rejet \(r\)), poser
\(\Delta=\phi(c)-\phi(r)\). La perte logistique de cette comparaison est
\[
\ell(w_u;\Delta)=\log(1+\exp(-w_u^\top\Delta)).
\]

Le profil part du prior population \(w_0\). En mode SGD, la mise à jour en ligne
implémentée est
\[
w_u \leftarrow w_u+
\eta(1-\sigma(w_u^\top\Delta))\Delta
-\lambda(w_u-w_0).
\]
\(\eta\) règle l'amplitude de l'apprentissage à partir du choix; \(\lambda\)
règle le rappel vers le prior et limite la dérive au fil de la session. Cette
forme de rappel est volontaire : avec \(w_0=0\), elle devient le rappel vers
zéro. Le code fournit aussi un mode bayésien en ligne avec approximation de
Laplace, qui maintient une moyenne et une covariance du profil; ce mode n'est
pas le réglage de référence de l'évaluation.

## Sélection des comparaisons

Pour les activités éligibles \(i,j\), l'acquisition combine l'incertitude du
choix prédit et la différence des activités :
\[
A(i,j)=H_2(P(i\succ j))+\beta(1-\cos(\phi(i),\phi(j))),
\]
où \(H_2(p)=-p\log_2p-(1-p)\log_2(1-p)\). L'entropie favorise les choix
incertains; la distance cosinus évite de dépenser une comparaison sur deux
activités presque identiques. \(\beta\) règle le compromis entre information et
diversité.

La recherche se fait dans un pool borné : moitié des activités les mieux
classées par le profil, moitié tirées uniformément parmi les autres. Les paires
déjà présentées et les vecteurs identiques sont écartés. Le côté gauche/droite
est randomisé. Les filtres de créneau, budget et zone doivent être appliqués
avant la sélection; le module ML reçoit uniquement les activités éligibles.

Après la session, les activités sont triées par \(s_u(i)\). Le classement peut
appliquer un re-ranking MMR pour équilibrer pertinence et diversité dans la
liste présentée.

## Paramètres et statut

Valeurs initiales du code : \(\eta=0{,}5\), \(\lambda=0{,}05\),
\(\beta=0{,}5\); pool de 40 activités et diversité MMR de 0,3. Ce sont des
valeurs de départ, **pas des paramètres validés**. Le réglage doit être fait sur
la validation, sans consulter les utilisateurs du test. Le modèle nécessite un
prior \(w_0\) exprimé dans le même ordre de features que \(\phi\).

La spécification décrit l'algorithme implémenté dans
`src/swiparis/ml/model.py` et `src/swiparis/ml/selection.py`. Les règles
d'évaluation, les limites des données implicites et le statut du pilote sont
consignés dans [evaluation.md](./evaluation.md).
