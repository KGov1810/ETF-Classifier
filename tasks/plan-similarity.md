# Plan de mise en œuvre : module `similarity`

Référence : `SPEC-similarity.md` (validée le 25/10/2026) · Tâches détaillées : `tasks/todo-similarity.md`

## Vue d'ensemble

1. **Le cœur** : l'index (vecteurs de composition et de texte), la recherche des voisins avec leur explication et l'alerte quasi-doublon, puis le vote des voisins et la sauvegarde.
2. **L'intégration** : l'index construit par `model.train` et rechargé avec le modèle, puis les commandes `search` et `evaluate`.

## Décisions d'architecture

- **Vocabulaire de tickers propre à l'index**, construit sur tout le jeu de référence. Il ne reprend pas celui du modèle, appris sur l'entraînement seul : sinon les tickers présents uniquement dans les ETF de test seraient invisibles pour la recherche.
- **La norme d'un ETF recherché inclut tous ses tickers**, même ceux absents de la référence. Sans cela, un ETF dont la moitié des lignes sont inconnues paraîtrait aussi proche qu'un ETF entièrement couvert.
- **Vecteurs de texte tirés du pipeline du modèle** : les embeddings si le modèle les utilise, sinon le TF-IDF. On compare donc les ETF avec la même représentation que celle qui a servi à les catégoriser.
- **Calculs matriciels** (produit creux pour la composition, produit dense ou creux pour le texte) : aucune boucle sur les 10 000 ETF de référence.
- **Exclusion de soi par identifiant de ligne** (`row_id`), et non par nom, puisque des noms d'ETF peuvent être en double.

## Modifications de modules existants

- `model.train` construit l'index sur toutes les lignes exploitables et l'enregistre dans le dossier du modèle (`similarity.joblib`).
- `model.registry` le recharge avec le modèle. Un modèle entraîné avant ce module n'a pas d'index : message clair invitant à le réentraîner.

## Liste des tâches

### Phase A : le cœur
- [x] S1 : configuration et construction de l'index
- [x] S2 : recherche des voisins, explication et alerte quasi-doublon
- [x] S3 : vote des voisins, sauvegarde et rechargement

### Point d'étape A
- [x] Tests verts, dont le cas « petit ETF contenu dans un grand »
- [x] 100 requêtes sur une référence de 10 000 ETF en moins de 10 s

### Phase B : l'intégration
- [x] S4 : index construit par `train` et rechargé avec le modèle
- [x] S5 : commande `search`
- [x] S6 : commande `evaluate` (vote des voisins face au modèle et à la référence triviale)

### Point d'étape final
- [ ] Les 4 critères de succès de `SPEC-similarity.md`
- [ ] Après le premier entraînement réel : poids composition/texte et usage du vote décidés avec vous

## Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Taille sur disque : l'index d'un modèle avec embeddings pèse environ 120 Mo (10 000 × 3 072 nombres) ; avec un réentraînement par jour, les versions s'accumulent | Moyen | Mesuré et affiché dans la fiche du modèle ; politique de conservation (garder les N dernières versions) à prévoir dans le module `retraining` |
| Modèles existants sans index | Faible | Message clair au chargement ; `search` et `evaluate` indiquent de réentraîner |
| Noms d'ETF en double | Faible | Exclusion de soi par `row_id` |
| Évaluation biaisée du vote | Moyen | `evaluate` construit l'index sur l'entraînement seul ; même jeu de test que le modèle |
| Embeddings des nouveaux ETF | Faible | Même cache et même confirmation de coût que `features`, `train` et `predict` |

## Statut (25/10/2026)

Tâches S1 à S6 codées et testées (251 tests verts sur l'ensemble du projet). 100 requêtes sur 10 000 ETF : 1,7 s.

Correction de la spec : la règle « texte seul si l'un des deux ETF n'a pas de composition » mettait les scores
sur des échelles différentes (un ETF sans composition passait devant des ETF partageant les mêmes tickers).
Désormais : texte seul uniquement si l'ETF recherché n'a pas de composition ; sinon, un voisin sans composition
a une similarité de composition de 0 dans le score combiné.

Limite connue : l'alerte quasi-doublon reprend la règle du module `data` (recouvrement rapporté à la plus petite
couverture), donc un petit ETF contenu dans un grand peut être signalé. À revoir avec le CSV réel.

## Questions ouvertes

Aucune bloquante.
