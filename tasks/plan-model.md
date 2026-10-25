# Plan de mise en œuvre : module `model`

Référence : `SPEC-model.md` (validée le 25/10/2026) · Tâches détaillées : `tasks/todo-model.md`

## Vue d'ensemble

Trois phases :
1. **Le cœur** : métriques et référence triviale, puis le classifieur (seuils, règle « au moins une catégorie », indicateur « incertain »), puis le réglage sur validation interne. Dès la fin de cette phase, on vérifie sur un jeu synthétique que le modèle bat nettement la référence.
2. **La traçabilité** : variables influentes et rapport d'évaluation, sauvegarde versionnée avec registre, puis la commande `train` qui enchaîne tout.
3. **L'usage** : `predict` sur un CSV de nouveaux ETF, et `compare` pour mesurer l'apport des embeddings.

## Décisions d'architecture

- **Le réglage ne reçoit que le jeu d'entraînement** : la fonction `tune()` n'a pas accès au test, ce qui rend la fuite impossible par construction, et un test le vérifie.
- **Validation interne par groupe** : on réutilise `make_split()` du module `data` sur le jeu d'entraînement, pour que les quasi-doublons ne faussent pas non plus le réglage.
- **Un dossier par modèle** contenant modèle et pipeline de variables : l'application et le job quotidien chargent un seul dossier, sans risque de mélanger des versions.
- **Évaluation séparée du modèle** (`evaluate.py`) : la référence triviale et le classifieur sont évalués par exactement le même code.

## Modification d'un module existant

`predict` doit lire un CSV de **nouveaux ETF, sans colonne `Category`**. Aujourd'hui, le module `data` exclut une ligne sans catégorie. J'ajouterai un paramètre `require_categories` (vrai par défaut) à `load_dataset()`. Le comportement actuel reste inchangé, et les 90 tests existants doivent rester verts.

## Liste des tâches

### Phase A : le cœur
- [x] M1 : métriques d'évaluation et référence triviale
- [x] M2 : classifieur, seuils par catégorie et règles de prédiction
- [x] M3 : réglage sur validation interne (`C`, seuils) et réentraînement final

### Point d'étape A
- [x] Sur un jeu synthétique, le modèle bat nettement la référence triviale (F1 macro)
- [x] Test de non-utilisation du jeu de test vert

### Phase B : la traçabilité
- [x] M4 : variables influentes et rapport d'évaluation
- [x] M5 : sauvegarde versionnée, registre et modèle courant
- [x] M6 : commande `train`

### Point d'étape B
- [x] Entraînement de bout en bout sur la fixture ; rechargement avec des prédictions identiques
- [x] Entraînement sur 10 000 ETF synthétiques (blocs locaux) en moins de 5 minutes

### Phase C : l'usage
- [x] M7 : chargement sans catégorie et commande `predict`
- [x] M8 : commande `compare`

### Point d'étape final
- [ ] Les 5 critères de succès de `SPEC-model.md`, dont ceux sur le CSV réel, quand vous l'aurez
- [ ] Objectif chiffré défini avec vous après le premier entraînement réel

## Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Catégories rares : trop peu d'exemples en validation pour régler un seuil fiable | Moyen | Seuil par défaut à 0,5 sous un effectif minimal (configurable) ; signalé dans le rapport |
| Temps d'entraînement avec le bloc embeddings (3 072 colonnes denses × 10 000 ETF) | Moyen | Mesuré par le test de performance ; solveur adapté aux matrices denses si besoin ; grille de `C` limitée à 3 valeurs |
| Fuite du test vers le réglage | Élevé | `tune()` ne reçoit que l'entraînement ; test dédié |
| Régression du module `data` en ajoutant `require_categories` | Faible | Paramètre facultatif, valeur par défaut inchangée ; suite `data` relancée |
| Comparaison biaisée entre expériences | Moyen | `compare` utilise le même découpage et la même graine pour toutes les configurations |

## Statut (25/10/2026)

Tâches M1 à M8 codées et testées (225 tests verts sur l'ensemble du projet).
Jeu synthétique : F1 macro 0,969 contre 0,067 pour la référence triviale. Entraînement de 10 000 ETF : 4,3 s.
Reste : premier entraînement sur le CSV réel et objectif chiffré à fixer avec vous.

Ajustements : extraction de `announce_embedding_plan` / `confirm_spending` dans `features/build.py`
(réutilisées par train, predict, compare) et ajout de `FeaturePipeline.attach_embedder`.

## Questions ouvertes

Aucune bloquante.
