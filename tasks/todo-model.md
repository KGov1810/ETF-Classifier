# Tâches : module `model`

Définition de « terminé » pour chaque tâche : tests verts (`pytest -v`), aucune régression sur les modules
`data`, `costs` et `features`, aucun appel réseau dans les tests, aucun commit sans votre accord.

---

## M1 : métriques d'évaluation et référence triviale

**Description :** section `model` dans `settings.yaml` (grille de `C`, effectif minimal pour régler un seuil, seuil « incertain », nombre maximal de catégories, graine) ; `evaluate.py` (F1 micro, macro, par ETF, exactitude stricte, Hamming loss, détail par catégorie) ; `baseline.py` (prédit les catégories les plus fréquentes de l'entraînement).

**Critères d'acceptation :**
- [x] Les métriques correspondent aux valeurs calculées à la main sur un petit cas
- [x] La référence triviale prédit, pour chaque ETF, le nombre médian de catégories parmi les plus fréquentes
- [x] Une catégorie absente des prédictions et des vraies valeurs ne fausse pas le F1 macro (gérée explicitement)

**Vérification :** `pytest tests/model/test_evaluate.py tests/model/test_baseline.py -v`

**Dépendances :** aucune

**Fichiers :** `config/settings.yaml`, `src/etf_classifier/model/__init__.py`, `src/etf_classifier/model/evaluate.py`, `src/etf_classifier/model/baseline.py`, `tests/model/…`

**Taille :** M

---

## M2 : classifieur, seuils par catégorie et règles de prédiction

**Description :** régression logistique « un contre tous » avec `class_weight="balanced"` ; choix du seuil maximisant le F1 de chaque catégorie ; `predict()` avec au moins une catégorie, 5 au maximum, scores triés et indicateur « incertain ».

**Critères d'acceptation :**
- [x] Sur des scores construits à la main, le seuil retenu maximise le F1 de la catégorie ; sous l'effectif minimal, le seuil vaut 0,5
- [x] Un ETF sans catégorie au-dessus des seuils reçoit la meilleure ; jamais plus de 5 catégories retenues
- [x] « Incertain » est vrai si et seulement si le meilleur score est sous le seuil configuré

**Vérification :** `pytest tests/model/test_classifier.py -v`

**Dépendances :** M1

**Fichiers :** `src/etf_classifier/model/classifier.py`, `tests/model/test_classifier.py`

**Taille :** M

---

## M3 : réglage sur validation interne et réentraînement final

**Description :** `tune(train_df, pipeline_factory, settings)` redécoupe l'entraînement par groupe (80/20), choisit `C` par F1 macro et fixe les seuils, puis réentraîne sur tout l'entraînement.

**Critères d'acceptation :**
- [x] `tune()` ne reçoit que le jeu d'entraînement (vérifié par un test de signature et de comportement)
- [x] Sur un jeu synthétique, le modèle réglé bat nettement la référence triviale en F1 macro
- [x] Mêmes données et même graine, mêmes paramètres et mêmes prédictions

**Vérification :** `pytest tests/model/test_tuning.py -v`

**Dépendances :** M2

**Fichiers :** `src/etf_classifier/model/classifier.py`, `tests/model/test_tuning.py`, `tests/model/synthetic.py`

**Taille :** M

---

## Point d'étape A
- [ ] Tests verts, dont le jeu synthétique et la non-utilisation du test
- [ ] Revue rapide avec vous des scores obtenus sur le jeu synthétique

---

## M4 : variables influentes et rapport d'évaluation

**Description :** `explain.py` (10 coefficients les plus positifs par catégorie, avec le nom lisible de la variable) ; rapport Markdown `reports/model_<version>.md` (métriques globales, comparaison à la référence, détail par catégorie, variables influentes, catégories non apprenables).

**Critères d'acceptation :**
- [x] Sur le jeu synthétique, les variables influentes d'une catégorie sont bien les mots et tickers qui la définissent
- [x] Le rapport contient toutes les sections de la spec
- [x] Une catégorie sans exemple positif à l'entraînement est listée comme non apprenable

**Vérification :** `pytest tests/model/test_explain.py tests/model/test_report.py -v`

**Dépendances :** M3

**Fichiers :** `src/etf_classifier/model/explain.py`, `src/etf_classifier/model/evaluate.py`, `tests/model/test_explain.py`, `tests/model/test_report.py`

**Taille :** M

---

## M5 : sauvegarde versionnée, registre et modèle courant

**Description :** `registry.py` : `save_bundle()` écrit `artifacts/models/<version>/` (modèle, pipeline de variables, `metrics.json`, `MODEL_CARD.md`) ; `load_bundle(version | "current")` ; `registry.json` (modèle courant, historique) ; `promote()`.

**Critères d'acceptation :**
- [x] Un modèle rechargé donne exactement les mêmes prédictions
- [x] Le premier modèle sauvegardé devient courant ; les suivants seulement avec `promote()`
- [x] Charger « current » sans modèle enregistré donne un message clair

**Vérification :** `pytest tests/model/test_registry.py -v`

**Dépendances :** M4

**Fichiers :** `src/etf_classifier/model/registry.py`, `tests/model/test_registry.py`

**Taille :** M

---

## M6 : commande `train`

**Description :** `python -m etf_classifier.model.train --csv … [--no-azure] [--yes] [--promote]` : préparation des données, variables (confirmation de coût sauf `--yes`), réglage, évaluation unique sur le test, rapport, sauvegarde, promotion éventuelle.

**Critères d'acceptation :**
- [x] Entraînement de bout en bout sur la fixture, sans Azure, avec rapport et dossier de modèle
- [x] Avec le faux client Azure, le bloc embeddings est utilisé et son coût apparaît dans le résumé
- [x] 10 000 ETF synthétiques (blocs locaux) en moins de 5 minutes

**Vérification :** `pytest tests/model/test_train.py -v`

**Dépendances :** M5

**Fichiers :** `src/etf_classifier/model/train.py`, `tests/model/test_train.py`

**Taille :** M

---

## Point d'étape B
- [ ] Tests verts, test de performance compris
- [ ] Revue avec vous d'un rapport d'évaluation généré sur la fixture

---

## M7 : chargement sans catégorie et commande `predict`

**Description :** paramètre `require_categories=False` dans `load_dataset()` (module `data`) ; `python -m etf_classifier.model.predict --model current --csv nouveaux.csv [--output …]` écrit un CSV de propositions (catégories retenues, scores des 5 premières, indicateur « incertain »).

**Critères d'acceptation :**
- [x] Un CSV sans colonne `Category` se charge ; les 90 tests du module `data` restent verts
- [x] Le CSV de sortie contient, pour chaque ETF, au moins une catégorie, des scores et l'indicateur « incertain »
- [x] Message clair si aucun modèle courant n'existe

**Vérification :** `pytest tests/data tests/model/test_predict.py -v`

**Dépendances :** M5

**Fichiers :** `src/etf_classifier/data/loader.py`, `src/etf_classifier/data/validation.py`, `src/etf_classifier/model/predict.py`, `tests/model/test_predict.py`, `tests/data/test_loader.py`

**Taille :** M

---

## M8 : commande `compare`

**Description :** `python -m etf_classifier.model.compare --csv … --experiments local,local+embeddings` : chaque configuration est entraînée sur le même découpage ; tableau comparatif dans `reports/model_comparison.md` (référence triviale incluse, F1 micro et macro, temps d'entraînement, coût Azure).

**Critères d'acceptation :**
- [x] Toutes les expériences utilisent exactement le même jeu de test (vérifié par empreinte)
- [x] Le tableau contient la référence triviale et chaque expérience demandée
- [x] Une expérience inconnue donne un message clair listant les expériences disponibles

**Vérification :** `pytest tests/model/test_compare.py -v`

**Dépendances :** M6

**Fichiers :** `src/etf_classifier/model/compare.py`, `config/settings.yaml`, `tests/model/test_compare.py`

**Taille :** M

---

## Point d'étape final
- [ ] Les 5 critères de succès de `SPEC-model.md` (ceux sur le CSV réel dès que vous l'aurez)
- [ ] Objectif chiffré fixé avec vous après le premier entraînement réel
