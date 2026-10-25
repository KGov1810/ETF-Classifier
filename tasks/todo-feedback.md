# Tâches : module `feedback`

Définition de « terminé » pour chaque tâche : tests verts (`pytest -v`), aucune régression sur les modules
existants, aucun appel réseau dans les tests, aucun commit sans votre accord.

---

## F1 : configuration, dépendance `filelock` et écriture verrouillée du journal

**Description :** section `feedback` dans `settings.yaml` (chemins du journal, du registre et des sauvegardes, délai du verrou, nombre de sauvegardes conservées, seuils de l'alerte de biais) ; `store.py` : schéma du journal, écriture d'une ligne complète sous verrou, lecture sous verrou, identifiant de validation.

**Critères d'acceptation :**
- [x] Une ligne écrite se relit à l'identique ; l'en-tête n'est écrit qu'une fois
- [x] Plusieurs processus écrivant en même temps : toutes les lignes présentes, aucune mélangée
- [x] Verrou indisponible au-delà du délai : erreur claire, aucune écriture

**Vérification :** `pytest tests/feedback/test_store.py -v`

**Dépendances :** aucune

**Fichiers :** `requirements.txt`, `config/settings.yaml`, `src/etf_classifier/feedback/__init__.py`, `src/etf_classifier/feedback/store.py`, `tests/feedback/…`

**Taille :** M

---

## F2 : contrôles, registre des catégories et enregistrement d'une validation

**Description :** `checks.py` (nom, composition, catégories, doublon de nom) ; `categories.py` (registre initialisé depuis le CSV d'origine, création tracée) ; `record_validation()` qui contrôle, déduit la décision (`accepté` / `corrigé`) et écrit.

**Critères d'acceptation :**
- [x] Chaque contrôle refuse avec un message clair (nom vide, composition illisible, aucune catégorie, catégorie inconnue sans création, ETF déjà présent)
- [x] Une catégorie créée avec `create_categories=True` entre dans le registre avec son auteur et sa date
- [x] La décision vaut `accepté` si les catégories validées sont celles proposées, `corrigé` sinon

**Vérification :** `pytest tests/feedback/test_record.py -v`

**Dépendances :** F1

**Fichiers :** `src/etf_classifier/feedback/checks.py`, `src/etf_classifier/feedback/categories.py`, `src/etf_classifier/feedback/store.py`, `tests/feedback/test_record.py`

**Taille :** M

---

## F3 : annulation et sauvegarde quotidienne

**Description :** `cancel_validation()` ajoute une ligne `annulation` ; `active_validations()` renvoie les validations non annulées ; sauvegarde datée du journal au premier enregistrement de la journée, avec conservation des 30 dernières.

**Critères d'acceptation :**
- [x] Une validation annulée n'apparaît plus dans les validations actives ; la ligne d'annulation reste dans le journal
- [x] Annuler un identifiant inconnu ou déjà annulé : erreur claire
- [x] Une seule sauvegarde par jour ; au-delà de 30, les plus anciennes sont supprimées

**Vérification :** `pytest tests/feedback/test_cancel_backup.py -v`

**Dépendances :** F2

**Fichiers :** `src/etf_classifier/feedback/store.py`, `tests/feedback/test_cancel_backup.py`

**Taille :** S

---

## Point d'étape A
- [ ] Test de concurrence et test de verrou indisponible verts
- [ ] Revue rapide avec vous d'un journal fictif

---

## F4 : fusion du journal par le module `data`

**Description :** `load_dataset()` ajoute les validations actives du journal (lues sous verrou) au CSV d'origine, avec la colonne `source` et des `row_id` séparés ; mêmes contrôles de qualité.

**Critères d'acceptation :**
- [x] Un ETF validé apparaît dans le jeu de référence ; un ETF annulé n'y apparaît pas
- [x] Le CSV d'origine n'est jamais modifié (empreinte) ; aucun `row_id` en double
- [x] Sans journal, résultat identique à aujourd'hui (92 tests `data` verts)

**Vérification :** `pytest tests/data tests/feedback/test_merge.py -v`

**Dépendances :** F3

**Fichiers :** `src/etf_classifier/data/loader.py`, `tests/feedback/test_merge.py`

**Taille :** M

---

## F5 : commande d'import d'un lot

**Description :** `python -m etf_classifier.feedback.importer --csv valides.csv --by "Prénom Nom" [--create-categories]` : format de sortie de `predict` complété d'une colonne `Category` ; compte rendu ligne par ligne.

**Critères d'acceptation :**
- [x] Les lignes valides sont enregistrées avec la décision déduite des colonnes `PredictedCategories` et `Category`
- [x] Les lignes refusées sont listées avec leur raison, sans bloquer les autres
- [x] Le compte rendu indique le nombre de lignes acceptées, refusées et de catégories créées

**Vérification :** `pytest tests/feedback/test_import.py -v`

**Dépendances :** F2

**Fichiers :** `src/etf_classifier/feedback/importer.py`, `tests/feedback/test_import.py`

**Taille :** S

---

## F6 : rapport de suivi

**Description :** `python -m etf_classifier.feedback.report [--since …]` écrit `reports/feedback.md` : validations par jour et par validateur, taux d'acceptation sans modification, alerte de biais, catégories souvent ajoutées ou retirées, catégories créées, annulations.

**Critères d'acceptation :**
- [x] Taux d'acceptation global et par validateur conformes au calcul à la main
- [x] Alerte de biais au-delà du seuil (avec le nombre minimal de validations)
- [x] Journal absent : message clair, sans erreur

**Vérification :** `pytest tests/feedback/test_report.py -v`

**Dépendances :** F3

**Fichiers :** `src/etf_classifier/feedback/report.py`, `tests/feedback/test_report.py`

**Taille :** M

---

## Point d'étape final
- [ ] Les 4 critères de succès de `SPEC-feedback.md`
- [ ] Boucle complète sur la fixture : `predict` → import des validations → `train` qui voit les nouveaux ETF
