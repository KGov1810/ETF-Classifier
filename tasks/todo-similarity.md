# Tâches : module `similarity`

Définition de « terminé » pour chaque tâche : tests verts (`pytest -v`), aucune régression sur les modules
existants, aucun appel réseau dans les tests, aucun commit sans votre accord.

---

## S1 : configuration et construction de l'index

**Description :** section `similarity` dans `settings.yaml` (poids composition/texte, nombre de voisins, nombre de tickers partagés affichés, valeurs de k évaluées) ; `SimilarityIndex.build(reference_df, pipeline)` : vecteurs de composition (vocabulaire propre, poids normalisés L2) et vecteurs de texte (embeddings ou TF-IDF du pipeline).

**Critères d'acceptation :**
- [x] Le vocabulaire couvre tous les tickers de la référence, y compris ceux absents du vocabulaire du modèle
- [x] Le bloc de texte utilisé est celui des embeddings s'il est actif, sinon le TF-IDF ; il est indiqué dans l'index
- [x] Un ETF de la référence sans composition a un vecteur de composition nul, sans erreur

**Vérification :** `pytest tests/similarity/test_index.py -v`

**Dépendances :** aucune

**Fichiers :** `config/settings.yaml`, `src/etf_classifier/similarity/__init__.py`, `src/etf_classifier/similarity/index.py`, `tests/similarity/…`

**Taille :** M

---

## S2 : recherche des voisins, explication et alerte quasi-doublon

**Description :** `index.query(df, k, exclude_row_ids=None)` : score combiné (composition et texte, poids renormalisés si une composition est vide), k voisins triés, explication (recouvrement pondéré en %, 5 principaux tickers partagés), alerte quasi-doublon (règle du module `data`).

**Critères d'acceptation :**
- [x] Deux compositions identiques : similarité de composition de 1 et alerte quasi-doublon
- [x] Un petit ETF contenu dans un grand : similarité de composition nettement inférieure à 1, recouvrement de 100 %
- [x] Un ETF de la référence n'est jamais son propre voisin ; composition vide : texte seul, avec une mention

**Vérification :** `pytest tests/similarity/test_query.py -v`

**Dépendances :** S1

**Fichiers :** `src/etf_classifier/similarity/index.py`, `tests/similarity/test_query.py`

**Taille :** M

---

## S3 : vote des voisins, sauvegarde et rechargement

**Description :** `neighbor_vote(neighbors)` (score par catégorie = somme des scores des voisins qui l'ont / somme des scores) et décision avec les mêmes règles que le modèle (au moins 1 catégorie, au plus 5) ; `save()` / `load()` de l'index.

**Critères d'acceptation :**
- [x] Le vote calculé à la main sur un petit cas est retrouvé
- [x] Sur le jeu synthétique, le vote bat nettement la référence triviale
- [x] Sauvegarde puis rechargement : mêmes voisins et mêmes scores

**Vérification :** `pytest tests/similarity/test_vote.py -v`

**Dépendances :** S2

**Fichiers :** `src/etf_classifier/similarity/vote.py`, `src/etf_classifier/similarity/index.py`, `tests/similarity/test_vote.py`

**Taille :** S

---

## Point d'étape A
- [ ] Tests verts
- [ ] 100 requêtes sur une référence de 10 000 ETF en moins de 10 s

---

## S4 : index construit par `train` et rechargé avec le modèle

**Description :** `model.train` construit l'index sur toutes les lignes exploitables et l'écrit dans le dossier du modèle ; `load_bundle()` le recharge (`bundle.similarity`), ou `None` avec un message clair pour un modèle plus ancien. Taille de l'index indiquée dans la fiche du modèle.

**Critères d'acceptation :**
- [x] Après `train`, le dossier du modèle contient `similarity.joblib`
- [x] `load_bundle()` renvoie un index qui donne les mêmes voisins qu'avant la sauvegarde
- [x] Un dossier de modèle sans index se charge quand même (`similarity` vaut `None`)

**Vérification :** `pytest tests/model tests/similarity/test_integration.py -v`

**Dépendances :** S3

**Fichiers :** `src/etf_classifier/model/train.py`, `src/etf_classifier/model/registry.py`, `tests/similarity/test_integration.py`

**Taille :** S

---

## S5 : commande `search`

**Description :** `python -m etf_classifier.similarity.search --csv nouveaux.csv [--model current] [--k 5] [--yes]` : CSV des voisins (un ETF × un voisin par ligne) et résumé Markdown ; embeddings via cache et confirmation de coût.

**Critères d'acceptation :**
- [x] Pour chaque nouvel ETF, k voisins avec scores, catégories et explication
- [x] Un quasi-doublon est signalé dans le résumé
- [x] Message clair si aucun modèle courant, ou un modèle sans index

**Vérification :** `pytest tests/similarity/test_search.py -v`

**Dépendances :** S4

**Fichiers :** `src/etf_classifier/similarity/search.py`, `tests/similarity/test_search.py`

**Taille :** M

---

## S6 : commande `evaluate`

**Description :** `python -m etf_classifier.similarity.evaluate --csv … [--yes]` : index construit sur l'entraînement seul, vote des voisins pour chaque ETF de test (k = 5, 10, 20), comparaison au modèle réglé sur le même découpage et à la référence triviale ; rapport `reports/similarity_evaluation.md`.

**Critères d'acceptation :**
- [x] L'index d'évaluation ne contient aucun ETF de test
- [x] Le rapport contient le vote pour chaque k, le modèle et la référence triviale (F1 macro et micro)
- [x] Même jeu de test que le modèle (empreinte affichée)

**Vérification :** `pytest tests/similarity/test_evaluate.py -v`

**Dépendances :** S3

**Fichiers :** `src/etf_classifier/similarity/evaluate.py`, `tests/similarity/test_evaluate.py`

**Taille :** M

---

## Point d'étape final
- [ ] Les 4 critères de succès de `SPEC-similarity.md`
- [ ] Après le premier entraînement réel : poids composition/texte et usage du vote décidés avec vous
