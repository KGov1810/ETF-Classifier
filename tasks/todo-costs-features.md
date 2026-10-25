# Tâches : modules `costs` et `features`

Définition de « terminé » pour chaque tâche : tests verts (`pytest -v`), aucune régression sur le module `data`,
aucun appel réseau dans les tests, aucun commit sans votre accord.

---

## A1 : grille tarifaire et chargement contrôlé

**Description :** créer `config/pricing.yaml` (USD, Data Zone, borne haute) et `pricing.py`, qui lit et contrôle la grille et associe un nom de déploiement à son modèle.

**Critères d'acceptation :**
- [x] Une grille valide se charge ; un prix manquant ou négatif lève une erreur claire
- [x] Un déploiement inconnu est détecté (sans coût fantôme à zéro)
- [x] L'âge de la grille (`as_of`) est calculable, pour l'avertissement au-delà de 90 jours

**Vérification :** `pytest tests/costs/test_pricing.py -v`

**Dépendances :** aucune

**Fichiers :** `config/pricing.yaml`, `src/etf_classifier/costs/__init__.py`, `src/etf_classifier/costs/pricing.py`, `tests/costs/test_pricing.py`, `.gitignore` (ajout de `logs/`)

**Taille :** S

---

## A2 : comptage des tokens et estimations

**Description :** `count_tokens()` avec `tiktoken`, `estimate_embeddings()` et `estimate_chat()`, avec tarif long contexte et indicateur « approximatif ».

**Critères d'acceptation :**
- [x] 1 000 000 de tokens d'embedding valent exactement 0,13 $ avec la grille de test
- [x] Le tarif long contexte s'applique au-delà du seuil
- [x] L'estimation de GPT-6.1 Sol est marquée approximative

**Vérification :** `pytest tests/costs/test_estimate.py -v`

**Dépendances :** A1

**Fichiers :** `requirements.txt`, `src/etf_classifier/costs/tokens.py`, `src/etf_classifier/costs/estimate.py`, `tests/costs/test_estimate.py`

**Taille :** S

---

## A3 : journal du coût réel (`CostTracker`)

**Description :** `CostTracker.record()` calcule le coût à partir de l'usage renvoyé par Azure (entrée, cache, sortie) et ajoute une ligne à `logs/azure_usage.csv`.

**Critères d'acceptation :**
- [x] Une ligne par appel, avec toutes les colonnes de la spec ; jamais le contenu des textes
- [x] Déploiement inconnu : ligne enregistrée avec ses tokens, coût vide et avertissement
- [x] L'estimation préalable, si elle est fournie, est enregistrée à côté du coût réel

**Vérification :** `pytest tests/costs/test_tracker.py -v`

**Dépendances :** A2

**Fichiers :** `src/etf_classifier/costs/tracker.py`, `tests/costs/test_tracker.py`

**Taille :** S

---

## A4 : rapport des coûts (CLI)

**Description :** `python -m etf_classifier.costs.report [--since] [--until]` : totaux par jour, module, opération et déploiement, écart estimation/réel, âge de la grille ; écrit `reports/azure_costs.md`.

**Critères d'acceptation :**
- [x] Les totaux d'un journal fictif correspondent aux valeurs calculées à la main
- [x] Le filtre de période fonctionne ; un journal absent donne un message clair (pas d'erreur)
- [x] Avertissement si la grille a plus de 90 jours

**Vérification :** `pytest tests/costs/test_report.py -v` ; lancement manuel sur un journal fictif

**Dépendances :** A3

**Fichiers :** `src/etf_classifier/costs/report.py`, `tests/costs/test_report.py`

**Taille :** S

---

## Point d'étape A
- [ ] Tous les tests passent sans réseau
- [ ] Vous vérifiez les prix de `config/pricing.yaml` avec votre contrat Azure

---

## B1 : dépendances, configuration et encodage des catégories

**Description :** ajouter `scikit-learn`, `openai`, `python-dotenv` ; section `features` dans `settings.yaml` (blocs actifs, poids, paramètres) ; `labels.py` pour la matrice multi-étiquettes.

**Critères d'acceptation :**
- [x] `fit_labels` sur l'entraînement, puis `transform_labels` sur le test donnent des colonnes identiques
- [x] Une catégorie absente de l'entraînement est signalée (avertissement), jamais ignorée en silence
- [x] La configuration permet d'activer ou désactiver chaque bloc

**Vérification :** `pytest tests/features/test_labels.py -v`

**Dépendances :** aucune (peut commencer après la phase A)

**Fichiers :** `requirements.txt`, `config/settings.yaml`, `src/etf_classifier/features/__init__.py`, `src/etf_classifier/features/labels.py`, `tests/features/test_labels.py`

**Taille :** S

---

## B2 : bloc `text_tfidf`

**Description :** TF-IDF sur « Name. Description » : mots et paires de mots, mots vides anglais, `sublinear_tf`, fréquence minimale de 2.

**Critères d'acceptation :**
- [x] Un mot présent uniquement dans le test n'entre pas dans le vocabulaire
- [x] Une description vide donne une ligne nulle, sans erreur
- [x] Les noms de variables sont lisibles (`text:clean energy`)

**Vérification :** `pytest tests/features/test_text.py -v`

**Dépendances :** B1

**Fichiers :** `src/etf_classifier/features/text.py`, `tests/features/test_text.py`

**Taille :** S

---

## B3 : blocs `holdings` et `composition_stats`

**Description :** vecteur creux ticker → poids pondéré par l'IDF et normalisé L2 ; statistiques de composition (couverture, nombre de lignes, 1ʳᵉ ligne, concentration des 10 premières, Herfindahl, composition vide, part de tickers inconnus), mises à l'échelle.

**Critères d'acceptation :**
- [x] Un ticker présent uniquement dans le test est ignoré, et sa part de poids apparaît dans « part de tickers inconnus »
- [x] Concentration et Herfindahl correspondent aux valeurs calculées à la main
- [x] Une composition vide ne provoque ni erreur ni valeur infinie

**Vérification :** `pytest tests/features/test_holdings.py -v`

**Dépendances :** B1

**Fichiers :** `src/etf_classifier/features/holdings.py`, `tests/features/test_holdings.py`

**Taille :** M

---

## B4 : pipeline apprendre/appliquer, sauvegarde et rechargement

**Description :** `FeaturePipeline` combine les blocs actifs avec leurs poids ; `fit` sur l'entraînement, `transform` sur tout jeu ; `save`/`load` dans `artifacts/features/<version>/`.

**Critères d'acceptation :**
- [x] Même entraînement, même matrice ; sauvegarde puis rechargement donnent une transformation identique
- [x] Un bloc désactivé disparaît de la matrice combinée sans autre changement
- [x] Les 3 blocs locaux se construisent sur 10 000 ETF synthétiques en moins de 60 s

**Vérification :** `pytest tests/features/test_pipeline.py -v`

**Dépendances :** B2, B3

**Fichiers :** `src/etf_classifier/features/pipeline.py`, `tests/features/test_pipeline.py`, `.gitignore` (ajout de `artifacts/`, `cache/`)

**Taille :** M

---

## Point d'étape B
- [ ] Tests d'absence de fuite et de reproductibilité verts
- [ ] Revue rapide avec vous des variables produites sur la fixture

---

## C1 : client d'embeddings

**Description :** `embeddings.py` : découpage en lots (64), troncature à 8 192 tokens, nouvelles tentatives sur 429 et erreurs serveur, vecteurs normalisés L2 ; client injectable, testé avec un faux client.

**Critères d'acceptation :**
- [x] 150 textes donnent 3 lots (64, 64, 22) et 150 vecteurs de 3 072 dimensions, dans l'ordre
- [x] Un texte trop long est tronqué avec un avertissement
- [x] Sans variables Azure, le bloc est désactivé avec un message clair et le pipeline continue

**Vérification :** `pytest tests/features/test_embeddings.py -v`

**Dépendances :** B4, A3

**Fichiers :** `src/etf_classifier/features/embeddings.py`, `tests/features/test_embeddings.py`, `tests/features/fake_azure.py`

**Taille :** M

---

## C2 : cache SQLite et suivi des coûts par lot

**Description :** cache `cache/embeddings.sqlite` (clé = empreinte du modèle, de la dimension et du texte) ; une ligne `CostTracker` par lot réellement envoyé.

**Critères d'acceptation :**
- [x] Un deuxième passage sur les mêmes textes ne fait aucun appel au client
- [x] Changer de dimension ne réutilise pas les vecteurs d'une autre dimension
- [x] Une ligne de coût par lot envoyé, aucune pour les textes servis par le cache

**Vérification :** `pytest tests/features/test_embeddings.py -v`

**Dépendances :** C1

**Fichiers :** `src/etf_classifier/features/embeddings.py`, `src/etf_classifier/features/cache.py`, `tests/features/test_cache.py`

**Taille :** M

---

## C3 : ligne de commande `build` et `.env.example`

**Description :** `python -m etf_classifier.features.build --csv … [--estimate-only] [--yes]` : estimation (textes hors cache, tokens, USD), confirmation, construction, sauvegarde du pipeline, résumé par bloc et coût réel.

**Critères d'acceptation :**
- [x] `--estimate-only` n'appelle jamais le client
- [x] Sans `--yes`, une réponse autre que « o » annule sans dépense
- [x] `.env.example` liste les variables nécessaires sans aucune valeur secrète

**Vérification :** `pytest tests/features/test_build.py -v` ; lancement manuel sur la fixture sans Azure

**Dépendances :** C2

**Fichiers :** `src/etf_classifier/features/build.py`, `tests/features/test_build.py`, `.env.example`, `README.md`

**Taille :** M

---

## Point d'étape C : essai réel
- [ ] Vous lancez `build --estimate-only` puis `build` sur un petit échantillon avec votre clé Azure
- [ ] Écart entre estimation et coût réel inférieur à 2 %
- [ ] Un deuxième lancement ne fait aucun appel Azure
