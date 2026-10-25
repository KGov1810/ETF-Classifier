# Tâches : module `data`

Définition de « terminé » pour chaque tâche : tests verts (`pytest tests/data -v`), aucune régression,
code conforme au style de `SPEC-data.md`, aucun commit sans votre accord.

---

## Tâche 1 : squelette du projet et configuration

**Description :** créer l'arborescence de la spec, l'environnement Python, la configuration et un CSV fictif de test couvrant les cas limites.

**Critères d'acceptation :**
- [x] `pip install -r requirements.txt` puis `pytest` s'exécutent sans erreur
- [x] `config/settings.yaml` contient les noms de colonnes, le séparateur `;`, l'encodage et tous les seuils de la spec
- [x] `tests/data/fixtures/sample.csv` (≈ 20 lignes fictives) couvre chaque règle du tableau de qualité

**Vérification :** `pytest -q` (0 test en échec) ; relecture de `settings.yaml`

**Dépendances :** aucune

**Fichiers :** `requirements.txt`, `pyproject.toml`, `.gitignore`, `config/settings.yaml`, `src/etf_classifier/__init__.py`, `src/etf_classifier/config.py`, `tests/data/fixtures/sample.csv`

**Taille :** M

---

## Tâche 2 : parsing de Composition/Category et `load_dataset()` minimal

**Description :** lire le CSV et convertir Composition en `dict[str, float]` et Category en `list[str]` avec `ast.literal_eval`. `load_dataset()` renvoie le DataFrame normalisé (`row_id`, `name`, `description`, `composition`, `categories`).

**Critères d'acceptation :**
- [x] `[{'AAPL': 6.5}, {'MSFT': 6.1}]` donne `{'AAPL': 6.5, 'MSFT': 6.1}` ; un ticker présent deux fois voit ses poids additionnés
- [x] Une cellule illisible lève une `ParsingError` avec un extrait de la cellule
- [x] `load_dataset()` charge la fixture avec le séparateur `;` et l'encodage UTF-8

**Vérification :** `pytest tests/data/test_parsing.py tests/data/test_loader.py -v`

**Dépendances :** tâche 1

**Fichiers :** `src/etf_classifier/data/parsing.py`, `src/etf_classifier/data/loader.py`, `tests/data/test_parsing.py`, `tests/data/test_loader.py`

**Taille :** M

---

## Tâche 3 : règles de qualité, exclusions motivées, colonne `coverage`

**Description :** appliquer les règles du tableau de qualité. Les lignes invalides sont exclues du jeu exploitable et consignées avec leur raison, les avertissements sont tracés, et la couverture (somme des poids) est calculée.

**Critères d'acceptation :**
- [x] Chaque règle de la spec a au moins un test (exclusion ou avertissement)
- [x] `load_dataset()` renvoie le jeu valide et un objet `QualityLog` (row_id, règle, gravité, extrait)
- [x] La colonne `coverage` est présente ; une somme > 105 génère un avertissement

**Vérification :** `pytest tests/data/test_validation.py -v`

**Dépendances :** tâche 2

**Fichiers :** `src/etf_classifier/data/validation.py`, `src/etf_classifier/data/loader.py`, `tests/data/test_validation.py`

**Taille :** S

---

## Tâche 4 : rapport de qualité minimal (CLI + Markdown)

**Description :** commande `python -m etf_classifier.data.report --csv <chemin>` qui affiche et écrit dans `reports/data_quality.md` le nombre de lignes valides, en avertissement et exclues (avec raisons), le nombre de catégories, l'effectif par catégorie et le nombre moyen de catégories par ETF.

**Critères d'acceptation :**
- [x] La commande fonctionne sur la fixture et produit le fichier Markdown
- [x] Les chiffres du rapport correspondent à ceux attendus pour la fixture (test)
- [x] Code d'erreur clair si le fichier ou une colonne est absent

**Vérification :** `pytest tests/data/test_report.py -v` ; lancement manuel sur la fixture

**Dépendances :** tâche 3

**Fichiers :** `src/etf_classifier/data/report.py`, `tests/data/test_report.py`

**Taille :** S

---

## Point d'étape 1 : premier passage sur le CSV réel
- [x] Tous les tests passent
- [ ] Rapport généré sur le CSV réel
- [ ] Revue avec vous : taux d'exclusion, formats inattendus, ajustements éventuels de la spec avant de continuer

---

## Tâche 5 : normalisation des tickers et fichier d'alias

**Description :** clé normalisée par ticker (majuscules, espaces) ; suffixes de place conservés, mais retirables par option ; application de `config/holding_aliases.csv` ; détection des tickers ressemblants pour le rapport.

**Critères d'acceptation :**
- [x] `' aapl '` → `'AAPL'` ; `'AAPL US'` reste distinct de `'AAPL'` avec l'option désactivée, et fusionne avec l'option activée
- [x] Un alias `GOOG;GOOGL` est appliqué ; le ticker d'origine reste accessible
- [x] Les paires de tickers ressemblants (même racine, suffixe différent) sont listées

**Vérification :** `pytest tests/data/test_normalize.py -v`

**Dépendances :** tâche 2

**Fichiers :** `src/etf_classifier/data/normalize.py`, `config/holding_aliases.csv`, `tests/data/test_normalize.py`

**Taille :** S

---

## Tâche 6 : détection des quasi-doublons

**Description :** recouvrement `Σ min(wA, wB) / min(ΣwA, ΣwB)` calculé par index inversé et matrices creuses ; deux ETF sont liés si le recouvrement est ≥ 0,80 et s'ils ont au moins 3 tickers communs ; les groupes sont formés par composantes connexes (`group_id`).

**Critères d'acceptation :**
- [x] Deux trackers fictifs du même indice, l'un complet et l'autre limité aux 10 premières lignes, sont dans le même groupe
- [x] Deux ETF partageant une seule grosse ligne ne sont pas regroupés
- [x] 10 000 ETF synthétiques sont traités en moins de 60 s ; un avertissement est émis si un groupe dépasse 2 % du jeu

**Vérification :** `pytest tests/data/test_dedup.py -v` (le test de performance est marqué `@pytest.mark.slow`)

**Dépendances :** tâches 3 et 5

**Fichiers :** `src/etf_classifier/data/dedup.py`, `tests/data/test_dedup.py`

**Taille :** M

---

## Point d'étape 2
- [x] Tous les tests passent, y compris le test de performance
- [ ] Revue avec vous des 10 plus gros groupes de quasi-doublons sur le CSV réel ; seuil confirmé ou ajusté

---

## Tâche 7 : découpage entraînement/test par groupe

**Description :** `make_split(df, test_size=0.2, seed=42)`. Un groupe n'est jamais coupé en deux ; on cherche à couvrir des deux côtés chaque catégorie ayant au moins 2 groupes ; les catégories sous le seuil de rareté sont listées.

**Critères d'acceptation :**
- [x] Aucun `group_id` n'est présent à la fois dans l'entraînement et dans le test
- [x] Même graine, même découpage (test de reproductibilité)
- [x] Les catégories rares et les catégories absentes du test sont renvoyées dans un résumé

**Vérification :** `pytest tests/data/test_split.py -v`

**Dépendances :** tâche 6

**Fichiers :** `src/etf_classifier/data/split.py`, `tests/data/test_split.py`

**Taille :** S

---

## Tâche 8 : rapport complet et contrôle de non-modification du CSV source

**Description :** ajouter au rapport les groupes de quasi-doublons (nombre, plus gros groupe), les catégories rares, le résumé du découpage, les tickers ressemblants et les tickers les plus fréquents. Vérifier par empreinte (hash) que le CSV source n'est jamais modifié.

**Critères d'acceptation :**
- [x] Le rapport contient toutes les sections de la spec
- [x] Un test vérifie que l'empreinte du CSV est identique avant et après une exécution complète
- [x] Le rapport complet sur 10 000 ETF prend moins de 2 minutes

**Vérification :** `pytest tests/data -v` ; lancement manuel sur le CSV réel

**Dépendances :** tâches 4, 5, 6 et 7

**Fichiers :** `src/etf_classifier/data/report.py`, `tests/data/test_report.py`, `tests/data/test_integrity.py`

**Taille :** M

---

## Point d'étape final
- [ ] Les 4 critères de succès de `SPEC-data.md` sont remplis
- [x] `load_dataset()` et `make_split()` sont utilisables directement par le module `features`
- [ ] Revue avec vous avant de passer à la spec `features`
