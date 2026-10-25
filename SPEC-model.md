# Spec : module `model`

Statut : **validée** (25/10/2026)

## Objectif

Entraîner, évaluer et versionner le **classifieur multi-étiquettes** qui propose une ou plusieurs
catégories pour chaque ETF, avec un score de confiance par catégorie.

C'est le cœur pédagogique du projet. On avance par comparaisons mesurées :
1. une **référence triviale** (prédire les catégories les plus fréquentes), qui sert de plancher ;
2. un **modèle de référence** sur les trois blocs locaux (TF-IDF, tickers, statistiques) ;
3. le même modèle **avec les embeddings Azure**, pour mesurer leur apport réel.

Utilisateurs : les modules `app` (propositions à valider), `llm-justify` (cas incertains), `retraining`
(entraînement quotidien et comparaison), et vous, via des rapports d'évaluation lisibles.

Hors périmètre : la règle de promotion automatique d'un nouveau modèle (module `retraining`),
l'interface (module `app`), les modèles de deep learning.

## Données d'entrée
- `PreparedDataset` (module `data`) : découpage entraînement/test par groupe de quasi-doublons.
- `FeaturePipeline` (module `features`) : matrices de variables et matrice des catégories.

## Fonctionnalités

### 1. Modèle
- **Une régression logistique par catégorie** (stratégie « un contre tous »), sur la matrice combinée des blocs actifs.
  - Choix justifié : rapide sur des matrices creuses de dizaines de milliers de colonnes, probabilités exploitables, coefficients interprétables (on peut montrer quels mots ou tickers poussent vers une catégorie).
  - `class_weight="balanced"` pour ne pas sacrifier les catégories rares.
- **Réglage sur un jeu de validation interne** : le jeu d'entraînement est redécoupé (80/20, par groupe de quasi-doublons) pour :
  - choisir la force de régularisation `C` dans une petite grille (0,1 · 1 · 10) ;
  - fixer un **seuil de décision par catégorie** qui maximise son F1 (0,5 par défaut si la catégorie a trop peu d'exemples en validation).
- Le modèle final est ensuite réentraîné sur tout le jeu d'entraînement, avec les paramètres et seuils retenus.
- **Le jeu de test n'est utilisé qu'une fois**, pour l'évaluation finale. Il ne sert jamais au réglage.
- Catégorie sans aucun exemple positif à l'entraînement : non apprenable, signalée dans le rapport.

### 2. Prédiction
- `predict(df) -> list[Prediction]`. Pour chaque ETF :
  - les **scores de toutes les catégories** (probabilités), triés ;
  - les **catégories retenues**, celles qui dépassent leur seuil ;
  - **au moins une catégorie** retenue, la meilleure, même si aucune ne dépasse son seuil, puisque tout ETF a au moins une catégorie ;
  - un indicateur **« incertain »** si le meilleur score est sous un seuil de confiance (configurable, 0,5 par défaut). C'est le signal pour l'interface et pour `llm-justify` : cas difficile ou catégorie nouvelle possible.
- `python -m etf_classifier.model.predict --model current --csv nouveaux_etfs.csv` écrit un CSV de propositions (catégories retenues, scores, indicateur incertain). Utile avant l'arrivée de l'interface.

### 3. Évaluation
Calculée sur le jeu de test et écrite dans `reports/model_<version>.md` :
- **F1 micro** (chaque décision ETF × catégorie compte autant : favorise les catégories fréquentes) ;
- **F1 macro** (chaque catégorie compte autant : révèle les catégories rares mal apprises) ;
- **F1 par ETF** (« samples »), exactitude stricte (toutes les catégories justes), Hamming loss ;
- **par catégorie** : précision, rappel, F1, effectif, seuil retenu ;
- **comparaison à la référence triviale**, sur le même jeu de test ;
- **variables les plus influentes par catégorie** (10 coefficients les plus positifs) : le rapport montre par exemple que « gold », « NEM » et « miners » poussent vers « Or ».

### 4. Expériences comparées
`python -m etf_classifier.model.compare --csv … --experiments local,local+embeddings` entraîne plusieurs
configurations de blocs **sur le même découpage** et produit un tableau comparatif
(`reports/model_comparison.md`) : F1 micro et macro, temps d'entraînement et coût Azure de chaque configuration.

### 5. Versionnement
- Chaque entraînement produit `artifacts/models/<version>/` contenant :
  - le modèle et **son propre pipeline de variables** (un seul dossier à charger pour prédire) ;
  - `metrics.json` (métriques, paramètres, seuils, empreinte du jeu d'entraînement, blocs utilisés) ;
  - une fiche lisible `MODEL_CARD.md`.
- `artifacts/models/registry.json` indique le modèle **courant** (`current`) et l'historique.
- `train --promote` désigne le nouveau modèle comme courant. Le tout premier modèle l'est automatiquement. La promotion automatique conditionnelle viendra avec `retraining`.

### 6. Ligne de commande d'entraînement
```
python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure          # référence locale
python -m etf_classifier.model.train --csv data/raw/etfs.csv --yes --promote     # avec embeddings
```
Les embeddings passent par la confirmation de coût du module `features`, sauf avec `--yes`.

## Tech Stack
Aucune nouvelle dépendance : scikit-learn, numpy, scipy, pandas, joblib (déjà présents).

## Structure
```
src/etf_classifier/model/
├── classifier.py      # entraînement, réglage de C, seuils par catégorie, prédiction
├── baseline.py        # référence triviale (catégories les plus fréquentes)
├── evaluate.py        # métriques et rapport d'évaluation
├── explain.py         # variables les plus influentes par catégorie
├── registry.py        # sauvegarde, chargement, modèle courant
├── train.py           # CLI d'entraînement
├── compare.py         # CLI de comparaison d'expériences
└── predict.py         # CLI de prédiction sur un CSV
tests/model/
```

## Style de code
Identique aux modules précédents. Graine aléatoire fixe partout (`random_state` issu de `settings.yaml`).

## Stratégie de test
- **Jeu synthétique où la réponse est connue** (catégories déterminées par certains mots et tickers) : le modèle doit nettement battre la référence triviale.
- **Seuils** : sur des scores construits à la main, le seuil choisi maximise bien le F1 de la catégorie.
- **Règle « au moins une catégorie »** et indicateur « incertain ».
- **Non-utilisation du test pour le réglage** : la fonction de réglage ne reçoit que le jeu d'entraînement (test de signature et de comportement).
- **Reproductibilité** : même données et même graine, mêmes prédictions.
- **Registre** : sauvegarde, rechargement, prédictions identiques ; désignation du modèle courant.
- **Performance** : entraînement sur 10 000 ETF synthétiques (blocs locaux) en moins de 5 minutes.

## Limites
- **Toujours :** régler sur la validation interne, évaluer une seule fois sur le test ; comparer à la référence triviale ; sauvegarder modèle et pipeline ensemble.
- **Demander d'abord :** changer d'algorithme ou ajouter une dépendance ; modifier la métrique principale ; changer la règle « au moins une catégorie ».
- **Jamais :** utiliser le jeu de test pour choisir un paramètre ou un seuil ; versionner `artifacts/`.

## Critères de succès
1. Sur votre CSV réel, le modèle de référence local bat nettement la référence triviale en F1 micro et macro.
2. Le rapport de comparaison chiffre l'apport (ou non) des embeddings.
3. `predict` produit pour chaque ETF au moins une catégorie, des scores et l'indicateur « incertain ».
4. Un modèle sauvegardé se recharge et donne exactement les mêmes prédictions.
5. Les tests passent sans réseau.

Un **objectif chiffré** (par exemple F1 micro ≥ 0,85) ne peut pas être fixé honnêtement sans vos données. Je propose de le définir avec vous après le premier entraînement sur le CSV réel.

## Décisions prises
- **Métrique principale : F1 macro** (chaque catégorie compte autant), pour le choix de `C` et la comparaison des modèles ; le F1 micro reste affiché.
- **5 catégories proposées au maximum** par ETF ; les scores des autres catégories restent visibles.
- **Seuil de l'indicateur « incertain » : 0,5** au départ, à ajuster après le premier entraînement sur le CSV réel.
