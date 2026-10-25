# Spec : module `similarity`

Statut : **validée** (25/10/2026)

## Objectif

Pour un ETF (nouveau ou existant), retrouver les **ETF déjà catégorisés les plus proches** et expliquer
pourquoi ils se ressemblent. Trois usages :
1. **Aider le validateur** dans l'interface : « cet ETF ressemble à X et Y, catégorisés Or et Matières premières ».
2. **Donner du contexte au LLM** (`llm-justify`) pour justifier une proposition ou traiter un cas incertain.
3. **Signaler un quasi-doublon** : un nouvel ETF qui réplique le même indice qu'un ETF existant a très probablement les mêmes catégories.

C'est aussi une seconde méthode de catégorisation, par **vote des voisins**, que l'on compare au modèle
pour l'apprentissage (deux approches, mêmes données, même jeu de test).

Hors périmètre : combiner le vote des voisins avec le modèle pour la prédiction finale (à décider après
mesure), l'interface (module `app`).

## Données d'entrée
- **Jeu de référence** : les ETF catégorisés du CSV (lignes exploitables du module `data`).
- **Pipeline de variables** du modèle courant (module `features`) : vecteurs de tickers, TF-IDF, embeddings s'ils sont actifs.
- Les nouveaux ETF à comparer, avec ou sans catégorie.

## Fonctionnalités

### 1. Deux similarités, puis un score combiné

| Similarité | Calcul | Pourquoi |
|---|---|---|
| **Composition** | Cosinus entre les vecteurs de poids des tickers | Symétrique : un petit ETF de 3 lignes contenues dans un grand ETF n'est pas jugé identique à lui, contrairement au recouvrement « rapporté à la plus petite couverture » |
| **Texte** | Cosinus entre les embeddings si le modèle les utilise, sinon entre les vecteurs TF-IDF | Rapproche les ETF de même thème, même sans titres communs |

- **Score combiné** = 0,5 × composition + 0,5 × texte (poids configurables).
- Tous les voisins d'une même recherche sont notés sur la même échelle :
  - si l'**ETF recherché** n'a pas de composition, tous les voisins sont notés sur le texte seul, et c'est indiqué ;
  - si seul le **voisin** n'a pas de composition, sa similarité de composition vaut 0 dans le score combiné.
  *(Correction du 25/10/2026 : la règle initiale, « texte seul si l'un des deux n'a pas de composition », plaçait
  un ETF sans composition devant des ETF partageant les mêmes tickers, car son score n'était pas sur la même échelle.)*

### 2. Recherche des voisins
- `SimilarityIndex.build(reference_df, pipeline)` construit l'index ; `index.query(df, k=5)` renvoie pour chaque ETF ses `k` voisins.
- Pour chaque voisin :
  - nom et catégories ;
  - score combiné, similarité de composition, similarité de texte ;
  - **part du poids en commun** (recouvrement pondéré, en %) et les 5 principaux tickers partagés, pour expliquer le rapprochement de façon lisible.
- Un ETF déjà présent dans la référence n'est jamais son propre voisin.
- **Alerte quasi-doublon** : même règle que le module `data` (recouvrement ≥ 0,80 et au moins 3 tickers communs).

### 3. Vote des voisins
- `neighbor_vote(neighbors)` donne pour chaque catégorie un score : somme des scores des voisins qui l'ont, divisée par la somme de tous les scores.
- Évaluation : `python -m etf_classifier.similarity.evaluate --csv …` construit l'index **sur l'entraînement seulement**, fait voter les voisins pour chaque ETF de test, et compare au modèle courant et à la référence triviale (F1 macro et micro, pour k = 5, 10 et 20). On saura ainsi si le vote aide, notamment sur les catégories rares.

### 4. Construction, sauvegarde, commande
- L'index est construit par `model.train` sur toutes les lignes exploitables (entraînement + test), puis sauvegardé **dans le dossier du modèle** (`similarity.joblib`). Le modèle et son index restent toujours cohérents.
- `python -m etf_classifier.similarity.search --csv nouveaux_etfs.csv [--model current] [--k 5]` écrit un CSV des voisins (un ETF × un voisin par ligne) et un résumé Markdown.
- Les embeddings des nouveaux ETF passent par le cache et la confirmation de coût du module `features`.

## Tech Stack
Aucune nouvelle dépendance (numpy, scipy, scikit-learn).

## Structure
```
src/etf_classifier/similarity/
├── index.py       # SimilarityIndex : construction, recherche, explication, sauvegarde
├── vote.py        # vote des voisins
├── search.py      # CLI de recherche
└── evaluate.py    # CLI d'évaluation (vote vs modèle vs référence triviale)
tests/similarity/
```
Modification d'un module existant : `model.train` construit et sauvegarde l'index dans le dossier du modèle ; `model.registry` le recharge.

## Stratégie de test
- Deux ETF de composition identique : similarité de composition égale à 1 et alerte quasi-doublon.
- Un petit ETF contenu dans un grand : similarité de composition nettement inférieure à 1, alors que le recouvrement vaut 100 %.
- Un ETF existant n'est jamais son propre voisin ; les voisins sont triés par score décroissant.
- Composition vide : seul le texte compte, avec une mention.
- Vote des voisins sur le jeu synthétique : nettement meilleur que la référence triviale.
- Sauvegarde puis rechargement : mêmes voisins, mêmes scores.
- Performance : 100 requêtes sur une référence de 10 000 ETF en moins de 10 s.

## Limites
- **Toujours :** expliquer chaque voisin (part du poids en commun, tickers partagés) ; exclure l'ETF lui-même ; évaluer le vote avec un index construit sur l'entraînement seul.
- **Demander d'abord :** utiliser le vote pour modifier les prédictions du modèle ; changer la mesure de similarité.
- **Jamais :** envoyer à Azure autre chose que le nom et la description.

## Critères de succès
1. Pour un ETF de la fixture, les voisins proposés ont des catégories cohérentes, avec une explication lisible.
2. Un quasi-doublon d'un ETF existant est signalé.
3. Le rapport d'évaluation chiffre le vote des voisins face au modèle et à la référence triviale.
4. Les tests passent sans réseau ; les performances sont tenues.

## Décisions prises
- Poids du score combiné : **0,5 composition / 0,5 texte** au départ, à ajuster après mesure du vote sur le test.
- **5 voisins** affichés par défaut.
- Le vote des voisins est d'abord **limité à une évaluation** ; son usage pour compléter le modèle (cas incertains) sera décidé après mesure.
