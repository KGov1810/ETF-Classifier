# Spec : module `features`

Statut : **validée** (25/10/2026)

## Objectif

Transformer chaque ETF (nom, description, composition pondérée) et ses catégories en matrices
numériques utilisables par les modules `model` et `similarity`, de façon **reproductible** et **sans fuite** :
le vocabulaire, les pondérations et les mises à l'échelle sont appris sur le jeu d'entraînement seulement,
puis appliqués tels quels au jeu de test et aux nouveaux ETF.

Le module est organisé en **blocs de variables** activables séparément. On peut ainsi mesurer l'apport
de chacun : d'abord une référence sans Azure, puis l'ajout des embeddings.

## Données d'entrée
- `PreparedDataset` du module `data` : colonnes `name`, `description`, `composition` (tickers normalisés), `categories`, `coverage`, `n_holdings`, plus le découpage entraînement/test.
- Descriptions en **anglais**.
- Accès Azure OpenAI : déploiement `text-embedding-3-large`. Endpoint, clé et version d'API sont lus depuis des variables d'environnement (fichier `.env` non versionné). **La clé ne transite jamais par le code, les journaux ou le dépôt.**

## Blocs de variables

| Bloc | Contenu | Azure | Rôle pédagogique |
|---|---|---|---|
| `text_tfidf` | TF-IDF du nom et de la description : mots et paires de mots, mots vides anglais retirés, `sublinear_tf`, fréquence minimale de 2 documents | Non | Référence simple et interprétable |
| `holdings` | Vecteur creux ticker → poids (en %), pondéré par la rareté du ticker (IDF), normalisé L2 | Non | Le modèle apprend quels tickers sont typiques d'une catégorie |
| `composition_stats` | Couverture, nombre de lignes, poids de la 1ʳᵉ ligne, concentration des 10 premières, indice de Herfindahl, composition vide, part du poids sur des tickers inconnus du vocabulaire | Non | Distingue un ETF large d'un ETF thématique ou concentré |
| `embeddings` | Embedding `text-embedding-3-large` du texte « Name. Description » (3 072 dimensions), normalisé L2 | Oui | Sens du texte au-delà des mots exacts |

- Les blocs actifs et leur poids relatif sont définis dans `config/settings.yaml` (section `features`).
- Les cibles sont encodées en matrice multi-étiquettes (une colonne par catégorie). Une catégorie inconnue au moment de l'apprentissage est signalée, jamais ignorée en silence.

## Fonctionnalités

### 1. Pipeline apprendre / appliquer
- `FeaturePipeline(settings).fit(train_df)` apprend vocabulaires, IDF et mises à l'échelle **sur l'entraînement uniquement**.
- `transform(df) -> FeatureSet` contient une matrice par bloc, la matrice combinée (creuse) et la liste des noms de variables.
- `fit_labels(train_df)` / `transform_labels(df)` produisent la matrice des catégories et la liste des catégories.
- `save(path)` / `load(path)` dans `artifacts/features/<version>/`, avec la configuration utilisée et une empreinte du jeu d'entraînement, pour que le réentraînement quotidien et l'application utilisent exactement la même transformation.

### 2. Embeddings Azure
- Client `AzureOpenAI` (bibliothèque officielle `openai`), lots de textes configurables (64 par défaut), nouvelles tentatives automatiques en cas de limite de débit ou d'erreur serveur.
- Textes trop longs tronqués à la limite du modèle (8 192 tokens), avec un avertissement.
- **Cache** SQLite `cache/embeddings.sqlite` (bibliothèque standard), clé = empreinte de (modèle, dimensions, texte). Un texte déjà calculé n'est jamais renvoyé à Azure : le réentraînement quotidien ne paie que les nouveaux ETF.
- **Coûts** : avant un calcul, `costs.estimate_embeddings` donne le nombre de textes à calculer (hors cache), les tokens et le coût. Après chaque lot, `CostTracker.record(module="features", operation="embeddings_batch", …)`.
- **Sans Azure** (variables d'environnement absentes ou bloc désactivé) : le bloc `embeddings` est ignoré avec un message clair, et les autres blocs fonctionnent. Les tests utilisent un faux client, sans réseau.

### 3. Ligne de commande
```
python -m etf_classifier.features.build --csv data/raw/etfs.csv --estimate-only
python -m etf_classifier.features.build --csv data/raw/etfs.csv            # demande confirmation avant de payer
python -m etf_classifier.features.build --csv data/raw/etfs.csv --yes      # sans confirmation (job quotidien)
```
Affiche pour chaque bloc la dimension et la densité, le nombre d'embeddings tirés du cache ou calculés, et le coût estimé puis réel.

## Tech Stack
- Existant : pandas, numpy, scipy, PyYAML, pytest
- **Nouvelles dépendances** :
  - `scikit-learn` (TF-IDF, mise à l'échelle, encodage multi-étiquettes, sauvegarde via `joblib`)
  - `openai` (client Azure officiel)
  - `python-dotenv` (lecture du fichier `.env`)
  - `tiktoken` (via le module `costs`)

## Structure
```
.env.example                          # modèle des variables Azure, sans valeur secrète (versionné)
src/etf_classifier/features/
├── text.py                           # bloc text_tfidf
├── holdings.py                       # blocs holdings et composition_stats
├── embeddings.py                     # client Azure, lots, cache, coûts
├── labels.py                         # encodage des catégories
├── pipeline.py                       # FeaturePipeline, sauvegarde/chargement
└── build.py                          # ligne de commande
cache/                                # cache d'embeddings (non versionné)
artifacts/                            # pipelines sauvegardés (non versionnés)
tests/features/
```

## Style de code
Identique aux modules précédents. Chaque bloc expose la même interface (`fit`, `transform`, `feature_names`), pour pouvoir en ajouter un sans toucher au pipeline.

## Stratégie de test
- **Absence de fuite** : un mot ou un ticker présent uniquement dans le test n'entre pas dans le vocabulaire.
- **Reproductibilité** : même entraînement, même matrice ; sauvegarde puis rechargement donnent une transformation identique.
- **Variables de composition** calculées à la main sur des cas simples (concentration, Herfindahl, composition vide, part de tickers inconnus).
- **Embeddings avec un faux client** : découpage en lots, cache (deuxième passage sans aucun appel), troncature, enregistrement d'une ligne de coût par lot.
- **Mode sans Azure** : le pipeline fonctionne avec les trois autres blocs.
- **Performance** : sur 10 000 ETF synthétiques, la construction des blocs sans Azure prend moins de 60 s.

## Limites
- **Toujours :** apprendre sur l'entraînement seulement ; passer par le cache et par `costs` pour tout appel Azure ; demander confirmation avant une dépense, sauf avec `--yes`.
- **Demander d'abord :** changer de modèle d'embeddings ou de dimension (cela invalide le cache et les modèles entraînés) ; ajouter un bloc de variables.
- **Jamais :** versionner `.env`, `cache/`, `artifacts/` ; écrire une clé dans le code ou un journal ; envoyer à Azure autre chose que le nom et la description.

## Critères de succès
1. Sur le CSV réel, `build --estimate-only` donne le coût de l'embedding complet avant toute dépense.
2. Un deuxième `build` sur les mêmes données ne fait aucun appel Azure et ne coûte rien.
3. `transform` sur le jeu de test n'utilise que ce qui a été appris sur l'entraînement (test d'absence de fuite).
4. Les tests passent sans réseau ; le pipeline fonctionne sans Azure avec les trois blocs locaux.

## Décisions prises
- Embeddings en **3 072 dimensions** (pleine précision).
- Texte envoyé : **« Name. Description »**.
- **La catégorie n'est jamais incluse** dans le texte embarqué : c'est la valeur à prédire. L'inclure créerait une fuite
  (le modèle lirait la réponse à l'entraînement, alors qu'un nouvel ETF n'a pas encore de catégorie).
- **La composition n'est pas incluse par défaut** : elle est déjà portée par les blocs `holdings` et `composition_stats`,
  des tickers apportent peu de sens à un modèle de langage, et chaque rebalancement changerait le texte, donc
  ferait recalculer et repayer l'embedding. Une option `embeddings.include_top_tickers` (désactivée) permettra de
  mesurer plus tard si l'ajout des 10 premiers tickers améliore le score.
