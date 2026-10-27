# Spec: `features` module

Status: **approved** (2026-10-25)

> **Amended on 2026-10-27** by [CHANGE-input-format](CHANGE-input-format.md): input CSV `Ticker;Nom;Description;Classification;Composition`, one category per ETF, ticker as the ETF identifier. Where this spec and the change spec differ, the change spec prevails.

## Objective

Turn every ETF (name, description, weighted composition) and its categories into numerical
matrices usable by the `model` and `similarity` modules, in a **reproducible** way and **without leaks**:
the vocabulary, the weightings and the scalings are learned on the training set only,
then applied as they are to the test set and to new ETFs.

The module is organised in **feature blocks** that can be enabled separately. This makes it possible to
measure the contribution of each one: first a baseline without Azure, then the addition of the embeddings.

## Input data
- `PreparedDataset` from the `data` module: columns `name`, `description`, `composition` (normalised tickers), `categories`, `coverage`, `n_holdings`, plus the train/test split.
- Descriptions in **English**.
- Azure OpenAI access: `text-embedding-3-large` deployment. Endpoint, key and API version are read from environment variables (`.env` file, not committed). **The key never goes through the code, the logs or the repository.**

## Feature blocks

| Block | Content | Azure | Learning purpose |
|---|---|---|---|
| `text_tfidf` | TF-IDF of the name and description: words and word pairs, English stop words removed, `sublinear_tf`, minimum frequency of 2 documents | No | Simple, interpretable baseline |
| `holdings` | Sparse ticker → weight vector (in %), weighted by ticker rarity (IDF), L2-normalised | No | The model learns which tickers are typical of a category |
| `composition_stats` | Coverage, number of holdings, weight of the top holding, top-10 concentration, Herfindahl index, empty composition, share of the weight on tickers unknown to the vocabulary | No | Distinguishes a broad ETF from a thematic or concentrated one |
| `embeddings` | `text-embedding-3-large` embedding of the text "Name. Description" (3,072 dimensions), L2-normalised | Yes | Meaning of the text beyond exact words |

- Active blocks and their relative weight are defined in `config/settings.yaml` (`features` section).
- Targets are encoded as a multi-label matrix (one column per category). A category unknown at training time is reported, never silently ignored.

## Features

### 1. Fit / transform pipeline
- `FeaturePipeline(settings).fit(train_df)` learns vocabularies, IDF and scalings **on training only**.
- `transform(df) -> FeatureSet` holds one matrix per block, the combined (sparse) matrix and the list of feature names.
- `fit_labels(train_df)` / `transform_labels(df)` produce the category matrix and the list of categories.
- `save(path)` / `load(path)` in `artifacts/features/<version>/`, with the configuration used and a fingerprint of the training set, so that daily retraining and the application use exactly the same transformation.

### 2. Azure embeddings
- `AzureOpenAI` client (official `openai` library), configurable text batches (64 by default), automatic retries on rate limits or server errors.
- Texts that are too long are truncated to the model limit (8,192 tokens), with a warning.
- SQLite **cache** `cache/embeddings.sqlite` (standard library), key = fingerprint of (model, dimensions, text). A text already computed is never sent to Azure again: daily retraining only pays for new ETFs.
- **Costs**: before a computation, `costs.estimate_embeddings` gives the number of texts to compute (excluding the cache), the tokens and the cost. After each batch, `CostTracker.record(module="features", operation="embeddings_batch", …)`.
- **Without Azure** (environment variables missing or block disabled): the `embeddings` block is skipped with a clear message, and the other blocks work. Tests use a fake client, without network.

### 3. Command line
```
python -m etf_classifier.features.build --csv data/raw/etfs.csv --estimate-only
python -m etf_classifier.features.build --csv data/raw/etfs.csv            # asks for confirmation before paying
python -m etf_classifier.features.build --csv data/raw/etfs.csv --yes      # no confirmation (daily job)
```
Displays for each block the dimension and density, the number of embeddings taken from the cache or computed, and the estimated then actual cost.

## Tech Stack
- Existing: pandas, numpy, scipy, PyYAML, pytest
- **New dependencies**:
  - `scikit-learn` (TF-IDF, scaling, multi-label encoding, saving via `joblib`)
  - `openai` (official Azure client)
  - `python-dotenv` (reading the `.env` file)
  - `tiktoken` (through the `costs` module)

## Structure
```
.env.example                          # template of the Azure variables, no secret value (committed)
src/etf_classifier/features/
├── text.py                           # text_tfidf block
├── holdings.py                       # holdings and composition_stats blocks
├── embeddings.py                     # Azure client, batches, cache, costs
├── labels.py                         # category encoding
├── pipeline.py                       # FeaturePipeline, save/load
└── build.py                          # command line
cache/                                # embeddings cache (not committed)
artifacts/                            # saved pipelines (not committed)
tests/features/
```

## Code style
Same as the previous modules. Every block exposes the same interface (`fit`, `transform`, `feature_names`), so that one can be added without touching the pipeline.

## Test strategy
- **No leak**: a word or ticker present only in the test set does not enter the vocabulary.
- **Reproducibility**: same training, same matrix; saving then reloading gives an identical transformation.
- **Composition features** computed by hand on simple cases (concentration, Herfindahl, empty composition, share of unknown tickers).
- **Embeddings with a fake client**: batching, cache (second pass without any call), truncation, one cost line recorded per batch.
- **Without-Azure mode**: the pipeline works with the other three blocks.
- **Performance**: on 10,000 synthetic ETFs, building the blocks without Azure takes less than 60 s.

## Boundaries
- **Always:** learn on training only; go through the cache and through `costs` for every Azure call; ask for confirmation before an expense, except with `--yes`.
- **Ask first:** changing the embeddings model or dimension (this invalidates the cache and the trained models); adding a feature block.
- **Never:** commit `.env`, `cache/`, `artifacts/`; write a key in the code or a log; send Azure anything other than the name and the description.

## Success criteria
1. On the real CSV, `build --estimate-only` gives the cost of the full embedding before any expense.
2. A second `build` on the same data makes no Azure call and costs nothing.
3. `transform` on the test set only uses what was learned on training (no-leak test).
4. Tests pass without network; the pipeline works without Azure with the three local blocks.

## Decisions made
- Embeddings in **3,072 dimensions** (full precision).
- Text sent: **"Name. Description"**.
- **The category is never included** in the embedded text: it is the value to predict. Including it would create a leak
  (the model would read the answer at training time, whereas a new ETF has no category yet).
- **The composition is not included by default**: it is already carried by the `holdings` and `composition_stats` blocks,
  tickers carry little meaning for a language model, and every rebalancing would change the text, hence
  force the embedding to be recomputed and paid again. An `embeddings.include_top_tickers` option (disabled) will make it possible to
  measure later whether adding the top 10 tickers improves the score.
