# ETF Classifier

A **multi-label ETF categorisation** tool built with machine learning: from a dataset of
5,000 to 10,000 already-categorised ETFs, it proposes one or more categories for every new ETF.
A team member validates or corrects the proposal, and the validated ETF enriches the dataset.
The model is retrained once a day.

The project is also a machine-learning learning ground: it is built step by step, each step
measured against a simple baseline.

## Progress

| Module | Role | Status |
|---|---|---|
| `data` | Loading, quality checks, normalisation, near-duplicates, split | ✅ Done |
| `costs` | Azure OpenAI cost estimation and tracking (USD) | ✅ Done |
| `features` | Vectorisation of the text and composition, Azure embeddings | ✅ Done |
| `model` | Multi-label classifier, evaluation, versioning | ✅ Done |
| `similarity` | Closest ETFs, near-duplicate alert, neighbour vote | ✅ Done |
| `feedback` | Validation journal, category registry, import and monitoring | ✅ Done |
| `app` | Streamlit interface | — |
| `llm-justify` | LLM justification (Azure OpenAI) | — |
| `retraining` | Daily retraining | — |

See `CAPABILITY-MAP.md` for the architecture and the build order, and the `SPEC-<module>.md` files for the details of each module.

## Expected data format

UTF-8 CSV, `;` separator, with these columns:

| Column | Example |
|---|---|
| `Name` | `iShares Core MSCI World` |
| `Description` | `Tracks the MSCI World index…` |
| `Composition` | `[{'AAPL': 5.1}, {'MSFT': 4.6}, {'NVDA': 4.2}]` (tickers and weights in %) |
| `Category` | `['Global Equity', 'ESG']` |

Cells containing a `;` (for example a description) must be quoted.
The composition may be partial (top holdings only).

## Installation

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Put the CSV in `data/raw/` (this folder is never committed), then generate the quality report:

```bash
python -m etf_classifier.data.report --csv data/raw/etfs.csv
```

The report is printed and written to `reports/data_quality.md`. It shows the excluded rows and
their reason, the categories, the composition coverage, the tickers to harmonise, the
near-duplicate groups and the train/test split.

From Python, for the downstream modules:

```python
from etf_classifier.data.loader import prepare_dataset

prepared = prepare_dataset("data/raw/etfs.csv")
prepared.split.train   # training ETFs
prepared.split.test    # test ETFs (no near-duplicate shared with training)
prepared.log           # exclusions and warnings
```

## Azure OpenAI access

Copy `.env.example` to `.env` (never committed) and fill in the endpoint, the key, the API version
and the name of the embeddings deployment. The deployment name must also appear in
`config/pricing.yaml` (`deployments` section). **Check the prices of this price list** against your
contract: they are pre-filled with the public Data Zone Standard prices (upper end).

Without `.env`, everything works with the three local blocks; only the embeddings block is disabled.

## Building the features

```bash
python -m etf_classifier.features.build --csv data/raw/etfs.csv --estimate-only   # estimated cost, no call
python -m etf_classifier.features.build --csv data/raw/etfs.csv                   # asks for confirmation
python -m etf_classifier.features.build --csv data/raw/etfs.csv --yes             # no confirmation (daily job)
python -m etf_classifier.features.build --csv data/raw/etfs.csv --no-azure        # local blocks only
```

Embeddings are cached (`cache/embeddings.sqlite`): an ETF already processed is never paid for again.
The fitted pipeline is saved to `artifacts/features/<version>/`.

## Tracking Azure costs

Every Azure call is recorded in `logs/azure_usage.csv` (token counters and cost, never the texts).

```bash
python -m etf_classifier.costs.report --since 2026-10-01
```

Token counting uses `tiktoken`, which downloads its tables on first use. If the network blocks it,
the estimate switches to approximate mode (flagged); the actual cost returned by Azure always
prevails. For an offline machine, pre-download the tables and set `TIKTOKEN_CACHE_DIR`.

## Training and using the model

```bash
python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure          # baseline without Azure
python -m etf_classifier.model.train --csv data/raw/etfs.csv --yes --promote     # with embeddings, promoted
python -m etf_classifier.model.compare --csv data/raw/etfs.csv --experiments local,local+embeddings
python -m etf_classifier.model.predict --csv new_etfs.csv                        # CSV without a Category column
```

- Tuning (regularisation and one threshold per category) is done on an internal validation set; the test set is only used for the final evaluation.
- Every model is compared with a trivial baseline (most frequent categories). Main metric: macro F1.
- Every model is saved to `artifacts/models/<version>/` with its feature pipeline, its metrics and a card (`MODEL_CARD.md`); `registry.json` points to the current model. The first model becomes current, later ones with `--promote`.
- Reports in `reports/`: `model_<version>.md` (metrics, details per category, most influential features), `model_comparison.md`.
- `predict` writes, for every ETF, the selected categories (at least 1, at most 5), the scores and an "uncertain" flag.

## Finding the closest ETFs

```bash
python -m etf_classifier.similarity.search --csv new_etfs.csv --k 5         # explained neighbours + near-duplicate alert
python -m etf_classifier.similarity.evaluate --csv data/raw/etfs.csv        # neighbour vote vs model vs baseline
```

- Combined score: 0.5 × composition similarity (cosine of the weights) + 0.5 × text similarity (embeddings or TF-IDF).
- Every neighbour is explained: share of the weight in common and main shared tickers.
- The index is built by `model.train` and stored in the model folder (`similarity.joblib`).

## Recording the team's validations

Validations are appended to a separate journal, `data/validations.csv`, which is automatically merged
with the original CSV (never rewritten) on every load: the next training therefore sees the new ETFs.

```bash
python -m etf_classifier.model.predict --csv new_etfs.csv --output to_validate.csv
# the team adds a Category column (validated categories) to to_validate.csv
python -m etf_classifier.feedback.importer --csv to_validate.csv --by "First Last" [--create-categories]
python -m etf_classifier.feedback.report                       # acceptance rate, bias alert
```

- Writes happen under a file lock, one complete line at a time, with a daily backup in `data/backups/`.
- A mistake is fixed with a cancellation (nothing is ever deleted).
- **Keep `data/` on a local disk of the server**: a file lock is not reliable on a network drive.
- Do not open the journal in Excel during validations (Windows then locks the file).

## Tests

```bash
pytest -v                 # full suite (≈ 30 s, no network call)
pytest -m "not slow"      # without the performance tests
```

## Configuration

All thresholds live in `config/settings.yaml` (separator, column names, near-duplicate overlap
threshold, test set size…). Tickers to merge manually (for example `GOOG` and `GOOGL`) are
declared in `config/holding_aliases.csv`.

## Structure

```
config/                       configuration and ticker aliases
data/raw/                     real CSV (not committed)
data/validations.csv          validation journal (not committed)
reports/                      generated reports (not committed)
src/etf_classifier/data/      data module
src/etf_classifier/costs/     costs module
src/etf_classifier/features/  features module
src/etf_classifier/model/     model module
src/etf_classifier/similarity/ similarity module
src/etf_classifier/feedback/  feedback module
logs/ cache/ artifacts/       logs, cache and pipelines (not committed)
tests/                        tests and fictitious CSV
tasks/                        plans and tasks
```
