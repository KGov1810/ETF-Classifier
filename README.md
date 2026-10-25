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
| `app` | Streamlit interface | ✅ Done |
| `llm-justify` | LLM justification (Azure OpenAI) | — |
| `retraining` | Daily retraining | ✅ Done |

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

## Using the interface

```bash
streamlit run src/etf_classifier/app/Home.py      # opens http://localhost:8501
```

Start it from the project root (the theme lives in `.streamlit/config.toml`). A current model is needed
(`model.train`). Choose your name in the sidebar: the list of validators is `app.validators` in
`config/settings.yaml` (placeholder names to replace with the team's).

- **Single ETF**: enter the name, the description and the composition (editable table; two columns can be
  pasted from Excel), then *Propose categories*. You see the proposed categories (pre-ticked), the scores,
  the closest ETFs and any near-duplicate alert; adjust the categories and *Validate*.
- **Batch**: upload a CSV `Name;Description;Composition` (UTF-8, `;`). Refused rows are listed with their
  reason; the others are validated one by one (*Next ETF*, *Skip*), uncertain ones first if you wish.
- **History**: recent validations, filters, and cancellation of a mistake (with a mandatory comment).
- **Monitoring**: acceptance rates and bias alerts, current model, Azure costs.

If the model uses Azure embeddings, the estimated cost is shown and must be confirmed before any call.
After the daily retraining, the new current model is picked up automatically (or use *Reload the model*).
v1 runs locally; sharing it on the team's network will be documented later.

## Daily retraining

```bash
python -m etf_classifier.retraining.run              # what the scheduler runs every day
python -m etf_classifier.retraining.run --force      # retrain even if nothing changed
python -m etf_classifier.retraining.run --dry-run    # train and compare, never promote
```

- Nothing new since the last training (no validation added or cancelled): the run stops in a few seconds.
- Otherwise a candidate is trained (Azure embeddings paid without confirmation) and compared with the current model
  on test ETFs the current model never saw. It is promoted only if its macro F1 is at least the current one minus 0.01,
  no category collapses, and it beats the trivial baseline (thresholds in `config/settings.yaml`, `retraining` section).
- Every run leaves a line in `artifacts/models/retraining_history.csv`, a report `reports/retraining_<version>.md`
  and a technical log `logs/retraining.log`. The interface shows the last runs in *Monitoring > Current model*.
- Every model version is kept (no automatic deletion).
- A model trained before this module does not know its training ETFs, which makes the comparison favour it:
  retrain it once with `python -m etf_classifier.model.train --csv data/raw/etfs.csv --promote`.

### Windows Task Scheduler

From the project folder, in PowerShell (no administrator rights needed):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_daily_retraining.ps1 -Time 10:00
Start-ScheduledTask -TaskName "ETF Classifier - daily retraining"     # test run now
Get-ScheduledTaskInfo -TaskName "ETF Classifier - daily retraining"   # LastTaskResult 0 = OK, 1 = failure
powershell -ExecutionPolicy Bypass -File scripts\unregister_daily_retraining.ps1   # remove the task
```

The task runs every day at 10:00 while your Windows session is open, starts as soon as possible if that time was
missed, and never runs twice at the same time. The console output goes to `logs/retraining_console.log`.
The `.venv` virtual environment must exist in the project folder (see Installation).

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
src/etf_classifier/app/       Streamlit interface
src/etf_classifier/retraining/ daily retraining
scripts/                      Windows Task Scheduler scripts
logs/ cache/ artifacts/       logs, cache and pipelines (not committed)
tests/                        tests and fictitious CSV
tasks/                        plans and tasks
```
