# ETF Classifier

Proposes one or more categories for every new ETF, learned from 5,000 to 10,000 ETFs the team has already
categorised. A team member validates or corrects each proposal in a web interface; validated ETFs enrich the
dataset, and the model is retrained every day.

```
 new ETFs ──► model proposes categories ──► team validates in the app ──► journal of validations
                  ▲          + closest ETFs       (+ optional GPT-6.1 Sol          │
                  │          + near-duplicates      second opinion)               │
                  └──────────── daily retraining, promoted only if not worse ◄─────┘
```

The project is also a machine-learning learning ground: every step is measured against a trivial baseline,
and the reports explain what the model learned.

---

## Quick start

Python 3.11 or later, from the project folder.

1. **Install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Add your data**: put the categorised CSV in `data/raw/etfs.csv` (never committed; format below).
3. **Check its quality**
   ```bash
   python -m etf_classifier.data.report --csv data/raw/etfs.csv
   ```
   Excluded rows and their reason, categories, coverage, tickers to harmonise, near-duplicates
   → `reports/data_quality.md`.
4. **Train a first model** (local blocks, no Azure needed)
   ```bash
   python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure
   ```
   Macro F1 against the trivial baseline → `reports/model_<version>.md`.
5. **Open the interface**
   ```bash
   streamlit run src/etf_classifier/app/Home.py      # http://localhost:8501
   ```

Azure OpenAI (embeddings and GPT-6.1 Sol) and the daily retraining are optional: see [One-time setup](#one-time-setup).

---

## Daily use (team guide)

Choose your name in the sidebar, then:

| Screen | What you do |
|---|---|
| **Single ETF** | Enter the name, the description and the composition (editable table; two columns can be pasted from Excel), then *Propose categories*. Review the proposed categories (pre-ticked), the scores, the closest ETFs and any near-duplicate alert, adjust, then *Validate*. |
| **Batch** | Upload a CSV `Name;Description;Composition`. Refused rows are listed with their reason; the others are validated one by one (*Next ETF*, *Skip*), uncertain ones first if you wish. |
| **History** | Recent validations, with filters. Cancel a mistake (a comment is required: nothing is ever deleted). Read the LLM justification of a validation. |
| **Monitoring** | Acceptance rates and bias alerts, current model and its last retraining runs, Azure costs. |

**Second opinion with GPT-6.1 Sol.** Under a proposal, *Explain with GPT-6.1 Sol* gives a verdict for each
proposed category, categories to add or remove, a possible new category, and an alert when the description and
the holdings contradict each other. The estimated cost is shown and must be confirmed; the same request is never
paid twice. *Apply the suggestions* updates your selection, and you still decide.

**Good habits**
- A proposal flagged **uncertain** deserves a careful look: it may need a new category.
- If you always accept proposals unchanged, the Monitoring page raises a **bias alert**: the model would end up
  learning its own mistakes.
- A new category is proposed by the model only after the next retraining, once it has enough examples.
- Do not open `data/validations.csv` in Excel while the team validates (Windows locks the file).

Each validation is saved immediately. The next daily retraining learns from it; the app picks up the new
model by itself (or use *Reload the model*).

---

## One-time setup

### Azure OpenAI (optional)

1. Copy `.env.example` to `.env` (never committed) and fill in the endpoint, the key, the API version and the
   names of your deployments (`AZURE_OPENAI_EMBEDDING_DEPLOYMENT`, `AZURE_OPENAI_CHAT_DEPLOYMENT`).
2. Make sure these deployment names appear in `config/pricing.yaml` (`deployments` section), and **check the
   prices** against your contract: they are pre-filled with public Data Zone Standard prices (upper end).
3. Estimate, then train with the embeddings:
   ```bash
   python -m etf_classifier.features.build --csv data/raw/etfs.csv --estimate-only   # cost, no call
   python -m etf_classifier.model.train --csv data/raw/etfs.csv --promote            # asks for confirmation
   ```

Without `.env`, everything works with the local blocks; only the embeddings and the LLM button are disabled.
Every Azure call is recorded with its cost (`logs/azure_usage.csv`, counters only, never the texts).

### Team members

Replace the placeholder names of `app.validators` in `config/settings.yaml` with the team's names.

### Daily retraining on Windows

From the project folder, in PowerShell (no administrator rights needed):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_daily_retraining.ps1 -Time 10:00
Start-ScheduledTask -TaskName "ETF Classifier - daily retraining"     # test run now
Get-ScheduledTaskInfo -TaskName "ETF Classifier - daily retraining"   # LastTaskResult 0 = OK, 1 = failure
powershell -ExecutionPolicy Bypass -File scripts\unregister_daily_retraining.ps1   # remove the task
```

The task runs every day at 10:00 while your Windows session is open, starts as soon as possible if that time
was missed, and never runs twice at the same time. It needs the `.venv` of the quick start.

What a run does:
- nothing new since the last training (no validation added or cancelled) → stops in a few seconds;
- otherwise trains a candidate (Azure embeddings paid without confirmation) and compares it with the current model
  on test ETFs the current model never saw;
- promotes it only if its macro F1 is at least the current one minus 0.01, no category collapses, and it beats the
  trivial baseline;
- writes a report (`reports/retraining_<version>.md`) and a line in the history shown in the Monitoring page.

A model trained before the retraining module does not know its training ETFs, which biases the comparison in its
favour: retrain it once with `python -m etf_classifier.model.train --csv data/raw/etfs.csv --promote`.

---

## Reference

### Data format

UTF-8 CSV, `;` separator (cells containing a `;` must be quoted):

| Column | Example |
|---|---|
| `Name` | `iShares Core MSCI World` |
| `Description` | `Tracks the MSCI World index…` (English) |
| `Composition` | `[{'AAPL': 5.1}, {'MSFT': 4.6}, {'NVDA': 4.2}]`: tickers and weights in %, top holdings only is fine |
| `Category` | `['Global Equity', 'ESG']`: one or more categories (omitted for new ETFs) |

### Modules

Each module has its specification in [`docs/specs/`](docs/specs/); the architecture and the build order are in
[`docs/CAPABILITY-MAP.md`](docs/CAPABILITY-MAP.md).

| Module | Role | Spec |
|---|---|---|
| `data` | Loading, quality checks, ticker normalisation, near-duplicates, train/test split | [SPEC-data](docs/specs/SPEC-data.md) |
| `costs` | Azure OpenAI cost estimate before a call, actual cost after it, report in USD | [SPEC-costs](docs/specs/SPEC-costs.md) |
| `features` | Text (TF-IDF), holdings and composition statistics, cached Azure embeddings | [SPEC-features](docs/specs/SPEC-features.md) |
| `model` | One logistic regression per category, thresholds, evaluation, versioned registry | [SPEC-model](docs/specs/SPEC-model.md) |
| `similarity` | Closest ETFs (composition + text), near-duplicate alert, neighbour vote | [SPEC-similarity](docs/specs/SPEC-similarity.md) |
| `feedback` | Validation journal under lock, category registry, batch import, bias monitoring | [SPEC-feedback](docs/specs/SPEC-feedback.md) |
| `app` | Streamlit interface | [SPEC-app](docs/specs/SPEC-app.md) |
| `retraining` | Daily retraining, fair comparison, conditional promotion | [SPEC-retraining](docs/specs/SPEC-retraining.md) |
| `llm-justify` | GPT-6.1 Sol second opinion, on demand and cost-confirmed | [SPEC-llm-justify](docs/specs/SPEC-llm-justify.md) |

### Commands

Every command has `--help`; all but `costs.report` accept `--config <file>`. Commands that may call Azure show the
estimated cost and ask for confirmation, unless `--yes` is given (the daily retraining never asks).

| Command | Purpose | Output |
|---|---|---|
| `python -m etf_classifier.data.report --csv …` | Data quality report | `reports/data_quality.md` |
| `python -m etf_classifier.features.build --csv … [--estimate-only] [--no-azure]` | Build and save the features | `artifacts/features/<version>/` |
| `python -m etf_classifier.model.train --csv … [--no-azure] [--promote]` | Train, evaluate once, save | `artifacts/models/<version>/`, `reports/model_<version>.md` |
| `python -m etf_classifier.model.compare --csv … --experiments local,local+embeddings` | Compare feature sets on the same split | `reports/model_comparison.md` |
| `python -m etf_classifier.model.predict --csv new.csv [--output …]` | Proposals for new ETFs | CSV of proposals |
| `python -m etf_classifier.similarity.search --csv new.csv [--k 5]` | Closest ETFs, explained | CSV + Markdown summary |
| `python -m etf_classifier.similarity.evaluate --csv …` | Neighbour vote vs model vs baseline | `reports/similarity_evaluation.md` |
| `python -m etf_classifier.feedback.importer --csv validated.csv --by "First Last"` | Import validations made outside the app | journal |
| `python -m etf_classifier.feedback.report [--since …]` | Acceptance rates, bias alert | `reports/feedback.md` |
| `python -m etf_classifier.costs.report [--since …]` | Azure costs per day, module, operation | `reports/azure_costs.md` |
| `python -m etf_classifier.retraining.run [--force] [--dry-run]` | Daily retraining (what the scheduler runs) | history + `reports/retraining_<version>.md` |

### Where things are kept

None of these files is committed.

| File or folder | Content |
|---|---|
| `data/raw/` | The original CSV, never rewritten |
| `data/validations.csv` | Validation journal: appended under a lock, merged with the original CSV on every load; mistakes are fixed by cancellation lines |
| `data/categories.csv` | Category registry (historical and created, with author and description) |
| `data/backups/` | Daily backups of the journal (last 30) |
| `data/llm_justifications.jsonl` | GPT-6.1 Sol answers, linked to the validations that followed them |
| `artifacts/models/` | One folder per model (model, features, similarity index, metrics, card), `registry.json` (current model), retraining history |
| `cache/` | Embeddings and LLM answers already paid for |
| `logs/` | Azure costs, retraining log |
| `reports/` | Generated reports |

Keep `data/` on a local disk: a file lock is not reliable on a network drive.

### Configuration

- `config/settings.yaml`: every threshold and path (CSV format, near-duplicate threshold, split, model, similarity
  weights, validators, retraining rule, LLM limits).
- `config/pricing.yaml`: dated Azure price list, in USD per million tokens.
- `config/holding_aliases.csv`: tickers to merge by hand (for example `GOOG` and `GOOGL`).
- `.env`: Azure endpoint, key, API version and deployment names (see `.env.example`).

Token counting uses `tiktoken`, which downloads its tables on first use; on an offline machine the estimate is
flagged as approximate (pre-download the tables and set `TIKTOKEN_CACHE_DIR` to avoid it). The actual cost returned
by Azure always prevails.

### Tests

```bash
pytest -v                 # full suite (about 50 s, no network call: Azure is faked)
pytest -m "not slow"      # without the performance tests
```

### Project structure

```
config/                         settings, price list, ticker aliases
docs/CAPABILITY-MAP.md          modules, dependencies, build order, decisions
docs/specs/                     one specification per module
src/etf_classifier/
├── data/  costs/  features/    data preparation, Azure costs, features
├── model/  similarity/         classifier, closest ETFs
├── feedback/  retraining/      validation journal, daily retraining
├── llm/                        GPT-6.1 Sol justification
├── app/                        Streamlit interface (services.py holds the logic, pages only display)
├── common.py                   shared helpers (timestamps, Markdown tables, append-only CSV)
└── azure.py                    Azure OpenAI client shared by the embeddings and the LLM
scripts/                        Windows Task Scheduler scripts
tests/                          tests (shared fixtures in conftest.py, fakes in support/, fictitious CSV in data/fixtures/)
tasks/                          implementation plans and task lists, per module
```
