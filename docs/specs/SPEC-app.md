# Spec: `app` module

Status: **approved** (2026-10-25)

## Objective

A Streamlit interface for the ETF team. It replaces the command-line round trip
(`predict` → edit a CSV → `feedback.importer`) with a guided validation flow:
1. enter or load new ETFs;
2. see the model's proposals with scores, the closest ETFs and any near-duplicate alert;
3. validate or correct the categories, and the validation goes into the journal (`feedback` module);
4. look back at what was validated, cancel a mistake, and monitor acceptance rates and Azure costs.

Users: the team members (validators), on the team's server, through a browser.
v1 has no LLM justification (`llm-justify` module) and no authentication.

## Screens (v1)

### 1. Single ETF
- Fields: **Name**, **Description**, **Composition** in an **editable table** (columns `Ticker`, `Weight (%)`, rows can be added, deleted or pasted from Excel). Live total of the weights.
- "Propose categories" button → result panel:
  - **proposed categories pre-ticked** (at least 1, at most 5), scores of the other categories visible (top 10, sorted);
  - **"uncertain" banner** when the best score is below the threshold;
  - **closest ETFs** (5 by default): name, categories, combined score, share of the weight in common, main shared tickers;
  - **near-duplicate alert** if the ETF looks like an existing one, with its categories.
- Validation form: category multi-select (pre-ticked with the proposal, all known categories available), "create a new category" option (name + description, with the reminder that it will only be proposed after a retraining with enough examples), optional comment, **Validate** button.
- Feedback after validation: decision (`accepted` / `corrected`), validation identifier, form reset.

### 2. Batch from a CSV
- Upload of a CSV `Name;Description;Composition` (same format and checks as `model.predict`); excluded rows are listed with their reason.
- Proposals computed for the whole batch, then a **queue**: one ETF at a time, same result panel and validation form as screen 1, plus **Skip** and progress (`12 / 40 · 3 uncertain`).
- Filter "uncertain first". The composition of the current ETF can be corrected in the editable table before validating.
- Each validation is written to the journal immediately; leaving the page loses only the queue, never a validation (already-validated ETFs are then refused as duplicates, with the existing record).

### 3. History and cancellation
- Latest validations (default: 7 days), filterable by validator, decision and name.
- **Cancel** a validation: confirmation + mandatory comment → `cancellation` line in the journal (nothing is deleted).
- Cancelled validations stay visible, flagged.

### 4. Monitoring
- **Feedback**: acceptance rate (global and per validator), bias alerts, categories most often added or removed, created categories (content of `feedback.report`).
- **Azure costs**: totals over the period, per day and per module (content of `costs.report`), price-list date.
- **Current model**: version, date, macro F1 against the trivial baseline, link to the model card.

### Common to all screens
- **Validator** chosen in a drop-down list defined in `settings.yaml` (`app.validators`), remembered for the session. No validation possible without a validator.
- Sidebar: current model version, validator, "Reload the model" button (after the daily retraining).
- **Azure cost confirmation**: if the current model uses embeddings, the estimated cost of new texts is shown and a confirmation is required before the call (same rule as the commands; the cache makes already-seen texts free). Without Azure configuration, a clear message, and only a model without embeddings can be used.
- Clear messages for every known error: no current model, model without similarity index, journal locked ("another record is being saved, try again in a few seconds"), validation refused by the checks (with the reason).
- Interface in **English** (repository decision).

## Architecture

- **A service layer without Streamlit** (`app/services.py`): load the current model, predict, search neighbours, record, cancel, read the journal and the reports. Pages only display and call services. All business logic is therefore tested with pytest, without a browser.
- **In-memory ETFs go through the same checks as the CSV**: a function `load_records(rows, settings, require_categories=False)` will be extracted from `data.loader.load_dataset()` (same parsing, quality rules and ticker normalisation). `load_dataset()` will call it, with unchanged behaviour.
- **Model cached once per process** (`st.cache_resource`), keyed by version; "Reload the model" clears it.
- **Writes only through `FeedbackStore`** (lock, checks, backup): the app never touches the files directly.
- Every proposal shown records the model version, which is passed to the journal with the predicted categories.

## Tech Stack
- **New dependency**: `streamlit` (≥ 1.40, latest stable 1.66) — includes `st.data_editor` (editable table) and `streamlit.testing.v1.AppTest` (page tests without a browser).
- Existing: pandas, the project modules.

## Commands
```
pip install -r requirements.txt
streamlit run src/etf_classifier/app/Home.py            # http://localhost:8501
pytest tests/app -v
```
v1 runs locally (`localhost:8501`); sharing it on the team's network will be documented later.

## Project structure
```
src/etf_classifier/app/
├── Home.py                    # entry point: sidebar, validator choice, model status
├── services.py                # business logic without Streamlit (tested with pytest)
├── components.py              # shared widgets: result panel, validation form, composition editor
└── pages/
    ├── 1_Single_ETF.py
    ├── 2_Batch.py
    ├── 3_History.py
    └── 4_Monitoring.py
config/settings.yaml           # new `app` section: validators, neighbours shown, history period
tests/app/
├── test_services.py
└── test_pages.py              # AppTest: every page renders, main flows
```

## Code style
Same as the previous modules. Pages stay thin:
```python
result = services.propose(etf, settings, confirm_cost=confirmed)
if result.needs_cost_confirmation:
    st.info(f"Embedding {result.cost.n_texts} new text(s) will cost about ${result.cost.usd:.4f}.")
    confirmed = st.button("Confirm and continue")
else:
    components.result_panel(result)
```

## Test strategy
- **Services** (pytest, temporary folders, fixture model trained on `sample.csv`, fake Azure client):
  - a proposal returns categories, scores, neighbours and the near-duplicate alert;
  - an ETF typed in the table gives the same proposal as the same ETF loaded from a CSV (same checks and normalisation);
  - a validation lands in the journal with validator, model version and predicted categories; a refusal returns the checker's message;
  - cancellation, journal reading with filters;
  - cost confirmation required before any Azure call when the model uses embeddings.
- **Pages** (`AppTest`): every page renders without error, with and without a current model; validating from screen 1 writes one journal line; no validation without a validator.
- **`data` module**: the 92 existing `data` tests stay green after extracting `load_records()`.
- No network call in the tests.

## Boundaries
- **Always:** go through `FeedbackStore` for every write; show the model version with every proposal; ask for confirmation before an Azure expense; keep pages free of business logic.
- **Ask first:** adding a dependency other than `streamlit`; changing the journal format; adding authentication; exposing the app outside the team's network.
- **Never:** write a secret in the code or the interface; write to the original CSV; delete a journal line; show the content of `.env`.

## Success criteria
1. A validator can enter an ETF, see proposals, neighbours and alerts, and validate it in under a minute; the validation appears in the journal and in the next training.
2. A CSV batch can be validated ETF by ETF from the queue, with skip and progress; refused rows show their reason.
3. A validation can be cancelled from the history, with a comment.
4. The monitoring page shows the acceptance rates, the bias alerts and the Azure costs.
5. All tests pass without network; the `data` tests stay green.

## Decisions made
- Screens v1: single ETF, batch from CSV, history/cancellation, monitoring.
- Composition entered in an editable table (ticker, weight).
- Validator chosen from a list in `settings.yaml`.
- Proposals pre-ticked and scores visible; bias is monitored by the feedback report rather than by hiding the proposal.

- Validator list: **placeholder names** for now (`Alice Martin`, `Bruno Leroy`, `Chloe Dubois`), to be replaced in `settings.yaml` with the team's real names.
- Hosting: the app and the daily retraining run **locally (`localhost`)** first; the README documents `streamlit run` on `localhost:8501`. Sharing on the team's network will be a later decision.

## Open questions
None.
