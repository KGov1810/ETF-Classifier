# Tasks: `app` module

Definition of "done" for every task: green tests (`pytest -v`), no regression on the existing
modules, no network call in the tests, no commit without your approval.

---

## P1: `streamlit` dependency, `app` settings section and `load_records()` extraction

**Description:** add `streamlit>=1.40` to `requirements.txt`; `app` section in `settings.yaml` (validators `Alice Martin`, `Bruno Leroy`, `Chloe Dubois`, neighbours shown, scores shown, history period); extract `load_records(rows, settings, require_categories)` from `data.loader.load_dataset()`, and `services.etf_from_inputs(name, description, holdings_table)` which builds a row from the editable table.

**Acceptance criteria:**
- [x] `load_dataset()` gives exactly the same result as before (92 `data` tests green)
- [x] An ETF built from a table (`Ticker`, `Weight (%)`) goes through the same quality rules and normalisation as a CSV row (` aapl ` → `AAPL`, negative weight → refused with its reason)
- [x] Empty table rows are ignored; a duplicate ticker has its weights added up, with a warning

**Verification:** `pytest tests/data tests/app/test_services.py -v`

**Dependencies:** none

**Files:** `requirements.txt`, `config/settings.yaml`, `src/etf_classifier/data/loader.py`, `src/etf_classifier/app/__init__.py`, `src/etf_classifier/app/services.py`, `tests/app/…`

**Size:** M

---

## P2: proposal service

**Description:** `services.load_current_model(settings)`; `services.estimate(etfs, bundle, settings)` (cost of new texts if the model uses embeddings); `services.propose(etfs, bundle, settings, confirmed)` returning, per ETF, the selected categories, top-10 scores, "uncertain" flag, neighbours, near-duplicate alert and model version.

**Acceptance criteria:**
- [x] On the fixture model, a proposal contains at least 1 category, scores, `k` neighbours and the model version; a copy of an existing ETF raises the near-duplicate alert
- [x] With an embeddings model and the fake client, no Azure call happens without `confirmed=True`; already-cached texts cost nothing
- [x] No current model, or a model without a similarity index: clear error (proposal still possible without neighbours in the second case)

**Verification:** `pytest tests/app/test_services.py -v`

**Dependencies:** P1

**Files:** `src/etf_classifier/app/services.py`, `tests/app/conftest.py`, `tests/app/test_services.py`

**Size:** M

---

## P3: validation, cancellation and history services

**Description:** `services.validate(etf, categories, validator, proposal, comment, new_categories)`, `services.cancel(validation_id, validator, comment)`, `services.history(settings, since, validator, decision, name)`, `services.known_categories(settings)`; feedback errors converted to readable messages.

**Acceptance criteria:**
- [x] A validation lands in the journal with validator, model version, predicted categories and the derived decision
- [x] Refusals (empty name, unknown category without creation, duplicate, locked journal, no validator) return the readable message, without writing
- [x] Cancellation requires a comment; the history shows cancelled validations as flagged and applies the filters

**Verification:** `pytest tests/app/test_services.py -v`

**Dependencies:** P2

**Files:** `src/etf_classifier/app/services.py`, `tests/app/test_services.py`

**Size:** M

---

## Checkpoint A
- [x] All tests pass, `data` suite unchanged
- [x] A typed ETF and the same ETF read from a CSV give the same proposal
- [ ] Review with you before the pages

---

## P4: home page, sidebar and "Single ETF" screen

**Description:** `Home.py` (model status, validator choice, "Reload the model"); `components.py` (composition editor with live weight total, result panel, validation form); `pages/1_Single_ETF.py` wiring the flow input → cost confirmation if needed → proposal → validation.

**Acceptance criteria:**
- [x] Every page renders without error, with and without a current model (`AppTest`)
- [x] From the "Single ETF" screen, proposing then validating writes exactly one journal line
- [x] No validation possible without a validator; the "Validate" button is disabled once the ETF is validated

**Verification:** `pytest tests/app -v`; manual run `streamlit run src/etf_classifier/app/Home.py`

**Dependencies:** P3

**Files:** `src/etf_classifier/app/Home.py`, `src/etf_classifier/app/components.py`, `src/etf_classifier/app/pages/1_Single_ETF.py`, `tests/app/test_pages.py`

**Size:** M

---

## P5: "Batch" screen

**Description:** `pages/2_Batch.py`: CSV upload, excluded rows with their reason, cost confirmation, proposals for the batch, queue (validate / skip / progress, "uncertain first" filter, composition correctable before validating).

**Acceptance criteria:**
- [x] A batch of the fixture produces a queue with the right count and the number of uncertain ETFs
- [x] Validating then moving on writes one line per validated ETF; "Skip" writes nothing
- [x] Excluded rows are listed with their reason; an ETF already validated is refused with the existing record

**Verification:** `pytest tests/app -v`; manual run with `tests/data/fixtures/sample.csv` without the Category column

**Dependencies:** P4

**Files:** `src/etf_classifier/app/pages/2_Batch.py`, `src/etf_classifier/app/services.py` (batch helpers), `tests/app/test_pages.py`, `tests/app/test_services.py`

**Size:** M

---

## P6: "History" screen

**Description:** `pages/3_History.py`: latest validations (7 days by default), filters (validator, decision, name), cancellation with confirmation and mandatory comment.

**Acceptance criteria:**
- [x] The table shows the journal's validations with the filters applied
- [x] Cancelling with a comment adds a `cancellation` line; the validation is then shown as cancelled
- [x] Empty journal: clear message

**Verification:** `pytest tests/app -v`

**Dependencies:** P4

**Files:** `src/etf_classifier/app/pages/3_History.py`, `tests/app/test_pages.py`

**Size:** S

---

## P7: "Monitoring" screen and README

**Description:** `pages/4_Monitoring.py`: feedback report (acceptance rates, bias alerts, categories added/removed, created categories), Azure costs over a chosen period, current model (version, macro F1 vs trivial baseline, model card); README section "Using the interface".

**Acceptance criteria:**
- [x] The figures shown are those of `build_feedback_report` and `build_cost_report` on a fictitious journal
- [x] Missing journal or cost log: clear messages, no error
- [x] The README explains how to start the app locally and the 4 screens

**Verification:** `pytest tests/app -v`; full suite `pytest -v`

**Dependencies:** P4

**Files:** `src/etf_classifier/app/pages/4_Monitoring.py`, `tests/app/test_pages.py`, `README.md`

**Size:** S

---

## Final checkpoint
- [ ] The 5 success criteria of `SPEC-app.md`
- [ ] You try the 4 screens on `localhost:8501` with a model trained on the fixture
