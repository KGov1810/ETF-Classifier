# Tasks: new input format, one category per ETF, ticker as identifier

Definition of "done" for every task: tests written first (red then green), no regression on the modules already
migrated, no network call in the tests, no commit without your approval.

---

## Phase 1: data

### T1: settings, fixture and synthetic data in the new format

**Description:** `csv.columns` gets `ticker: "Ticker"`, `name: "Nom"`, `category: "Classification"`;
`tests/data/fixtures/sample.csv` rewritten as `Ticker;Nom;Description;Classification;Composition` (one category,
including a row without description, a row without composition, a duplicated ticker, an empty ticker);
`make_synthetic` produces one category and a unique ticker per ETF.

**Acceptance criteria:**
- [x] The fixture covers every rule of the spec at least once
- [x] `make_synthetic` rows each have exactly one category and a unique `ticker`

**Verification:** `pytest tests/data/test_integrity.py -v` (fixture sanity)

**Dependencies:** none · **Files:** `config/settings.yaml`, `tests/data/fixtures/sample.csv`, `tests/support/synthetic.py`, `tests/conftest.py` · **Size:** S

### T2: quality rules and `ticker` column

**Description:** in `data.validation`: ticker normalised and required, name required, description *or* composition
required, `Classification` read as one plain-text category, ticker duplicated in the CSV → every such row excluded,
`duplicate_name` removed; `ticker` added to `DATA_COLUMNS`; `read_raw_rows` expects the new columns.

**Acceptance criteria:**
- [x] Each new rule excludes the right rows with its label; the existing warnings are unchanged
- [x] Two rows sharing a ticker (spaces or case aside) are both excluded; two rows sharing a name are kept
- [x] `Classification` "Gold" gives `categories == ["Gold"]`

**Verification:** `pytest tests/data -v`

**Dependencies:** T1 · **Files:** `data/validation.py`, `data/parsing.py`, `data/loader.py`, `tests/data/test_validation.py`, `tests/data/test_loader.py` · **Size:** M

### T3: journal by ticker, single-category validation, import

**Description:** `JOURNAL_COLUMNS` gets `Ticker`; `record_validation` requires exactly one category and a ticker,
and accepts a known ticker (replacement); the loader merges the journal by ticker (last active row wins);
`feedback.importer` reads the new CSV format.

**Acceptance criteria:**
- [x] A validation of a reference ticker replaces the reference row in the loaded dataset; a later one replaces it again; cancelling the latest restores the previous one
- [x] Zero or two categories → `ValidationError`; empty ticker → `ValidationError`
- [x] Importing a new-format CSV writes one journal line per valid row

**Verification:** `pytest tests/feedback tests/data -v`

**Dependencies:** T2 · **Files:** `feedback/store.py`, `feedback/checks.py`, `feedback/importer.py`, `data/loader.py`, `tests/feedback/…` · **Size:** M

### T4: quality report and fingerprint

**Description:** `data.report` lists the excluded ticker duplicates and the new rules; `dataset_fingerprint`
includes the ticker.

**Acceptance criteria:**
- [x] The report on the fixture lists every excluded row with its reason and the duplicated tickers
- [x] Changing only a ticker changes the fingerprint

**Verification:** `pytest tests/data/test_report.py tests/features/test_pipeline.py -v`

**Dependencies:** T2 · **Files:** `data/report.py`, `features/pipeline.py`, tests · **Size:** S

**Checkpoint 1:** `pytest tests/data tests/feedback`; `python -m etf_classifier.data.report --csv tests/data/fixtures/sample.csv` reviewed.

---

## Phase 2: model

### T5: single-category classifier, baseline and metrics

**Description:** multinomial logistic regression replacing one-vs-rest and thresholds; `Prediction` keeps
`categories` (one element), `scores` (sorted, summing to 1), `uncertain`; baseline = most frequent category;
metrics = accuracy and macro F1 (+ micro F1), confusion table of the most frequent mistakes in the report;
`max_categories`, thresholds and threshold settings removed.

**Acceptance criteria:**
- [x] Every prediction has exactly one category; its scores sum to 1
- [x] On the synthetic data the model clearly beats the baseline (accuracy and macro F1)
- [x] The report shows accuracy, macro F1, the baseline and the top confusions; no threshold column

**Verification:** `pytest tests/model -v`

**Dependencies:** T2 · **Files:** `model/classifier.py`, `model/baseline.py`, `model/evaluate.py`, `model/train.py`, `config/settings.yaml`, tests · **Size:** M

### T6: training tickers and retraining comparison by ticker

**Description:** `train_tickers.json` replaces `train_names.json`; `comparison_set` excludes by ticker.

**Acceptance criteria:**
- [x] A trained model folder holds the training tickers
- [x] The comparison set excludes exactly the tickers the current model trained on; without the file, the whole test set is used and flagged
- [x] `tests/retraining` green, promotion rule unchanged

**Verification:** `pytest tests/retraining -v`

**Dependencies:** T5 · **Files:** `model/registry.py`, `model/train.py`, `retraining/compare.py`, `retraining/run.py`, tests · **Size:** S

### T7: established classification check and prediction CLI

**Description:** pure function `check_established(prediction, established, classes)` → `agreement` /
`disagreement` / `unknown` / `None` with the established category's score; `model.predict` reads the new format
(Classification optional) and writes `Ticker`, `Name`, `Category`, `Scores` (top 3), `EstablishedCategory`,
`EstablishedScore`, `Check`, with counts in the console.

**Acceptance criteria:**
- [x] The three verdicts and the empty case are covered by tests
- [x] The CSV has the check columns only filled when a classification is given

**Verification:** `pytest tests/model/test_predict.py tests/model/test_check.py -v`

**Dependencies:** T5 · **Files:** `model/check.py`, `model/predict.py`, tests · **Size:** S

**Checkpoint 2:** `pytest tests/model tests/retraining tests/features tests/similarity`; `python -m etf_classifier.model.train --csv tests/data/fixtures/sample.csv --no-azure` reviewed.

---

## Phase 3: app, similarity, LLM

### T8: app services

**Description:** `etf_from_inputs` takes the ticker; `Proposal` gets `ticker`, `established`, `check`,
`established_score`; `load_batch` reads the new format; `queue_order` puts disagreements and unknown first, then
uncertain, then agreements; `validate` writes one category and the ticker; history gets the ticker.

**Acceptance criteria:**
- [x] Missing ticker → error message; ticker normalised
- [x] `queue_order` follows disagreements/unknown → uncertain → agreements
- [x] A validation writes the ticker and one category

**Verification:** `pytest tests/app/test_inputs.py tests/app/test_proposals.py tests/app/test_validation.py -v`

**Dependencies:** T3, T7 · **Files:** `app/services.py`, tests · **Size:** M

### T9: app pages

**Description:** Single ETF: required `Ticker` field. Validation form: one-category selector pre-filled with the
proposal, 2–3 alternatives with their score selectable in one click, *new category* kept. Batch: established
category and alert shown, picked in one click, agreement/disagreement counts in the summary. History: `Ticker`
column and filter.

**Acceptance criteria:**
- [x] AppTest: proposal without ticker refused; single-category validation saved
- [x] AppTest: a batch with a disagreement shows it first and lets the established category be picked
- [x] Every page renders without exception

**Verification:** `pytest tests/app -v`

**Dependencies:** T8 · **Files:** `app/components.py`, `app/pages/1_Single_ETF.py`, `app/pages/2_Batch.py`, `app/pages/3_History.py`, `tests/app/test_pages.py` · **Size:** M

### T10: similarity results with tickers (done in phase 2: needed by checkpoint 2)

**Description:** the index keeps the tickers; neighbours and near-duplicate alerts show `ticker · name`;
`similarity.search` CSV gets `Ticker` and `NeighborTicker`.

**Acceptance criteria:**
- [x] Neighbours carry their ticker; two ETFs with the same name are told apart
- [x] `tests/similarity` green

**Verification:** `pytest tests/similarity -v`

**Dependencies:** T2 · **Files:** `similarity/index.py`, `similarity/search.py`, `app/components.py`, tests · **Size:** S

### T11: LLM justification for one category

**Description:** new schema (verdict on the proposed category, optional alternative among the known categories,
optional new category, inconsistency, confidence); `check_answer` refuses an unknown alternative; context gives
the ticker and the established classification if any; *Apply the suggestion* sets the selector;
`PROMPT_VERSION` changed.

**Acceptance criteria:**
- [x] Schema and checks tested (valid answer, unknown alternative refused, JSON-mode fallback still parsed)
- [x] AppTest: applying the alternative changes the selected category

**Verification:** `pytest tests/llm tests/app/test_llm.py -v`

**Dependencies:** T9 · **Files:** `llm/prompts.py`, `llm/justify.py`, `app/components.py`, `app/services.py`, tests · **Size:** M

**Checkpoint 3:** full suite green; `streamlit run` on a model trained from the fixture, reviewed by hand.

---

## Phase 4: documentation

### T12: README, module specs, end-to-end check

**Description:** README (data format, team guide, batch check, prediction columns), the affected module specs
(`data`, `model`, `feedback`, `app`, `retraining`, `llm-justify`, `similarity`) and `CAPABILITY-MAP.md` aligned
with the change spec; end-to-end run on the fixture: report → train → predict with established classification →
retraining.

**Acceptance criteria:**
- [x] No reference left to the `Name;Description;Composition;Category` format or to multiple categories
- [x] `pytest -v` and `ruff check src tests` clean
- [x] End-to-end commands run without error

**Verification:** `pytest -v && ruff check src tests`

**Dependencies:** T1–T11 · **Files:** `README.md`, `docs/specs/*.md`, `docs/CAPABILITY-MAP.md` · **Size:** M

**Checkpoint 4:** commit proposed to you (message and push on your approval).
