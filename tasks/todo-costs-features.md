# Tasks: `costs` and `features` modules

Definition of "done" for every task: green tests (`pytest -v`), no regression on the `data` module,
no network call in the tests, no commit without your approval.

---

## A1: price list and checked loading

**Description:** create `config/pricing.yaml` (USD, Data Zone, upper end) and `pricing.py`, which reads and checks the price list and maps a deployment name to its model.

**Acceptance criteria:**
- [x] A valid price list loads; a missing or negative price raises a clear error
- [x] An unknown deployment is detected (without a phantom zero cost)
- [x] The age of the price list (`as_of`) can be computed, for the warning beyond 90 days

**Verification:** `pytest tests/costs/test_pricing.py -v`

**Dependencies:** none

**Files:** `config/pricing.yaml`, `src/etf_classifier/costs/__init__.py`, `src/etf_classifier/costs/pricing.py`, `tests/costs/test_pricing.py`, `.gitignore` (adds `logs/`)

**Size:** S

---

## A2: token counting and estimates

**Description:** `count_tokens()` with `tiktoken`, `estimate_embeddings()` and `estimate_chat()`, with long-context pricing and an "approximate" flag.

**Acceptance criteria:**
- [x] 1,000,000 embedding tokens cost exactly $0.13 with the test price list
- [x] Long-context pricing applies beyond the threshold
- [x] The GPT-6.1 Sol estimate is flagged as approximate

**Verification:** `pytest tests/costs/test_estimate.py -v`

**Dependencies:** A1

**Files:** `requirements.txt`, `src/etf_classifier/costs/tokens.py`, `src/etf_classifier/costs/estimate.py`, `tests/costs/test_estimate.py`

**Size:** S

---

## A3: actual cost journal (`CostTracker`)

**Description:** `CostTracker.record()` computes the cost from the usage returned by Azure (input, cache, output) and appends a line to `logs/azure_usage.csv`.

**Acceptance criteria:**
- [x] One line per call, with all the columns of the spec; never the content of the texts
- [x] Unknown deployment: line recorded with its tokens, empty cost and a warning
- [x] The prior estimate, when provided, is recorded next to the actual cost

**Verification:** `pytest tests/costs/test_tracker.py -v`

**Dependencies:** A2

**Files:** `src/etf_classifier/costs/tracker.py`, `tests/costs/test_tracker.py`

**Size:** S

---

## A4: cost report (CLI)

**Description:** `python -m etf_classifier.costs.report [--since] [--until]`: totals per day, module, operation and deployment, estimate/actual gap, price-list age; writes `reports/azure_costs.md`.

**Acceptance criteria:**
- [x] The totals of a fictitious journal match the values computed by hand
- [x] The period filter works; a missing journal gives a clear message (no error)
- [x] Warning if the price list is more than 90 days old

**Verification:** `pytest tests/costs/test_report.py -v`; manual run on a fictitious journal

**Dependencies:** A3

**Files:** `src/etf_classifier/costs/report.py`, `tests/costs/test_report.py`

**Size:** S

---

## Checkpoint A
- [ ] All tests pass without network
- [ ] You check the prices of `config/pricing.yaml` against your Azure contract

---

## B1: dependencies, configuration and category encoding

**Description:** add `scikit-learn`, `openai`, `python-dotenv`; `features` section in `settings.yaml` (active blocks, weights, parameters); `labels.py` for the multi-label matrix.

**Acceptance criteria:**
- [x] `fit_labels` on training, then `transform_labels` on test give identical columns
- [x] A category missing from training is reported (warning), never silently ignored
- [x] The configuration lets you enable or disable each block

**Verification:** `pytest tests/features/test_labels.py -v`

**Dependencies:** none (can start after phase A)

**Files:** `requirements.txt`, `config/settings.yaml`, `src/etf_classifier/features/__init__.py`, `src/etf_classifier/features/labels.py`, `tests/features/test_labels.py`

**Size:** S

---

## B2: `text_tfidf` block

**Description:** TF-IDF on "Name. Description": words and word pairs, English stop words, `sublinear_tf`, minimum frequency of 2.

**Acceptance criteria:**
- [x] A word present only in the test set does not enter the vocabulary
- [x] An empty description gives a zero row, without error
- [x] Feature names are readable (`text:clean energy`)

**Verification:** `pytest tests/features/test_text.py -v`

**Dependencies:** B1

**Files:** `src/etf_classifier/features/text.py`, `tests/features/test_text.py`

**Size:** S

---

## B3: `holdings` and `composition_stats` blocks

**Description:** sparse ticker → weight vector weighted by IDF and L2-normalised; composition statistics (coverage, number of holdings, top holding, top-10 concentration, Herfindahl, empty composition, share of unknown tickers), scaled.

**Acceptance criteria:**
- [x] A ticker present only in the test set is ignored, and its share of the weight appears in "share of unknown tickers"
- [x] Concentration and Herfindahl match the values computed by hand
- [x] An empty composition causes neither an error nor an infinite value

**Verification:** `pytest tests/features/test_holdings.py -v`

**Dependencies:** B1

**Files:** `src/etf_classifier/features/holdings.py`, `tests/features/test_holdings.py`

**Size:** M

---

## B4: fit/transform pipeline, saving and reloading

**Description:** `FeaturePipeline` combines the active blocks with their weights; `fit` on training, `transform` on any set; `save`/`load` in `artifacts/features/<version>/`.

**Acceptance criteria:**
- [x] Same training, same matrix; saving then reloading give an identical transformation
- [x] A disabled block disappears from the combined matrix without any other change
- [x] The 3 local blocks are built on 10,000 synthetic ETFs in less than 60 s

**Verification:** `pytest tests/features/test_pipeline.py -v`

**Dependencies:** B2, B3

**Files:** `src/etf_classifier/features/pipeline.py`, `tests/features/test_pipeline.py`, `.gitignore` (adds `artifacts/`, `cache/`)

**Size:** M

---

## Checkpoint B
- [ ] No-leak and reproducibility tests green
- [ ] Quick review with you of the features produced on the fixture

---

## C1: embeddings client

**Description:** `embeddings.py`: batching (64), truncation at 8,192 tokens, retries on 429 and server errors, L2-normalised vectors; injectable client, tested with a fake client.

**Acceptance criteria:**
- [x] 150 texts give 3 batches (64, 64, 22) and 150 vectors of 3,072 dimensions, in order
- [x] A text that is too long is truncated with a warning
- [x] Without Azure variables, the block is disabled with a clear message and the pipeline continues

**Verification:** `pytest tests/features/test_embeddings.py -v`

**Dependencies:** B4, A3

**Files:** `src/etf_classifier/features/embeddings.py`, `tests/features/test_embeddings.py`, `tests/features/fake_azure.py`

**Size:** M

---

## C2: SQLite cache and per-batch cost tracking

**Description:** cache `cache/embeddings.sqlite` (key = fingerprint of the model, the dimension and the text); one `CostTracker` line per batch actually sent.

**Acceptance criteria:**
- [x] A second pass on the same texts makes no call to the client
- [x] Changing the dimension does not reuse vectors of another dimension
- [x] One cost line per batch sent, none for texts served by the cache

**Verification:** `pytest tests/features/test_embeddings.py -v`

**Dependencies:** C1

**Files:** `src/etf_classifier/features/embeddings.py`, `src/etf_classifier/features/cache.py`, `tests/features/test_cache.py`

**Size:** M

---

## C3: `build` command line and `.env.example`

**Description:** `python -m etf_classifier.features.build --csv … [--estimate-only] [--yes]`: estimate (texts not in the cache, tokens, USD), confirmation, building, pipeline saving, per-block summary and actual cost.

**Acceptance criteria:**
- [x] `--estimate-only` never calls the client
- [x] Without `--yes`, any answer other than "y" cancels without spending
- [x] `.env.example` lists the required variables without any secret value

**Verification:** `pytest tests/features/test_build.py -v`; manual run on the fixture without Azure

**Dependencies:** C2

**Files:** `src/etf_classifier/features/build.py`, `tests/features/test_build.py`, `.env.example`, `README.md`

**Size:** M

---

## Checkpoint C: real run
- [ ] You run `build --estimate-only` then `build` on a small sample with your Azure key
- [ ] Gap between estimate and actual cost below 2 %
- [ ] A second run makes no Azure call
