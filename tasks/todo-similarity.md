# Tasks: `similarity` module

Definition of "done" for every task: green tests (`pytest -v`), no regression on the existing
modules, no network call in the tests, no commit without your approval.

---

## S1: configuration and index building

**Description:** `similarity` section in `settings.yaml` (composition/text weights, number of neighbours, number of shared tickers displayed, k values evaluated); `SimilarityIndex.build(reference_df, pipeline)`: composition vectors (own vocabulary, L2-normalised weights) and text vectors (embeddings or TF-IDF from the pipeline).

**Acceptance criteria:**
- [x] The vocabulary covers all the tickers of the reference, including those absent from the model's vocabulary
- [x] The text block used is the embeddings one if active, otherwise TF-IDF; it is recorded in the index
- [x] A reference ETF without composition has a zero composition vector, without error

**Verification:** `pytest tests/similarity/test_index.py -v`

**Dependencies:** none

**Files:** `config/settings.yaml`, `src/etf_classifier/similarity/__init__.py`, `src/etf_classifier/similarity/index.py`, `tests/similarity/…`

**Size:** M

---

## S2: neighbour search, explanation and near-duplicate alert

**Description:** `index.query(df, k, exclude_row_ids=None)`: combined score (composition and text, weights renormalised if a composition is empty), k sorted neighbours, explanation (weighted overlap in %, 5 main shared tickers), near-duplicate alert (rule of the `data` module).

**Acceptance criteria:**
- [x] Two identical compositions: composition similarity of 1 and near-duplicate alert
- [x] A small ETF contained in a large one: composition similarity clearly below 1, overlap of 100 %
- [x] A reference ETF is never its own neighbour; empty composition: text only, with a mention

**Verification:** `pytest tests/similarity/test_query.py -v`

**Dependencies:** S1

**Files:** `src/etf_classifier/similarity/index.py`, `tests/similarity/test_query.py`

**Size:** M

---

## S3: neighbour vote, saving and reloading

**Description:** `neighbor_vote(neighbors)` (score per category = sum of the scores of the neighbours that have it / sum of the scores) and decision with the same rules as the model (at least 1 category, at most 5); `save()` / `load()` of the index.

**Acceptance criteria:**
- [x] The vote computed by hand on a small case is reproduced
- [x] On the synthetic dataset, the vote clearly beats the trivial baseline
- [x] Saving then reloading: same neighbours and same scores

**Verification:** `pytest tests/similarity/test_vote.py -v`

**Dependencies:** S2

**Files:** `src/etf_classifier/similarity/vote.py`, `src/etf_classifier/similarity/index.py`, `tests/similarity/test_vote.py`

**Size:** S

---

## Checkpoint A
- [ ] Tests green
- [ ] 100 queries on a reference of 10,000 ETFs in less than 10 s

---

## S4: index built by `train` and reloaded with the model

**Description:** `model.train` builds the index on all usable rows and writes it to the model folder; `load_bundle()` reloads it (`bundle.similarity`), or `None` with a clear message for an older model. Index size shown in the model card.

**Acceptance criteria:**
- [x] After `train`, the model folder contains `similarity.joblib`
- [x] `load_bundle()` returns an index that gives the same neighbours as before saving
- [x] A model folder without an index still loads (`similarity` is `None`)

**Verification:** `pytest tests/model tests/similarity/test_integration.py -v`

**Dependencies:** S3

**Files:** `src/etf_classifier/model/train.py`, `src/etf_classifier/model/registry.py`, `tests/similarity/test_integration.py`

**Size:** S

---

## S5: `search` command

**Description:** `python -m etf_classifier.similarity.search --csv new.csv [--model current] [--k 5] [--yes]`: CSV of neighbours (one ETF × one neighbour per line) and Markdown summary; embeddings through the cache and cost confirmation.

**Acceptance criteria:**
- [x] For every new ETF, k neighbours with scores, categories and explanation
- [x] A near-duplicate is flagged in the summary
- [x] Clear message if there is no current model, or a model without an index

**Verification:** `pytest tests/similarity/test_search.py -v`

**Dependencies:** S4

**Files:** `src/etf_classifier/similarity/search.py`, `tests/similarity/test_search.py`

**Size:** M

---

## S6: `evaluate` command

**Description:** `python -m etf_classifier.similarity.evaluate --csv … [--yes]`: index built on training only, neighbour vote for every test ETF (k = 5, 10, 20), comparison with the model tuned on the same split and with the trivial baseline; report `reports/similarity_evaluation.md`.

**Acceptance criteria:**
- [x] The evaluation index contains no test ETF
- [x] The report contains the vote for each k, the model and the trivial baseline (macro and micro F1)
- [x] Same test set as the model (fingerprint displayed)

**Verification:** `pytest tests/similarity/test_evaluate.py -v`

**Dependencies:** S3

**Files:** `src/etf_classifier/similarity/evaluate.py`, `tests/similarity/test_evaluate.py`

**Size:** M

---

## Final checkpoint
- [ ] The 4 success criteria of `SPEC-similarity.md`
- [ ] After the first real training: composition/text weights and use of the vote decided with you
