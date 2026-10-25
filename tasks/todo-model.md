# Tasks: `model` module

Definition of "done" for every task: green tests (`pytest -v`), no regression on the
`data`, `costs` and `features` modules, no network call in the tests, no commit without your approval.

---

## M1: evaluation metrics and trivial baseline

**Description:** `model` section in `settings.yaml` (`C` grid, minimum count to tune a threshold, "uncertain" threshold, maximum number of categories, seed); `evaluate.py` (micro, macro and per-ETF F1, exact match, Hamming loss, details per category); `baseline.py` (predicts the most frequent training categories).

**Acceptance criteria:**
- [x] Metrics match the values computed by hand on a small case
- [x] The trivial baseline predicts, for every ETF, the median number of categories among the most frequent ones
- [x] A category absent from both predictions and true values does not skew macro F1 (handled explicitly)

**Verification:** `pytest tests/model/test_evaluate.py tests/model/test_baseline.py -v`

**Dependencies:** none

**Files:** `config/settings.yaml`, `src/etf_classifier/model/__init__.py`, `src/etf_classifier/model/evaluate.py`, `src/etf_classifier/model/baseline.py`, `tests/model/…`

**Size:** M

---

## M2: classifier, per-category thresholds and prediction rules

**Description:** one-vs-rest logistic regression with `class_weight="balanced"`; choice of the threshold maximising each category's F1; `predict()` with at least one category, at most 5, sorted scores and "uncertain" flag.

**Acceptance criteria:**
- [x] On hand-built scores, the selected threshold maximises the category's F1; below the minimum count, the threshold is 0.5
- [x] An ETF with no category above the thresholds receives the best one; never more than 5 categories selected
- [x] "Uncertain" is true if and only if the best score is below the configured threshold

**Verification:** `pytest tests/model/test_classifier.py -v`

**Dependencies:** M1

**Files:** `src/etf_classifier/model/classifier.py`, `tests/model/test_classifier.py`

**Size:** M

---

## M3: tuning on internal validation and final retraining

**Description:** `tune(train_df, pipeline_factory, settings)` splits training again by group (80/20), chooses `C` by macro F1 and sets the thresholds, then retrains on the whole training set.

**Acceptance criteria:**
- [x] `tune()` only receives the training set (checked by a signature and behaviour test)
- [x] On a synthetic dataset, the tuned model clearly beats the trivial baseline in macro F1
- [x] Same data and same seed, same parameters and same predictions

**Verification:** `pytest tests/model/test_tuning.py -v`

**Dependencies:** M2

**Files:** `src/etf_classifier/model/classifier.py`, `tests/model/test_tuning.py`, `tests/model/synthetic.py`

**Size:** M

---

## Checkpoint A
- [ ] Tests green, including the synthetic dataset and test-set non-use
- [ ] Quick review with you of the scores obtained on the synthetic dataset

---

## M4: influential features and evaluation report

**Description:** `explain.py` (10 most positive coefficients per category, with the readable feature name); Markdown report `reports/model_<version>.md` (global metrics, comparison with the baseline, details per category, influential features, categories that cannot be learned).

**Acceptance criteria:**
- [x] On the synthetic dataset, a category's influential features are indeed the words and tickers that define it
- [x] The report contains all the sections of the spec
- [x] A category without a positive training example is listed as impossible to learn

**Verification:** `pytest tests/model/test_explain.py tests/model/test_report.py -v`

**Dependencies:** M3

**Files:** `src/etf_classifier/model/explain.py`, `src/etf_classifier/model/evaluate.py`, `tests/model/test_explain.py`, `tests/model/test_report.py`

**Size:** M

---

## M5: versioned saving, registry and current model

**Description:** `registry.py`: `save_bundle()` writes `artifacts/models/<version>/` (model, feature pipeline, `metrics.json`, `MODEL_CARD.md`); `load_bundle(version | "current")`; `registry.json` (current model, history); `promote()`.

**Acceptance criteria:**
- [x] A reloaded model gives exactly the same predictions
- [x] The first saved model becomes current; later ones only with `promote()`
- [x] Loading "current" with no registered model gives a clear message

**Verification:** `pytest tests/model/test_registry.py -v`

**Dependencies:** M4

**Files:** `src/etf_classifier/model/registry.py`, `tests/model/test_registry.py`

**Size:** M

---

## M6: `train` command

**Description:** `python -m etf_classifier.model.train --csv … [--no-azure] [--yes] [--promote]`: data preparation, features (cost confirmation unless `--yes`), tuning, single evaluation on the test set, report, saving, optional promotion.

**Acceptance criteria:**
- [x] End-to-end training on the fixture, without Azure, with report and model folder
- [x] With the fake Azure client, the embeddings block is used and its cost appears in the summary
- [x] 10,000 synthetic ETFs (local blocks) in less than 5 minutes

**Verification:** `pytest tests/model/test_train.py -v`

**Dependencies:** M5

**Files:** `src/etf_classifier/model/train.py`, `tests/model/test_train.py`

**Size:** M

---

## Checkpoint B
- [ ] Tests green, performance test included
- [ ] Review with you of an evaluation report generated on the fixture

---

## M7: loading without categories and `predict` command

**Description:** `require_categories=False` parameter in `load_dataset()` (`data` module); `python -m etf_classifier.model.predict --model current --csv new.csv [--output …]` writes a CSV of proposals (selected categories, scores of the top 5, "uncertain" flag).

**Acceptance criteria:**
- [x] A CSV without a `Category` column loads; the 90 tests of the `data` module stay green
- [x] The output CSV contains, for every ETF, at least one category, scores and the "uncertain" flag
- [x] Clear message if no current model exists

**Verification:** `pytest tests/data tests/model/test_predict.py -v`

**Dependencies:** M5

**Files:** `src/etf_classifier/data/loader.py`, `src/etf_classifier/data/validation.py`, `src/etf_classifier/model/predict.py`, `tests/model/test_predict.py`, `tests/data/test_loader.py`

**Size:** M

---

## M8: `compare` command

**Description:** `python -m etf_classifier.model.compare --csv … --experiments local,local+embeddings`: every configuration is trained on the same split; comparison table in `reports/model_comparison.md` (trivial baseline included, micro and macro F1, training time, Azure cost).

**Acceptance criteria:**
- [x] All experiments use exactly the same test set (checked by fingerprint)
- [x] The table contains the trivial baseline and every requested experiment
- [x] An unknown experiment gives a clear message listing the available experiments

**Verification:** `pytest tests/model/test_compare.py -v`

**Dependencies:** M6

**Files:** `src/etf_classifier/model/compare.py`, `config/settings.yaml`, `tests/model/test_compare.py`

**Size:** M

---

## Final checkpoint
- [ ] The 5 success criteria of `SPEC-model.md` (those on the real CSV as soon as you have it)
- [ ] Numerical target set with you after the first real training
