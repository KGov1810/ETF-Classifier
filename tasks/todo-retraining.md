# Tasks: `retraining` module

Definition of "done" for every task: green tests (`pytest -v`), no regression on the existing
modules, no network call in the tests, no commit without your approval.

---

## R1: reusable training, saved training names and dataset fingerprint, `retraining` settings

**Description:** extract `train_and_save(prepared, settings, embedder, promote=False, make_current_if_first=True)` from `model.train.main()`; save `train_names.json` (training ETF names) and `dataset_fingerprint` (whole prepared dataset) with every model; `save_bundle(..., make_current_if_first=True)`; `retraining` section in `settings.yaml`.

**Acceptance criteria:**
- [x] The `model` tests pass unchanged; `python -m etf_classifier.model.train` behaves as before
- [x] A trained model folder contains `train_names.json` with exactly the training ETF names, and `metrics.json` contains `dataset_fingerprint`
- [x] The same data gives the same fingerprint; adding or cancelling a validation changes it

**Verification:** `pytest tests/model tests/retraining/test_fingerprint.py -v`

**Dependencies:** none

**Files:** `src/etf_classifier/model/train.py`, `src/etf_classifier/model/registry.py`, `config/settings.yaml`, `src/etf_classifier/retraining/__init__.py`, `tests/retraining/…`

**Size:** M

---

## R2: comparison set, scores on common categories, promotion rule

**Description:** `compare.py`: `comparison_set(candidate_test, current_bundle)`; `score(bundle, data, categories)` (macro F1 and per-category F1 on the given categories); `decide(current, candidate, baseline, n_etfs, cfg) -> Decision(promote, reasons)`.

**Acceptance criteria:**
- [x] ETFs in the current model's training names are excluded; without `train_names.json`, the whole test set is used and flagged
- [x] Categories unknown to one of the two models are excluded from the macro F1 and listed as new categories
- [x] `decide` covers: within tolerance → promote; macro F1 too low, a category collapsing (named), below the trivial baseline, comparison set too small → refused with the reason

**Verification:** `pytest tests/retraining/test_compare.py -v`

**Dependencies:** R1

**Files:** `src/etf_classifier/retraining/compare.py`, `tests/retraining/test_compare.py`

**Size:** M

---

## R3: run history and job lock

**Description:** `history.py`: `job_lock(settings)` (context manager, clear error if a run is in progress); `append_run(settings, record)`; `read_history(settings)`.

**Acceptance criteria:**
- [x] A second lock attempt while the first is held fails immediately with a clear message
- [x] Every appended run reads back identically; the header is written once
- [x] Missing history: empty table, no error

**Verification:** `pytest tests/retraining/test_history.py -v`

**Dependencies:** R1

**Files:** `src/etf_classifier/retraining/history.py`, `tests/retraining/test_history.py`

**Size:** S

---

## Checkpoint A
- [x] Tests green, `model` command unchanged
- [x] Promotion rule checked on hand-built cases

---

## R4: `retraining.run`

**Description:** CLI `python -m etf_classifier.retraining.run [--force] [--dry-run] [--config …]`: job lock → change detection → candidate training (embeddings paid without confirmation) → comparison → promotion or not → report `reports/retraining_<version>.md` → history line → exit code (0 normal outcomes, 1 failure).

**Acceptance criteria:**
- [x] First run without a current model → promoted; second run without change → `skipped`, nothing trained
- [x] After a new validation → candidate trained, compared, promoted or not according to the rule; the report explains the decision
- [x] `--dry-run` never promotes (not even a first model); a failure (unreadable CSV) writes a `failed` history line and returns 1

**Verification:** `pytest tests/retraining/test_run.py -v`

**Dependencies:** R2, R3

**Files:** `src/etf_classifier/retraining/run.py`, `tests/retraining/test_run.py`

**Size:** M

---

## Checkpoint B
- [x] End to end on the fixture: promoted, skipped, compared, dry run
- [ ] Review with you of a retraining report

---

## R5: Windows scripts and README

**Description:** `scripts/run_daily_retraining.ps1` (project folder, `.venv`, job, log), `scripts/register_daily_retraining.ps1 [-Time 10:00]` (task for the current user, only when logged on, start when available, no parallel run), `scripts/unregister_daily_retraining.ps1`; README section "Daily retraining".

**Acceptance criteria:**
- [x] The scripts contain no secret and no absolute path specific to this environment
- [x] The task is created for the current user, daily at the chosen time, with "start when available" and "do not start a new instance"
- [x] The README explains how to register, check and remove the task, and where to read the history

**Verification:** review of the scripts; `pytest -v` (no regression); your run on Windows at the final checkpoint

**Dependencies:** R4

**Files:** `scripts/run_daily_retraining.ps1`, `scripts/register_daily_retraining.ps1`, `scripts/unregister_daily_retraining.ps1`, `README.md`

**Size:** S

---

## R6: run history in the Monitoring page

**Description:** *Current model* tab of the Monitoring page: last runs (date, outcome, versions, scores, reasons, cost) from the history.

**Acceptance criteria:**
- [x] The last runs appear, newest first
- [x] No history yet: clear message
- [x] The other tabs are unchanged (existing page tests green)

**Verification:** `pytest tests/app -v`

**Dependencies:** R3

**Files:** `src/etf_classifier/app/pages/4_Monitoring.py`, `src/etf_classifier/app/services.py`, `tests/app/test_pages.py`

**Size:** S

---

## Final checkpoint
- [ ] The 5 success criteria of `docs/specs/SPEC-retraining.md`
- [ ] You register the task on Windows and check a first run in the history
