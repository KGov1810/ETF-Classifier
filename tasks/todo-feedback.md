# Tasks: `feedback` module

Definition of "done" for every task: green tests (`pytest -v`), no regression on the existing
modules, no network call in the tests, no commit without your approval.

---

## F1: configuration, `filelock` dependency and locked journal writes

**Description:** `feedback` section in `settings.yaml` (paths of the journal, the registry and the backups, lock timeout, number of backups kept, bias alert thresholds); `store.py`: journal schema, writing a complete line under the lock, reading under the lock, validation identifier.

**Acceptance criteria:**
- [x] A written line reads back identically; the header is written only once
- [x] Several processes writing at the same time: all lines present, none interleaved
- [x] Lock unavailable beyond the timeout: clear error, no write

**Verification:** `pytest tests/feedback/test_store.py -v`

**Dependencies:** none

**Files:** `requirements.txt`, `config/settings.yaml`, `src/etf_classifier/feedback/__init__.py`, `src/etf_classifier/feedback/store.py`, `tests/feedback/…`

**Size:** M

---

## F2: checks, category registry and recording a validation

**Description:** `checks.py` (name, composition, categories, duplicate name); `categories.py` (registry initialised from the original CSV, traced creation); `record_validation()` which checks, derives the decision (`accepted` / `corrected`) and writes.

**Acceptance criteria:**
- [x] Every check refuses with a clear message (empty name, unreadable composition, no category, unknown category without creation, ETF already present)
- [x] A category created with `create_categories=True` enters the registry with its author and date
- [x] The decision is `accepted` if the validated categories are the proposed ones, `corrected` otherwise

**Verification:** `pytest tests/feedback/test_record.py -v`

**Dependencies:** F1

**Files:** `src/etf_classifier/feedback/checks.py`, `src/etf_classifier/feedback/categories.py`, `src/etf_classifier/feedback/store.py`, `tests/feedback/test_record.py`

**Size:** M

---

## F3: cancellation and daily backup

**Description:** `cancel_validation()` appends a `cancellation` line; `active_validations()` returns the non-cancelled validations; dated backup of the journal at the first record of the day, keeping the last 30.

**Acceptance criteria:**
- [x] A cancelled validation no longer appears in the active validations; the cancellation line stays in the journal
- [x] Cancelling an unknown or already cancelled identifier: clear error
- [x] A single backup per day; beyond 30, the oldest are deleted

**Verification:** `pytest tests/feedback/test_cancel_backup.py -v`

**Dependencies:** F2

**Files:** `src/etf_classifier/feedback/store.py`, `tests/feedback/test_cancel_backup.py`

**Size:** S

---

## Checkpoint A
- [ ] Concurrency test and unavailable-lock test green
- [ ] Quick review with you of a fictitious journal

---

## F4: journal merge by the `data` module

**Description:** `load_dataset()` adds the active validations of the journal (read under the lock) to the original CSV, with the `source` column and separate `row_id`s; same quality checks.

**Acceptance criteria:**
- [x] A validated ETF appears in the reference dataset; a cancelled ETF does not
- [x] The original CSV is never modified (fingerprint); no duplicate `row_id`
- [x] Without a journal, same result as today (92 `data` tests green)

**Verification:** `pytest tests/data tests/feedback/test_merge.py -v`

**Dependencies:** F3

**Files:** `src/etf_classifier/data/loader.py`, `tests/feedback/test_merge.py`

**Size:** M

---

## F5: batch import command

**Description:** `python -m etf_classifier.feedback.importer --csv validated.csv --by "First Last" [--create-categories]`: `predict` output format completed with a `Category` column; line-by-line report.

**Acceptance criteria:**
- [x] Valid lines are recorded with the decision derived from the `PredictedCategories` and `Category` columns
- [x] Refused lines are listed with their reason, without blocking the others
- [x] The report gives the number of accepted lines, refused lines and created categories

**Verification:** `pytest tests/feedback/test_import.py -v`

**Dependencies:** F2

**Files:** `src/etf_classifier/feedback/importer.py`, `tests/feedback/test_import.py`

**Size:** S

---

## F6: monitoring report

**Description:** `python -m etf_classifier.feedback.report [--since …]` writes `reports/feedback.md`: validations per day and per validator, acceptance rate without changes, bias alert, categories often added or removed, created categories, cancellations.

**Acceptance criteria:**
- [x] Global and per-validator acceptance rates match the hand computation
- [x] Bias alert beyond the threshold (with the minimum number of validations)
- [x] Missing journal: clear message, no error

**Verification:** `pytest tests/feedback/test_report.py -v`

**Dependencies:** F3

**Files:** `src/etf_classifier/feedback/report.py`, `tests/feedback/test_report.py`

**Size:** M

---

## Final checkpoint
- [ ] The 4 success criteria of `docs/specs/SPEC-feedback.md`
- [ ] Full loop on the fixture: `predict` → import of validations → `train` that sees the new ETFs
