# Spec: `feedback` module

Status: **approved** (2026-10-25)

> **Amended on 2026-10-27** by [CHANGE-input-format](CHANGE-input-format.md): input CSV `Ticker;Nom;Description;Classification;Composition`, one category per ETF, ticker as the ETF identifier. Where this spec and the change spec differ, the change spec prevails.

## Objective

Record the team's validations **safely and traceably**, so that every validated ETF enriches
the reference dataset and feeds the daily retraining. This is the piece that closes the learning loop.

Requirements:
- **never lose or corrupt data**, even if several people validate at the same time or if retraining reads the file at the same moment;
- **trace** who validated what, when, with which model, and what the model had proposed;
- **measure** whether validators accept proposals without reviewing them. This is the bias risk identified at the start of the project: the model would end up reinforcing its own mistakes.

Users: the `app` module ("Validate" button), the `retraining` job, the team through a report.
Before the interface, a command imports a batch of validations from a CSV.

Out of scope (v1): permissions and authentication; changing the categories of historical ETFs.

## Storage

- **The original reference CSV stays intact.** Validations are appended, line after line, to a **separate journal** `data/validations.csv`, which is only ever appended to.
- **The `data` module merges both files on load**: reference dataset = original CSV + validated ETFs (minus cancelled validations). Retraining therefore automatically sees the new ETFs.
- Benefits: simple rollback, original CSV never rewritten, full traceability, short writes hence few conflicts.

Journal columns:

| Column | Content |
|---|---|
| `ValidationId` | Unique identifier of the validation |
| `Ticker`, `Name`, `Description`, `Composition`, `Category` | The ETF; `Category` = the validated category (one; amended on 2026-10-27). The latest active validation of a ticker replaces its reference row |
| `ValidatedBy` | Validator's name or e-mail |
| `ValidatedAt` | UTC date and time |
| `ModelVersion` | Version of the model that made the proposal |
| `PredictedCategory` | Category proposed by the model |
| `Decision` | `accepted` (proposal unchanged), `corrected` (categories added or removed), `cancellation` |
| `CancelsId` | For a cancellation, the cancelled validation |
| `Comment` | Optional free comment |

## Features

### 1. Recording a validation
- `record_validation(etf, categories, validated_by, prediction=None, model_version=None, comment="")`.
- Checks before writing, with clear messages:
  - non-empty name;
  - composition in the right format (same parsing as the `data` module);
  - at least one category;
  - known categories, unless explicitly created (point 3).
- **ETF already present** (same name in the reference or already validated): refused by default, with the existing record. The validator can confirm that it is a different ETF.
- The decision (`accepted` or `corrected`) is derived by comparing with the proposed categories.

### 2. Safe writing
- **File lock** held during every write and while the `data` module reads the journal. It works on Windows, Linux and macOS.
- One complete line written in a single operation; UTF-8 file, `;` separator.
- **Daily backup** of the journal in `data/backups/` (the last 30 kept).
- The lock has a timeout: past it, a clear error ("another record is being saved, try again"), never a write without the lock.

### 3. Categories
- Registry `data/categories.csv`: the known categories (historical and created), with who created them, when, and a description.
- An unknown category is only accepted with `create_categories=True`. Its creation is recorded.
- Information displayed: a new category will only be proposed by the model after the next retraining, and provided it has enough examples.

### 4. Cancellation
- `cancel_validation(validation_id, cancelled_by, comment)` appends a `cancellation` line. Nothing is ever deleted: the history stays complete, and the cancelled ETF disappears from the reference dataset on the next load.

### 5. Batch import (before the interface)
`python -m etf_classifier.feedback.importer --csv validated.csv --by "First Last" [--create-categories]`
- Input format: the input CSV format, `Classification` = the validated category; the output of `model.predict` can be imported as it is once `Classification` is completed or corrected (amended on 2026-10-27).
- Every line goes through the same checks. The import report shows the accepted and refused lines, with the reason.

### 6. Monitoring report
`python -m etf_classifier.feedback.report [--since …]` writes `reports/feedback.md`:
- number of validations per day and per validator;
- **acceptance rate without changes**, global and per validator; a rate close to 100 % over a long period is flagged as a bias risk;
- categories most often added or removed by validators, which indicates the model's weaknesses;
- new categories created, and cancelled validations.

## Tech Stack
- **New dependency**: `filelock` (file lock portable across Windows, Linux and macOS)
- Existing: pandas, PyYAML

## Structure
```
data/validations.csv            # validation journal (not committed, like data/raw/)
data/categories.csv             # category registry (not committed)
data/backups/                   # daily backups (not committed)
src/etf_classifier/feedback/
├── store.py                    # journal: locked writes, cancellation, reading, backup
├── checks.py                   # checks before writing
├── categories.py               # category registry
├── importer.py                 # batch import CLI
└── report.py                   # monitoring report CLI
tests/feedback/
```
Change to an existing module: `data.loader` merges the original CSV and the journal (non-cancelled validations), under the same lock.

## Test strategy
- Checks: empty name, unreadable composition, no category, unknown category without creation, ETF already present.
- **Concurrent writes**: several processes validate at the same time; no line lost or interleaved.
- Lock unavailable beyond the timeout: clear error, no write.
- Cancellation: the ETF disappears from the reference dataset, the cancellation line stays in the journal.
- Merge by `data`: validated ETFs appear, with the same quality checks as the original CSV; the original CSV is never modified (fingerprint).
- Report: acceptance rates computed by hand on a fictitious journal, bias alert.
- Import: accepted and refused lines with their reason.

## Boundaries
- **Always:** write under the lock; append, never rewrite or delete; record who, when and with which model.
- **Ask first:** changing the journal format; touching the original CSV.
- **Never:** write without the lock; delete a line from the journal; commit `data/`.

## Success criteria
1. A recorded validation appears in the reference dataset on the next load, hence at the next retraining.
2. No loss or corruption during simultaneous writes (concurrency test).
3. The monitoring report gives the acceptance rate per validator and flags a bias risk.
4. A batch validated outside the interface can be imported, with a line-by-line report.

## Decisions made
- **Storage B**: separate journal `data/validations.csv`, merged on load with the original CSV, which stays intact.
- **`filelock` dependency** accepted (portable lock for Windows / Linux / macOS).
- **Corrections of historical ETFs out of v1**; they may come later as `correction` lines in the journal.
- **Validator identity**: name or e-mail, without authentication in v1.
- **Additions during implementation**:
  - `no_proposal` decision for a validation entered without a model proposal (excluded from the acceptance rate);
  - the import command is called `feedback.importer` (`import` is a Python reserved word);
  - the output of `model.predict` now contains `Description`, `Composition` and `ModelVersion`, so that the
    file completed by the team is self-sufficient.
