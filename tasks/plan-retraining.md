# Implementation plan: `retraining` module

Reference: `docs/specs/SPEC-retraining.md` (approved on 2026-10-25) · Detailed tasks: `tasks/todo-retraining.md`

## Overview

1. **Foundations**: `model.train` exposes its training as a reusable function and saves what the comparison needs
   (training ETF names, dataset fingerprint); then the pure pieces: comparison and promotion rule, run history and job lock.
2. **The daily job**: `retraining.run` chains detection → training → comparison → promotion → report → history,
   with `--force` and `--dry-run`. Checkpoint: end-to-end runs on the fixture.
3. **Around the job**: Windows scripts and README, then the run history in the Monitoring page.

## Architecture decisions

- **One training code path**: `model.train.train_and_save()` is extracted from `main()`; the command and the job call it.
  The command keeps exactly the same behaviour (its tests stay green).
- **The promotion rule is a pure function** (`compare.decide`) taking two score sets and the configuration, returning
  a decision with every reason. Easy to read, easy to test, easy to change later.
- **Comparison set built from names**: the candidate's test ETFs minus the names in the current model's `train_names.json`.
  Names are the identifier already used by the duplicate check of the journal.
- **Both models predict with their own pipeline** on the comparison set; scores are computed on the categories known by both.
- **Job lock separate from the journal lock** (`retraining.lock`, `filelock`): the interface can keep validating during a
  training; the journal lock is only held while reading it (as today in `load_dataset`).
- **History in CSV** (`retraining_history.csv`), appended one line per run, consistent with the rest of the project.
- **Failures never go unnoticed**: any exception is caught at the top of the job, logged with its cause, written to the
  history as `failed`, and turned into exit code 1.

## Changes to existing modules

- `model.train`: extraction of `train_and_save()`; `train_names.json` and `dataset_fingerprint` saved with every model.
- `model.registry`: `save_bundle(..., make_current_if_first=True)` so that `--dry-run` never makes a model current.
- `app` Monitoring page: last retraining runs.
- `config/settings.yaml`: new `retraining` section.

## Task list

### Phase A: foundations
- [x] R1: reusable training, saved training names and dataset fingerprint, `retraining` settings
- [x] R2: comparison set, scores on common categories, promotion rule
- [x] R3: run history and job lock

### Checkpoint A
- [x] Tests green, `model` command unchanged
- [x] Promotion rule checked on hand-built cases

### Phase B: the daily job
- [x] R4: `retraining.run` (detection, training, comparison, promotion, report, history, exit codes)

### Checkpoint B
- [x] End to end on the fixture: first run promoted, second run skipped, run after a validation compared, `--dry-run` never promotes

### Phase C: around the job
- [x] R5: Windows scripts and README
- [x] R6: run history in the Monitoring page

### Final checkpoint
- [ ] The 5 success criteria of `docs/specs/SPEC-retraining.md`
- [ ] You register the task on your Windows machine and check a first run in the history

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Regression of `model.train` during the extraction | Medium | Pure extraction, existing `model` tests unchanged and re-run |
| Comparison set too small at the beginning (few new ETFs, small test set) | Medium | Minimum size (20 ETFs, configurable); below it, no automatic promotion, and the report says why |
| Current model trained before this module (no `train_names.json`) | Low | Comparison on the whole test set, flagged in the report (cautious: favours the current model) |
| Training runs while the team validates | Low | Journal lock only while reading; validations saved during the training go into the next day's run |
| PowerShell scripts cannot be run in the test environment (no Windows) | Medium | Scripts kept short and standard (`Register-ScheduledTask`), reviewed by reading; you run them once and check the task |
| Execution policy blocks the scripts on your machine | Low | Commands documented with `-ExecutionPolicy Bypass` for these scripts only |
| Disk usage grows (every version kept, ~120 MB each with embeddings) | Low | Size shown in the history; deletion stays a manual decision (approved) |

## Status (2026-10-25)

Phases A, B and C (R1 to R6) coded and tested: 376 green tests across the project.
The PowerShell scripts could not be run here (no Windows): they were reviewed by reading; your first run on Windows
is the remaining check. Task name uses a plain hyphen ("ETF Classifier - daily retraining") to avoid encoding issues
in Windows PowerShell 5.1.

Correction found during the end-to-end run: change detection compared the dataset only with the *current* model.
After a refused candidate, the job would have retrained (and stored a new version) every day without any change.
It now also compares with the last trained candidate (fingerprint stored in the history); a dedicated test covers it.

Known limitation: a current model trained before this module has no `train_names.json`, so it is compared on the whole
test set, which it may partly have seen; candidates can then stay refused. Retraining it once with
`python -m etf_classifier.model.train --promote` removes the limitation.

## Open questions

None blocking.
