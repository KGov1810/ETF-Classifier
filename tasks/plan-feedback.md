# Implementation plan: `feedback` module

Reference: `docs/specs/SPEC-feedback.md` (approved on 2026-10-25) · Detailed tasks: `tasks/todo-feedback.md`

## Overview

1. **The safe journal**: locked writes, checks and category registry, cancellation and backup. The concurrency test is the critical checkpoint: no line lost or interleaved when several processes validate at the same time.
2. **The closed loop**: the `data` module merges the original CSV and the journal, then come batch import and the monitoring report.

## Architecture decisions

- **A single write point** (`store.py`): every write goes through the same function, under the lock. Other modules never touch the file directly.
- **Lock file next to the journal** (`validations.csv.lock`, `filelock` library), with a configurable timeout.
- **Validation identifier**: UTC date + short random suffix (`20261025-142233-a1b2`), readable and unique without a shared counter.
- **Journal lines identified separately** on load (`source = validation`), with `row_id`s that cannot collide with those of the original CSV. Quality checks apply identically.
- **Category registry initialised automatically** from the original CSV on first use. Historical categories are therefore known without any input.

## Change to an existing module

`data.loader`: `load_dataset()` and `prepare_dataset()` add the non-cancelled validations from the journal, read under the same lock. Without a journal, the behaviour stays strictly identical: the 92 tests of the `data` module must stay green.

## Task list

### Phase A: the safe journal
- [x] F1: configuration, `filelock` dependency and locked journal writes
- [x] F2: checks, category registry and recording a validation
- [x] F3: cancellation and daily backup

### Checkpoint A
- [x] Concurrency test green: several processes, no line lost or interleaved
- [x] Lock unavailable: clear error, no write

### Phase B: the closed loop
- [x] F4: journal merge by the `data` module
- [x] F5: batch import command
- [x] F6: monitoring report (acceptance rate, bias alert)

### Final checkpoint
- [ ] The 4 success criteria of `docs/specs/SPEC-feedback.md`
- [x] Full loop on the fixture: `predict` → import of validations → `train` that sees the new ETFs

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Unreliable lock on a network drive (Windows share, NFS) | High | Recommend a local server disk for `data/`; documented in the README |
| Journal open in Excel on Windows (file locked) | Medium | Clear "file used by another application" error; never a partial write |
| `row_id` collision between the original CSV and the journal | Medium | Separate identifier space and `source` column; dedicated test |
| Poorly calibrated bias alert threshold | Low | Configurable threshold and minimum number of validations (95 % over at least 50 validations by default) |
| Regression of the `data` module | Medium | Without a journal, identical behaviour; `data` suite re-run |

## Status (2026-10-25)

Tasks F1 to F6 coded and tested (291 green tests across the project). Concurrency: 4 processes,
60 simultaneous validations, no loss. Full loop checked on the fixture (15 → 17 ETFs at retraining).

## Open questions

None blocking.
