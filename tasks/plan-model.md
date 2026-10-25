# Implementation plan: `model` module

Reference: `docs/specs/SPEC-model.md` (approved on 2026-10-25) · Detailed tasks: `tasks/todo-model.md`

## Overview

Three phases:
1. **The core**: metrics and trivial baseline, then the classifier (thresholds, "at least one category" rule, "uncertain" flag), then tuning on internal validation. At the end of this phase, we check on a synthetic dataset that the model clearly beats the baseline.
2. **Traceability**: influential features and evaluation report, versioned saving with a registry, then the `train` command that chains everything.
3. **Usage**: `predict` on a CSV of new ETFs, and `compare` to measure the contribution of the embeddings.

## Architecture decisions

- **Tuning only receives the training set**: the `tune()` function has no access to the test set, which makes a leak impossible by construction, and a test checks it.
- **Internal validation by group**: we reuse `make_split()` from the `data` module on the training set, so that near-duplicates do not skew tuning either.
- **One folder per model** containing model and feature pipeline: the application and the daily job load a single folder, with no risk of mixing versions.
- **Evaluation separate from the model** (`evaluate.py`): the trivial baseline and the classifier are evaluated by exactly the same code.

## Change to an existing module

`predict` must read a CSV of **new ETFs, without a `Category` column**. Today, the `data` module excludes a row without a category. I will add a `require_categories` parameter (true by default) to `load_dataset()`. The current behaviour stays unchanged, and the 90 existing tests must stay green.

## Task list

### Phase A: the core
- [x] M1: evaluation metrics and trivial baseline
- [x] M2: classifier, per-category thresholds and prediction rules
- [x] M3: tuning on internal validation (`C`, thresholds) and final retraining

### Checkpoint A
- [x] On a synthetic dataset, the model clearly beats the trivial baseline (macro F1)
- [x] Test-set non-use test green

### Phase B: traceability
- [x] M4: influential features and evaluation report
- [x] M5: versioned saving, registry and current model
- [x] M6: `train` command

### Checkpoint B
- [x] End-to-end training on the fixture; reloading with identical predictions
- [x] Training on 10,000 synthetic ETFs (local blocks) in less than 5 minutes

### Phase C: usage
- [x] M7: loading without categories and `predict` command
- [x] M8: `compare` command

### Final checkpoint
- [ ] The 5 success criteria of `docs/specs/SPEC-model.md`, including those on the real CSV, once you have it
- [ ] Numerical target defined with you after the first real training

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Rare categories: too few examples in validation to tune a reliable threshold | Medium | Default threshold of 0.5 below a minimum count (configurable); flagged in the report |
| Training time with the embeddings block (3,072 dense columns × 10,000 ETFs) | Medium | Measured by the performance test; solver suited to dense matrices if needed; `C` grid limited to 3 values |
| Leak from the test set into tuning | High | `tune()` only receives training; dedicated test |
| Regression of the `data` module when adding `require_categories` | Low | Optional parameter, default value unchanged; `data` suite re-run |
| Biased comparison between experiments | Medium | `compare` uses the same split and the same seed for all configurations |

## Status (2026-10-25)

Tasks M1 to M8 coded and tested (225 green tests across the project).
Synthetic dataset: macro F1 0.969 versus 0.067 for the trivial baseline. Training on 10,000 ETFs: 4.3 s.
Remaining: first training on the real CSV and numerical target to set with you.

Adjustments: extraction of `announce_embedding_plan` / `confirm_spending` in `features/build.py`
(reused by train, predict, compare) and addition of `FeaturePipeline.attach_embedder`.

## Open questions

None blocking.
