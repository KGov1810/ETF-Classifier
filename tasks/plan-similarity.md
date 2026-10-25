# Implementation plan: `similarity` module

Reference: `SPEC-similarity.md` (approved on 2026-10-25) · Detailed tasks: `tasks/todo-similarity.md`

## Overview

1. **The core**: the index (composition and text vectors), neighbour search with its explanation and the near-duplicate alert, then the neighbour vote and saving.
2. **Integration**: the index built by `model.train` and reloaded with the model, then the `search` and `evaluate` commands.

## Architecture decisions

- **Ticker vocabulary specific to the index**, built on the whole reference set. It does not reuse the model's vocabulary, learned on training only: otherwise tickers present only in test ETFs would be invisible to the search.
- **The norm of a searched ETF includes all its tickers**, even those absent from the reference. Without this, an ETF whose holdings are half unknown would look as close as a fully covered ETF.
- **Text vectors taken from the model's pipeline**: the embeddings if the model uses them, otherwise TF-IDF. ETFs are therefore compared with the same representation that was used to categorise them.
- **Matrix computations** (sparse product for composition, dense or sparse product for text): no loop over the 10,000 reference ETFs.
- **Self-exclusion by row identifier** (`row_id`), not by name, since ETF names can be duplicated.

## Changes to existing modules

- `model.train` builds the index on all usable rows and saves it in the model folder (`similarity.joblib`).
- `model.registry` reloads it with the model. A model trained before this module has no index: clear message asking to retrain it.

## Task list

### Phase A: the core
- [x] S1: configuration and index building
- [x] S2: neighbour search, explanation and near-duplicate alert
- [x] S3: neighbour vote, saving and reloading

### Checkpoint A
- [x] Tests green, including the "small ETF contained in a large one" case
- [x] 100 queries on a reference of 10,000 ETFs in less than 10 s

### Phase B: integration
- [x] S4: index built by `train` and reloaded with the model
- [x] S5: `search` command
- [x] S6: `evaluate` command (neighbour vote against the model and the trivial baseline)

### Final checkpoint
- [ ] The 4 success criteria of `SPEC-similarity.md`
- [ ] After the first real training: composition/text weights and use of the vote decided with you

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Disk size: the index of a model with embeddings weighs about 120 MB (10,000 × 3,072 numbers); with one retraining per day, versions pile up | Medium | Measured and shown in the model card; retention policy (keep the last N versions) to plan in the `retraining` module |
| Existing models without an index | Low | Clear message on load; `search` and `evaluate` say to retrain |
| Duplicate ETF names | Low | Self-exclusion by `row_id` |
| Biased evaluation of the vote | Medium | `evaluate` builds the index on training only; same test set as the model |
| Embeddings of new ETFs | Low | Same cache and same cost confirmation as `features`, `train` and `predict` |

## Status (2026-10-25)

Tasks S1 to S6 coded and tested (251 green tests across the project). 100 queries on 10,000 ETFs: 1.7 s.

Spec correction: the rule "text only if either ETF has no composition" put the scores
on different scales (an ETF without composition ranked ahead of ETFs sharing the same tickers).
Now: text only if the searched ETF has no composition; otherwise, a neighbour without composition
has a composition similarity of 0 in the combined score.

Known limitation: the near-duplicate alert reuses the rule of the `data` module (overlap relative to the smallest
coverage), so a small ETF contained in a large one can be flagged. To be reviewed with the real CSV.

## Open questions

None blocking.
