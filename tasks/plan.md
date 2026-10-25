# Implementation plan: `data` module

Reference: `SPEC-data.md` (approved on 2026-10-25) · `CAPABILITY-MAP.md`

## Overview

Build the module that loads the ETF CSV (`Name;Description;Composition;Category`), checks it,
normalises tickers, groups near-duplicates and produces a train/test split without leaks.
We work in vertical slices: from task 4 onwards, a command produces a report on the real CSV,
which surfaces format surprises very early.

## Architecture decisions

- **Quality report from phase 1**: the first risk is the gap between the assumed format and the real CSV. The report acts as a full-scale test and grows with every task.
- **Pure functions + orchestrator**: `parsing`, `validation`, `normalize`, `dedup` and `split` neither read nor write files; only `loader` and `report` do I/O. This makes testing and reuse by `features` and `retraining` easier.
- **Centralised configuration** in `config/settings.yaml`, read once and passed explicitly to functions.
- **Sparse matrices (scipy) + inverted index** for near-duplicates: only ETF pairs sharing at least one ticker are compared.
- **No dependency outside the spec**: pandas, numpy, scipy, PyYAML, pytest.

## Task list

Full details in `tasks/todo.md`.

### Phase 1: end-to-end loading
- [x] Task 1: project skeleton and configuration
- [x] Task 2: Composition/Category parsing and minimal `load_dataset()`
- [x] Task 3: quality rules, justified exclusions, `coverage` column
- [x] Task 4: minimal quality report (CLI + Markdown)

### Checkpoint 1: first pass on the real CSV
- [ ] Tests green
- [ ] Report generated on the real CSV; exclusion rate and format anomalies reviewed with you

### Phase 2: making the data reliable
- [x] Task 5: ticker normalisation and alias file
- [x] Task 6: near-duplicate detection (weighted overlap, groups)

### Checkpoint 2
- [ ] Tests green, performance test (10,000 ETFs in less than 60 s) passed
- [ ] Review with you of the near-duplicate groups on the real CSV (sample of the 10 largest groups)

### Phase 3: split and finalisation
- [x] Task 7: train/test split by group
- [x] Task 8: full report and check that the source CSV is not modified

### Final checkpoint
- [ ] The 4 success criteria of `SPEC-data.md` are met
- [ ] Review with you before moving on to the `features` module

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| The real CSV departs from the assumed format (quotes, `;` in descriptions, weights as text) | High | Report from task 4; every rejected row shows its reason and an excerpt |
| Chaining effect in near-duplicates: A≈B, B≈C… creates a giant group | Medium | Size of the largest group in the report; warning above 2 % of the dataset; adjustable threshold |
| Near-duplicate detection too slow | Medium | Inverted index + sparse matrices; dedicated performance test |
| Rare categories missing from the test set | Medium | List of categories below the threshold in the report; possible merging decided with you |
| Inclusion of partial compositions: an ETF listing only a few large holdings can be grouped with a broader ETF that contains them (seen on the fixture: World ESG with US Tech) | Medium | To be examined at checkpoint 2 on the real CSV; if needed, an extra condition (minimum coverage, holding-count ratio) to be approved with you |
| Confidentiality of the real CSV | High | `data/raw/` excluded from version control; tests only on fictitious data |

## Status (2026-10-25)

Tasks 1 to 8 coded and tested (90 green tests). Checkpoints 1 and 2 waiting for the real CSV.

## Open questions

1. **Where will the code live?** Either here, in this workspace, and I send you the files; or in a folder on your computer linked through the Claude desktop app, where I write directly.
2. **The real CSV**: will you be able to provide it for checkpoint 1, or would you rather run the report yourself and send me the result (which only contains statistics and excerpts of rejected rows)?
