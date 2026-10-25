# Implementation plan: `app` module

Reference: `SPEC-app.md` (approved on 2026-10-25) · Detailed tasks: `tasks/todo-app.md`

## Overview

1. **The service layer first, without Streamlit**: in-memory ETFs through the same checks as the CSV,
   then proposal (model + neighbours + cost confirmation), then validation, cancellation and history.
   All business logic is tested with pytest before any page exists.
2. **The pages next, as vertical slices**: the "Single ETF" screen end to end (the riskiest flow:
   editable table → proposal → validation → journal), then the batch queue, the history and the monitoring page.

## Architecture decisions

- **`services.py` knows nothing about Streamlit**: plain functions and dataclasses (`Proposal`, `CostCheck`…).
  Pages only display results and call services. This is what makes the app testable without a browser.
- **`load_records(rows, settings, require_categories)`** extracted from `data.loader.load_dataset()`:
  it runs `validate_rows` then ticker normalisation on rows built in memory. `load_dataset()` calls it,
  so a typed ETF and a CSV ETF are handled by exactly the same code.
- **Two-step proposal when the model uses embeddings**: `services.estimate(...)` returns the cost of new texts
  (excluding the cache); `services.propose(..., confirmed=True)` is the only path that calls Azure.
  A model without embeddings skips the confirmation.
- **Model cached per process** (`st.cache_resource`) keyed by the current version; the sidebar button clears it.
- **Session state limited to UI state** (validator, batch queue, current position). Validations live only in the journal.
- **Tests of the pages with `streamlit.testing.v1.AppTest`**, on a model trained on the fixture in a temporary folder
  (settings pointing to `tmp_path`), never on `artifacts/` or `data/`.

## Change to an existing module

`data.loader`: extraction of `load_records()`. Without a journal and with a CSV, `load_dataset()` behaves strictly
as today; the 92 `data` tests and the whole suite (291 tests) must stay green.

## Task list

### Phase A: services
- [ ] P1: `streamlit` dependency, `app` settings section and `load_records()` extraction
- [ ] P2: proposal service (model, neighbours, near-duplicate, cost confirmation)
- [ ] P3: validation, cancellation and history services

### Checkpoint A
- [ ] Tests green, `data` suite unchanged
- [ ] A typed ETF and the same ETF read from a CSV give the same proposal

### Phase B: pages
- [ ] P4: home page, sidebar and "Single ETF" screen
- [ ] P5: "Batch" screen (CSV upload and queue)
- [ ] P6: "History" screen (filters and cancellation)
- [ ] P7: "Monitoring" screen and README

### Final checkpoint
- [ ] The 5 success criteria of `SPEC-app.md`
- [ ] Manual run on `localhost:8501` with a model trained on the fixture: you try the 4 screens

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Streamlit reruns the script on every interaction: the proposal or the queue could be recomputed or lost | Medium | Proposal stored in session state, keyed by the ETF content; model cached with `cache_resource` |
| Accidental Azure spending on every rerun | High | Azure is only called by `propose(confirmed=True)` after an explicit button; cache for already-seen texts; dedicated test |
| Double click on "Validate" writes two lines | Medium | Second attempt refused by the existing duplicate-name check; button disabled once validated |
| Regression of the `data` module during the extraction | Medium | Small extraction, default behaviour unchanged, full suite re-run |
| `AppTest` cannot drive every widget (file upload, editable table) | Low | Those inputs are tested at the service level; `AppTest` covers rendering and the main buttons |
| Streamlit version drift | Low | Minimum version pinned in `requirements.txt` (`streamlit>=1.40`) |

## Open questions

None blocking.
