# Implementation plan: `costs` then `features` modules

References: `SPEC-costs.md`, `SPEC-features.md` (approved on 2026-10-25) · `CAPABILITY-MAP.md`
Detailed tasks: `tasks/todo-costs-features.md`

## Overview

1. `costs` first: it depends on nothing, and `features` uses it from the first embedding call.
2. `features` next, in two steps:
   - the **three local blocks** (TF-IDF, tickers, composition statistics) and the full pipeline,
     usable without Azure; they form the numerical baseline of the future `model` module;
   - then the **embeddings block** (Azure client, cache, costs, command line).

This order gives an end-to-end working pipeline before touching the network,
and lets us test all the Azure code with a fake client.

## Architecture decisions

- **Common block interface** (`fit`, `transform`, `feature_names`): adding or removing a block does not touch the pipeline, which makes learning comparisons easier.
- **Injectable Azure client**: `embeddings.py` receives a client as a parameter. In production it is `AzureOpenAI`; in tests, a fake client that returns deterministic vectors and a token usage. No test touches the network.
- **SQLite cache** (standard library): robust to concurrent access (application and daily job), without an extra dependency.
- **`CostTracker` called in `embeddings.py` after each batch**, never by the caller: it is impossible to forget to trace a call.
- **Pipeline saved with `joblib`** (shipped with scikit-learn), together with a readable JSON file (configuration, version, training-set fingerprint).

## Task list

### Phase A: `costs` module
- [x] A1: price list and checked loading
- [x] A2: token counting and estimates
- [x] A3: actual cost journal (`CostTracker`)
- [x] A4: cost report (CLI)

### Checkpoint A
- [x] Tests green without network
- [ ] You check the prices of `config/pricing.yaml` against your Azure contract

### Phase B: `features` without Azure
- [x] B1: dependencies, configuration and category encoding
- [x] B2: `text_tfidf` block
- [x] B3: `holdings` and `composition_stats` blocks
- [x] B4: fit/transform pipeline, saving and reloading

### Checkpoint B
- [x] No-leak and reproducibility tests green
- [x] Building the 3 blocks on 10,000 synthetic ETFs in less than 60 s (3.8 s)

### Phase C: `features` with Azure
- [x] C1: embeddings client (batches, truncation, retries), tested with a fake client
- [x] C2: SQLite cache and per-batch cost tracking
- [x] C3: `build` command line (estimate, confirmation, `--yes`) and `.env.example`

### Checkpoint C: real run
- [ ] You run `build --estimate-only` then `build` on a small sample with your Azure key
- [ ] Gap between estimate and actual cost below 2 %; second run without any Azure call

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Default prices inaccurate for your region | Medium | Dated and sourced price list; checkpoint A dedicated to checking; warning if the price list is more than 90 days old |
| GPT-6.1 Sol tokenizer not published | Low | Estimate flagged as approximate; the actual cost returned by Azure prevails (mostly relevant to `llm-justify`) |
| Azure rate limits (429 errors) on 10,000 texts | Medium | Batches of 64, retries with increasing waits, resumable thanks to the cache |
| Change of model or dimension invalidating the cache | Medium | Model and dimension included in the cache key; change subject to your approval |
| Information leak from test to training | High | Learning on training only; dedicated test with a word and a ticker present only in the test set |
| Azure key exposed | High | `.env` excluded from version control; key never read anywhere but in the client; `.env.example` without values |

## Status (2026-10-25)

Tasks A1 to C3 coded and tested (171 green tests, 1 skipped for lack of access to the tiktoken tables).
Checks remaining on your side: price-list prices (checkpoint A) and real Azure run (checkpoint C).

Deviation from the spec: approximate fallback for token counting when the tiktoken tables cannot be
downloaded (restricted network), flagged in the estimate.

## Open questions

None blocking. For checkpoint C, you will need the exact names of your Azure deployments and the API version, to enter in your `.env`.
