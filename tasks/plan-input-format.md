# Plan: new input format, one category per ETF, ticker as identifier

Spec: [docs/specs/CHANGE-input-format.md](../docs/specs/CHANGE-input-format.md) (approved on 2026-10-27).
Tasks: [todo-input-format.md](todo-input-format.md).

## Architecture decisions

- **One-element list internally.** The dataset keeps `categories` as a list holding exactly one category. Labels,
  per-category F1, the neighbour vote and the retraining comparison then keep working without a rewrite; only
  the edges change (CSV reading, journal, model, form, LLM). It is the least intrusive path.
- **Ticker normalised once**, in `data.validation` (trim spaces, upper-case), and used as the key everywhere
  downstream (journal merge, `train_tickers.json`, comparison set, similarity results).
- **Journal merge by ticker** in `data.loader`: reference rows, then active validations in journal order; the last
  row of a ticker wins. Cancelling the latest validation therefore restores the previous one, with no extra code.
  Ticker duplicates *inside the reference CSV* are excluded before the merge (both rows).
- **Multinomial logistic regression** (`LogisticRegression` with `lbfgs`, `class_weight="balanced"`): one softmax
  over the categories, argmax = proposal. The thresholds disappear; tuning keeps the C grid on the internal
  validation set. The main metric keeps the name `f1_macro`, so the retraining rule and the history do not change.
- **Established classification check** in one pure function (`model.check`): used by `model.predict` and by the
  app services, so the CLI and the batch page give the same verdicts.
- **No migration**: the user starts from scratch; `train_names.json` becomes `train_tickers.json` and an old
  model without it falls back to "whole test set" (existing behaviour).

## Dependency graph

```
settings + fixtures + synthetic data (T1)
        │
data rules + ticker (T2) ──► journal by ticker + import (T3) ──► quality report + fingerprint (T4)
        │
single-category model (T5) ──► train_tickers + retraining by ticker (T6)
        │                  └─► established classification check + predict CLI (T7)
        │
app services (T8) ──► app pages (T9)
similarity display (T10)     LLM single category (T11)
        │
docs + end-to-end check (T12)
```

## Phases and checkpoints

1. **Data** (T1–T4). Checkpoint: `pytest tests/data tests/feedback`, a new-format CSV gives the expected
   exclusions in `data.report`.
2. **Model** (T5–T7). Checkpoint: `pytest tests/model tests/retraining tests/features tests/similarity`;
   `model.train --no-azure` on the fixture proposes one category per ETF.
3. **App and LLM** (T8–T11). Checkpoint: full suite; the app opened locally on the fixture.
4. **Docs and end to end** (T12). Checkpoint: full suite, `ruff`, README and specs consistent; commit proposed.

During phases 1–2 some tests of later modules will fail (they still use the old format); each phase fixes the
tests of the modules it touches, and the whole suite is green at the end of phase 3 at the latest.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Large change, many tests to rewrite | High | Phases with checkpoints; the one-element-list choice keeps most code and tests valid |
| The multinomial model scores worse than the one-vs-rest on rare categories | Medium | `class_weight="balanced"`; the report compares with the baseline; the C grid is unchanged |
| Synthetic data becoming too easy with one category | Low | Keep the noise words and tickers; the "beats the baseline" tests stay meaningful |
| Tests depend on names being unique (dedup, split, similarity) | Medium | Synthetic data and fixture get unique tickers; tests search for `"name"` usages in phase 1 |
| Old cached LLM answers in the old schema | Low | `PROMPT_VERSION` changes, so the cache key changes |

## Open questions

None.
