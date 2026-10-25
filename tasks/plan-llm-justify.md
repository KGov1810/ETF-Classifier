# Implementation plan: `llm-justify` module

Reference: `SPEC-llm-justify.md` (approved on 2026-10-25) · Detailed tasks: `tasks/todo-llm-justify.md`

## Overview

1. **The core without interface**: context and prompt, structured answer checks, Azure call with cost estimate,
   cost record and cache, then the justification store. Everything tested with a fake chat client.
2. **The interface**: button, cost confirmation, answer panel and "Apply the suggestions" in the *Single ETF* and *Batch*
   screens; the justification linked to the validation and shown in the *History* screen; README.

## Architecture decisions

- **Same pattern as the embeddings**: the Azure client is created from `.env` in one place (`llm/client.py`) and injected
  everywhere else, so that tests use a fake client returning a prepared JSON answer and a token usage.
- **Pure functions for the risky parts**: `build_context(proposal, known_categories, settings)` and
  `check_answer(raw_json, known_categories)` have no side effect and are tested in isolation.
- **Two-step call, as for the proposals**: `estimate(...)` returns the cost; `justify(..., confirmed=True)` is the only path
  that calls Azure. A cached answer needs no confirmation and costs nothing.
- **Cache keyed by the prompt fingerprint** (prompt version + deployment + full messages), stored in SQLite
  (`cache/llm.sqlite`, same approach as the embeddings cache).
- **Justification store in JSON Lines**, appended under a file lock; the link to a validation is a second line
  (`{"type": "link", "justification_id": …, "validation_id": …}`), so that nothing is ever rewritten.
- **Structured outputs** with a strict JSON schema; the answer is still re-checked on receipt (schema and known categories).

## Changes to existing modules
- `app`: button and panel in the *Single ETF* and *Batch* screens, apply suggestions, link at validation, *History* display.
- `config/settings.yaml` (`llm` section), `.env.example` (`AZURE_OPENAI_CHAT_DEPLOYMENT`), `README.md`.

## Task list

### Phase A: core
- [x] L1: settings, prompt and context, answer schema and checks
- [x] L2: Azure chat client, estimate, call, cost record, cache
- [x] L3: justification store (append, link to a validation, read)

### Checkpoint A
- [x] Tests green with the fake client; no call without confirmation; second identical request served by the cache

### Phase B: interface
- [x] L4: button, confirmation, answer panel and "Apply the suggestions" (*Single ETF* and *Batch*), link at validation
- [x] L5: justification in the *History* screen, README

### Final checkpoint
- [ ] The 5 success criteria of `SPEC-llm-justify.md`
- [ ] You try it with your GPT-6.1 Sol deployment (the only step that needs the real Azure)

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Structured outputs not supported by your deployment or API version | Medium | Fallback to JSON mode (`response_format: json_object`) with the schema in the prompt; the answer is checked either way |
| GPT-6.1 Sol tokenizer not published: the estimate can differ from the real cost | Low | Estimate flagged as approximate; the real cost from Azure is recorded |
| Reasoning tokens make the answer slow or long | Low | `max_completion_tokens` limit; spinner in the interface; clear message if the limit is reached |
| A description contains text that tries to steer the model | Low | ETF data marked as data in the system message; suggestions checked against the known categories; the validator decides |
| The model invents a category | Medium | Moved to "new category" or dropped, never shown as existing |
| Real Azure behaviour not testable here | Medium | Fake client in every test; your manual try at the final checkpoint |

## Status (2026-10-25)

L1 to L5 coded and tested with a fake chat client (no network).
Corrections during the work:
- `data/llm_justifications.jsonl*` was not covered by `.gitignore` (the spec assumed it was): rule added.
- the category selection of the validation form is now initialised through the session state (no `default=`),
  so that "Apply the suggestions" can change it without a Streamlit warning.
- full test suite hang found and fixed: the new `tests/llm/test_store.py` had the same module name as
  `tests/feedback/test_store.py` (renamed `test_justification_store.py`), and `AppTest` left the last page as `__main__`,
  which the `spawn` workers of the feedback concurrency test re-ran at start-up (restored after each app test);
  the concurrency test now fails after 120 s instead of hanging if a worker cannot start.
410 green tests across the project.
Remaining: your try with the real GPT-6.1 Sol deployment.

## Open questions

None blocking.
