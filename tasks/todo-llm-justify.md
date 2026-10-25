# Tasks: `llm-justify` module

Definition of "done" for every task: green tests (`pytest -v`), no regression on the existing
modules, no network call in the tests, no commit without your approval.

---

## L1: settings, prompt and context, answer schema and checks

**Description:** `llm` section in `settings.yaml`; `AZURE_OPENAI_CHAT_DEPLOYMENT` in `.env.example`; `llm/prompts.py`: `PROMPT_VERSION`, system message, `build_context(proposal, known_categories, settings)`, `build_messages(context)`, JSON schema; `check_answer(raw, known_categories)`.

**Acceptance criteria:**
- [x] The context holds at most 10 holdings and 5 neighbours, the proposed categories with scores and the known categories, and nothing else
- [x] A valid answer is accepted; an unknown suggested category is moved to the new category (or dropped if one is already proposed), with a note
- [x] An invalid answer (not JSON, missing field, wrong verdict) raises a clear error

**Verification:** `pytest tests/llm/test_prompts.py -v`

**Dependencies:** none

**Files:** `config/settings.yaml`, `.env.example`, `src/etf_classifier/llm/__init__.py`, `src/etf_classifier/llm/prompts.py`, `tests/llm/…`

**Size:** M

---

## L2: Azure chat client, estimate, call, cost record, cache

**Description:** `llm/client.py` (client from `.env`, deployment name, structured-output call with JSON-mode fallback, retries); `llm/justify.py`: `estimate(proposal, …)`, `justify(proposal, …, confirmed)` with cache, `CostTracker.record(module="llm-justify", operation="justify")`.

**Acceptance criteria:**
- [x] No call without `confirmed=True` (unless the answer is cached); the estimate is returned before
- [x] One cost line per call with the tokens returned; an identical second request makes no call and costs nothing
- [x] Without Azure configuration or without a chat deployment: clear error, no call

**Verification:** `pytest tests/llm/test_justify.py -v`

**Dependencies:** L1

**Files:** `src/etf_classifier/llm/client.py`, `src/etf_classifier/llm/justify.py`, `tests/llm/fake_chat.py`, `tests/llm/test_justify.py`

**Size:** M

---

## L3: justification store

**Description:** `llm/store.py`: append a justification (date, prompt version, deployment, model version, ETF name, answer, tokens, cost) under a file lock; `link(justification_id, validation_id)`; `for_validation(validation_id)`.

**Acceptance criteria:**
- [x] One JSON line per call, read back identically
- [x] Linking adds a line (nothing rewritten); reading by validation returns the linked justification
- [x] Missing file: empty result, no error

**Verification:** `pytest tests/llm/test_store.py -v`

**Dependencies:** L1

**Files:** `src/etf_classifier/llm/store.py`, `tests/llm/test_store.py`

**Size:** S

---

## Checkpoint A
- [x] Tests green with the fake client
- [x] No call without confirmation; identical request served by the cache

---

## L4: interface in the *Single ETF* and *Batch* screens

**Description:** `services.llm_estimate / llm_justify / apply_suggestions`; `components.justification_panel`; button "Explain with GPT-6.1 Sol", cost confirmation, answer panel, "Apply the suggestions" (selection and new-category fields); justification linked when the ETF is validated.

**Acceptance criteria:**
- [x] No call before the confirmation; after it, the panel shows summary, verdicts, suggestions, new category, inconsistency, confidence
- [x] "Apply the suggestions" updates the selection (and the new-category fields); the validator can still change everything
- [x] Validating after a justification links it to the `ValidationId`; without Azure the button is disabled with an explanation

**Verification:** `pytest tests/app tests/llm -v`

**Dependencies:** L2, L3

**Files:** `src/etf_classifier/app/services.py`, `src/etf_classifier/app/components.py`, `src/etf_classifier/app/pages/1_Single_ETF.py`, `src/etf_classifier/app/pages/2_Batch.py`, `tests/app/test_llm_pages.py`

**Size:** M

---

## L5: *History* screen and README

**Description:** in the *History* screen, the justification of the selected validation (when there is one); README section "LLM justification".

**Acceptance criteria:**
- [x] A validation with a justification shows it; one without shows nothing special
- [x] The README explains the button, the cost confirmation, what is sent to Azure and where the justifications are kept

**Verification:** `pytest -v`

**Dependencies:** L4

**Files:** `src/etf_classifier/app/pages/3_History.py`, `src/etf_classifier/app/services.py`, `tests/app/test_llm_pages.py`, `README.md`

**Size:** S

---

## Final checkpoint
- [ ] The 5 success criteria of `SPEC-llm-justify.md`
- [ ] You try it with your GPT-6.1 Sol deployment
