# Capability Map: multi-label ETF categorisation

Status: **approved** (2026-10-25) · updated on 2026-10-25: `costs` module added

## Confirmed intent

Multi-label ETF categorisation tool, trained on a CSV of 5,000 to 10,000 ETFs
(Name, Description, weighted Composition, Categories). It proposes categories for every
new ETF with a confidence score, the similar ETFs and an LLM justification (Azure OpenAI).
The team validates or corrects in Streamlit; validations enrich the CSV with traceability;
the model is retrained once a day. The project is also meant for learning ML, built in
measured steps.

## Modules

| Module id | Responsibility | Depends on |
|---|---|---|
| `data` | CSV loading, Composition/Categories parsing, quality checks, holding normalisation, near-duplicate detection, train/test split | — |
| `costs` | Token counting, cost estimate before an Azure call, log of the actual cost after each call (embedding or LLM), totals in USD | — |
| `features` | Vectorisation: text (TF-IDF, then cached Azure embeddings) and weighted composition | `data`, `costs` |
| `model` | Multi-label classifier, per-category thresholds, evaluation (micro/macro F1), versioning | `features` |
| `similarity` | Closest ETFs (weighted composition overlap + semantic proximity) | `features` |
| `feedback` | Safe writes (lock + backup), traceability columns, category creation | `data` |
| `app` | Streamlit interface: input, proposals, similar ETFs, validation | `model`, `similarity`, `feedback` |
| `llm-justify` | Azure LLM justification, suggestions for uncertain cases or new categories | `model`, `similarity`, `costs` |
| `retraining` | **Daily** retraining (once a day), comparison with the current model, promotion only if better | `model`, `feedback` |

## Build order

`data` → `costs` → `features` → `model` + `similarity` → `feedback` → `app` (v1, without LLM) → `retraining` → `llm-justify`

(2026-10-25: `retraining` moved before `llm-justify`, at your request.)

## Decisions

- CSV files remain the source of truth (no database in v1): the original CSV, never rewritten,
  and the validation journal `data/validations.csv`, merged on load (decision of 2026-10-25).
- Daily retraining triggered by the server scheduler (cron or Windows Task Scheduler), not by
  Streamlit. The job is skipped when there has been no new validation since the last training,
  and it respects the same file lock as the interface to avoid write conflicts.
- One spec per module: `SPEC-<module-id>.md`, at the project root.
- Azure OpenAI: `text-embedding-3-large` embeddings, GPT-6.1 Sol LLM, Data Zone Standard deployments. Descriptions in English.
  No budget constraint, but every Azure call goes through `costs` (estimate + actual tracking, in USD).
  Prices live in a dated configuration file and are never hard-coded.
- Repository language: English (decision of 2026-10-25).
