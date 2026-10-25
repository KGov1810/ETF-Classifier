# Spec: `costs` module

Status: **approved** (2026-10-25)

## Objective

Make every Azure OpenAI expense of the project visible and under control:
- **estimate before** a call or a batch of calls ("computing the embeddings of 8,000 ETFs will cost about $0.05");
- **record after** each call the actual cost, from the number of tokens returned by Azure;
- **summarise** expenses per day, per module (`features`, `llm-justify`…) and per operation.

Users: the `features` and `llm-justify` modules (calls), the `retraining` job (non-interactive estimate), the team through a report, and later the Streamlit interface.

Out of scope: reconciliation with the Azure invoice (Cost Management API), budgets and alerts (no budget constraint), "Provisioned" pricing (PTU).

## Input data

1. **Price list** `config/pricing.yaml`, in **USD per million tokens**, dated and sourced:
   ```yaml
   currency: USD
   as_of: "2026-10-25"
   source: "Public Data Zone Standard prices (upper end of the regional range), to be checked against your Azure contract"
   models:
     text-embedding-3-large:
       kind: embedding
       tokenizer: cl100k_base
       input_per_million: 0.13
     gpt-6.1-sol:
       kind: chat
       tokenizer: o200k_base          # approximation: tokenizer not published
       input_per_million: 2.40
       cached_input_per_million: 0.12
       output_per_million: 12.00       # includes reasoning tokens
       long_context_threshold: 272000  # above it, long-context pricing
       long_context:
         input_per_million: 4.80
         cached_input_per_million: 0.24
         output_per_million: 18.00
   deployments:                        # Azure deployment name → price-list model
     text-embedding-3-large: text-embedding-3-large
     gpt-6.1-sol: gpt-6.1-sol
   ```
   The default values are the public **Data Zone Standard** prices, at the upper end of the regional range (cautious estimate).
   **They must be checked** against your contract: the exact price depends on the region. The Data Zone price of `text-embedding-3-large` has not been confirmed by a public source ($0.13 is the standard price).
2. **Usage returned by Azure** after each call: `prompt_tokens`, `completion_tokens`, `prompt_tokens_details.cached_tokens` when present.

## Features

### 1. Token counting
- `count_tokens(text, model) -> int`, with the tokenizer declared in the price list (`tiktoken` library).
- For `text-embedding-3-large`, the `cl100k_base` tokenizer is exact. For GPT-6.1 Sol it is not published: the count is an **approximation**, flagged as such. The actual cost recorded after the call always prevails.

### 2. Estimate before a call
- `estimate_embeddings(texts, deployment) -> CostEstimate`: tokens and cost in USD for a batch of texts.
- `estimate_chat(messages, deployment, expected_output_tokens) -> CostEstimate`: input cost exact up to the tokenizer, output cost from an explicit assumption.
- `CostEstimate` holds: number of tokens, cost in USD, date of the price list used, "approximate" flag.

### 3. Recording the actual cost
- `CostTracker.record(module, operation, deployment, usage, metadata=None)` computes the cost and appends a line to the journal `logs/azure_usage.csv` (`;` separator, UTF-8):

  | Column | Example |
  |---|---|
  | `timestamp_utc` | `2026-10-25T14:03:11Z` |
  | `module` | `features` |
  | `operation` | `embeddings_batch` |
  | `deployment` | `text-embedding-3-large` |
  | `input_tokens` / `cached_input_tokens` / `output_tokens` | `48211` / `0` / `0` |
  | `cost_usd` | `0.006267` |
  | `pricing_as_of` | `2026-10-25` |
  | `estimated_cost_usd` | prior estimate, if any |
  | `note` | batch id, number of texts… |
- Long-context pricing applies when the input tokens of a call exceed the threshold.
- **Deployment missing from the price list**: the call is still recorded, with its tokens, but with an empty `cost_usd` and a warning. A trace is never lost, and a fake zero cost is never shown.
- Writing: one complete line per write, in append mode. The file lock shared with the interface comes with the `feedback` module.

### 4. Cost report
`python -m etf_classifier.costs.report [--since 2026-10-01] [--until …]` displays and writes to `reports/azure_costs.md`:
- the total in USD over the period;
- totals per day, per module, per operation and per deployment;
- the gap between estimates and actual costs, when both exist;
- the date of the price list used, with a warning if it is more than 90 days old.

## Tech Stack
- Python 3.11+, pandas, PyYAML (already present)
- **New dependency**: `tiktoken` (token counting)

## Commands
```
pytest tests/costs -v
python -m etf_classifier.costs.report --since 2026-10-01
```

## Structure
```
config/pricing.yaml                 # dated price list (committed)
logs/azure_usage.csv                # call journal (not committed)
src/etf_classifier/costs/
├── pricing.py                      # price-list loading and checks
├── tokens.py                       # token counting
├── estimate.py                     # estimates before a call
├── tracker.py                      # actual cost recording
└── report.py                       # cost report (CLI)
tests/costs/
```

## Code style
Same as the `data` module: short typed functions, docstrings in English, thresholds and prices only in the configuration. Amounts computed as `float` and rounded to 6 decimals in the journal.

## Test strategy
- Cost computations checked by hand: 1,000,000 embedding tokens cost exactly $0.13 with the test price list.
- Long-context pricing, cached tokens, output tokens.
- Unknown deployment: the line is recorded without a cost, with a warning.
- Invalid price list (missing or negative price): clear error on load.
- Report: totals per day and per module on a fictitious journal.
- No network call in the tests.

## Boundaries
- **Always:** go through `CostTracker` for every Azure call; date the price list.
- **Ask first:** changing the default price list; adding a dependency other than `tiktoken`.
- **Never:** hard-code a price; commit `logs/`; record the content of the texts sent in the journal (only the counters).

## Success criteria
1. Every Azure call of the project produces a line in the journal.
2. On an embeddings batch, the prior estimate and the actual cost differ by less than 2 %.
3. The report gives the total per day, per module and per operation.
4. Tests pass without network access.

## Decisions made
- LLM: **GPT-6.1 Sol**, **Data Zone Standard** deployment.
- Currency: USD. Default price list at the upper end of Data Zone prices, to be adjusted to your region.
