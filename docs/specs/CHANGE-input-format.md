# Change spec: new input format, one category per ETF, ticker as identifier

Status: **approved on 2026-10-27**. Cross-cutting change: it amends the module specs listed in
[Impact per module](#impact-per-module); each of them is updated in the same change.

## Objective

The team's CSV now has the structure:

```
Ticker;Nom;Description;Classification;Composition
```

- `Ticker`, `Nom` and `Classification` are always filled.
- `Description` may be empty; when it is, a `Composition` is expected.
- `Classification` holds **one** category (plain text, no brackets).

The whole system follows this format (reference CSV, app, batch, import, prediction, validation journal), becomes
**single-category** (one category per ETF, proposed, validated and learned), and uses the **ticker as the ETF
identifier** instead of the name. The model is retrained from scratch on the new CSV: no existing journal or
model needs migrating.

## Decisions (confirmed on 2026-10-27)

| Topic | Decision |
|---|---|
| Classification | One value per ETF; the system becomes single-category everywhere |
| Ticker | Unique identifier of an ETF (duplicates, journal, retraining comparison); displayed in the app; **not** a model feature |
| Empty description | Accepted if the composition is not empty; excluded if both are empty |
| Empty composition | Accepted if the description is not empty (warning, as today) |
| Same ticker twice in the reference CSV | Both rows excluded and listed in the quality report |
| Validating a ticker already known | Allowed: the most recent active validation replaces the reference row and earlier validations |
| Scope | Every input: reference CSV, single ETF, batch, validation import, command-line prediction, journal |
| Existing data | None to keep: new journal format, first model retrained on the new CSV |

## Format details (confirmed on 2026-10-27)

1. `Composition` keeps its current format: `[{'AAPL': 5.1}, {'MSFT': 4.6}]` (weights in %).
2. Header names are exactly `Ticker`, `Nom`, `Description`, `Classification`, `Composition`, set in
   `config/settings.yaml` (`csv.columns`), UTF-8, `;` separator, as today. Column order does not matter.
3. A ticker is compared after trimming spaces and upper-casing (`ishe ` and `ISHE` are the same ETF).
4. The CSV of new ETFs (batch, prediction) has the same columns; `Classification` is optional there. When it is
   filled, it is a **classification to check** (see [Checking an established classification](#checking-an-established-classification)).
5. Internal files (journal, history, reports) keep English column names; only the input CSV is in French.

## Assumptions (confirmed on 2026-10-27)

6. Two ETFs may share a name (two share classes): no longer a warning, since the ticker identifies them.
7. The 2–3 next categories with their score remain shown in the app as alternatives, so the validator can pick
   one in a click.

## Checking an established classification

A CSV of ETFs to classify (batch in the app, `model.predict`) may carry a `Classification` already set by
someone. The model then checks it:

- **Agreement**: the model's category equals the established one. **Disagreement**: it differs. **Unknown**: the
  established category is not a category the model knows (new or misspelt), reported as such.
- The score the model gives to the established category is shown next to its own proposal, so the validator sees
  how far apart they are.
- **Batch (app)**: the queue shows disagreements and unknown ones first, then uncertain proposals, then
  agreements; every ETF is still validated one by one. The form is pre-filled with the **model's** category; the
  established one is shown beside it with a *disagreement* alert and can be picked in one click. The batch summary
  gives the number of agreements and disagreements.
- **`model.predict`**: the output CSV keeps the input columns (`Ticker`, `Nom`, `Description`, `Composition`,
  `Classification`) and adds `PredictedCategory`, `Scores` (top 3), `Uncertain`, `EstablishedScore`, `Check`
  (`agreement`, `disagreement`, `unknown`, or empty when no classification was given) and `ModelVersion`; the
  console summary counts the verdicts. Once `Classification` is completed or corrected, the file can be imported
  as validations as it is (`feedback.importer`), `PredictedCategory` giving the accepted/corrected decision.
- An established classification is never written to the journal by itself: only a human validation is.

## Rules of the new format

Applied identically to a CSV row, an ETF typed in the app and an imported validation.

| Rule | Severity |
|---|---|
| Malformed line (wrong number of fields) | exclusion (unchanged) |
| Empty `Ticker` | exclusion (new) |
| Empty `Nom` | exclusion (new) |
| Empty `Classification` (reference / import only) | exclusion (existing `no_category`) |
| Empty `Description` **and** empty `Composition` | exclusion (new) |
| Ticker present on several rows of the reference CSV | exclusion of every such row (new) |
| Unreadable composition, non-numeric or negative weight | exclusion (unchanged) |
| Empty description, composition filled | warning (unchanged) |
| Empty composition, description filled | warning (unchanged) |
| Weights summing above 105 %, ticker listed twice in a composition | warning (unchanged) |
| Duplicate name | **removed** (the ticker identifies the ETF) |

## Impact per module

| Module | Change |
|---|---|
| `data` | New columns in `settings.yaml`; `ticker` column in the dataset; rules above; `Classification` read as one plain-text category; merge of the journal **by ticker** (latest active validation wins over the reference row and earlier validations); quality report lists ticker duplicates. The internal `categories` column stays a one-element list, so the downstream code changes as little as possible. |
| `features` | Unchanged (the ticker is not a feature). The fingerprint includes the ticker. |
| `model` | One-vs-rest + per-category thresholds replaced by a **multinomial logistic regression** (softmax): exactly one category per ETF, scores summing to 1. Tuning keeps the internal validation set and the C grid, without thresholds. Metrics: **accuracy** and **macro F1** (main metric, unchanged name so the retraining rule is unchanged), plus a confusion table of the most frequent mistakes; Hamming loss, per-ETF F1 and the threshold column disappear. Trivial baseline: the most frequent category. `max_categories` disappears; `uncertain_threshold` stays (best score below it → uncertain). Prediction CSV: see [Checking an established classification](#checking-an-established-classification). Micro F1 is dropped (equal to the accuracy with one category per ETF). |
| `similarity` | Results show the ticker and the name; the neighbour vote proposes the category with the heaviest vote. |
| `feedback` | Journal gets a `Ticker` column; `Category` holds one category; a validation requires exactly one category; a known ticker is accepted (replacement) instead of refused as a duplicate name; import reads the new format. |
| `app` | Batch: established classification checked (see above). Single ETF: `Ticker` field added (required). Batch: new format. Validation form: one category (selector pre-filled with the proposal, alternatives shown with their score, *new category* still possible). History: `Ticker` column and filter. |
| `retraining` | Comparison set and `train_names.json` (renamed `train_tickers.json`) by ticker; rule unchanged (macro F1 −0.01, no category collapse, beats the baseline, ≥ 20 ETFs). |
| `llm-justify` | Schema becomes: verdict on the proposed category (`fits` / `doubtful` / `does_not_fit`) with its reason, an optional **alternative** among the known categories, an optional new category, inconsistency, confidence. *Apply the suggestion* sets the selector to the alternative. `PROMPT_VERSION` changes, so old cache entries are not reused. |
| docs | README (data format, team guide), affected module specs, test fixtures (`tests/data/fixtures/sample.csv` and synthetic data) in the new format. |

## Commands (unchanged)

```bash
pytest -v
ruff check src tests
python -m etf_classifier.data.report --csv data/raw/etfs.csv
python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure
python -m etf_classifier.retraining.run --force
```

## Testing strategy

TDD per task, red then green. New or changed tests, at minimum:
- `data`: each new rule (empty ticker, empty name, description and composition both empty, ticker duplicated in
  the CSV → both excluded), plain-text classification, merge by ticker (validation replaces the reference row;
  a later validation replaces an earlier one; cancelling the latest restores the previous state).
- `model`: exactly one category per prediction, scores sum to 1, beats the baseline on the synthetic data,
  accuracy and macro F1 in the report.
- `feedback`: more than one category refused, re-validation of a known ticker accepted, import of the new format.
- `model`/`app`: agreement, disagreement and unknown detected; queue order disagreements → uncertain → agreements.
- `app` (AppTest): ticker required, single-category form, apply the LLM alternative.
- `retraining`: the comparison set excludes the tickers the current model trained on.
- `llm`: new schema checked; an alternative outside the known categories is refused.
The whole suite stays green and offline (Azure faked).

## Boundaries

- **Always**: same rules for CSV, app and import; never rewrite the source CSV; keep the journal append-only.
- **Ask first**: any change to the promotion rule; adding a dependency; anything beyond this spec.
- **Never**: commit or push without explicit approval and a commit message; commit data files, `.env` or secrets.

## Success criteria

1. `data.report` on a CSV in the new format excludes exactly the rows breaking the rules above and lists them with
   their reason.
2. `model.train` trains on it, proposes exactly one category per ETF, and reports accuracy and macro F1 against
   the baseline.
3. In the app, an ETF without description but with a composition gets a proposal; validating it writes one
   category and its ticker to the journal; re-validating the same ticker replaces it at the next load.
4. `retraining.run` trains, compares by ticker and promotes or refuses as before.
5. Every test passes; `ruff` is clean.

## Open questions

None blocking. To revisit after the first training on real data: with no description, should the embedding text
include the top holdings (the `include_top_tickers` option exists already, off by default)?
