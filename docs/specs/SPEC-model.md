# Spec: `model` module

Status: **approved** (2026-10-25)

> **Amended on 2026-10-27** by [CHANGE-input-format](CHANGE-input-format.md): input CSV `Ticker;Nom;Description;Classification;Composition`, one category per ETF, ticker as the ETF identifier. Where this spec and the change spec differ, the change spec prevails.

## Objective

Train, evaluate and version the **single-category classifier** (amended on 2026-10-27; first version: multi-label)
that proposes one category for every ETF, with a score for every category.

This is the learning core of the project. We move forward through measured comparisons:
1. a **trivial baseline** (predict the most frequent categories), which serves as a floor;
2. a **reference model** on the three local blocks (TF-IDF, tickers, statistics);
3. the same model **with the Azure embeddings**, to measure their real contribution.

Users: the `app` module (proposals to validate), `llm-justify` (uncertain cases), `retraining`
(daily training and comparison), and you, through readable evaluation reports.

Out of scope: the automatic promotion rule for a new model (`retraining` module),
the interface (`app` module), deep-learning models.

## Input data
- `PreparedDataset` (`data` module): train/test split by near-duplicate group.
- `FeaturePipeline` (`features` module): feature matrices and category matrix.

## Features

### 1. Model
- **One multinomial logistic regression** (softmax over the categories; amended on 2026-10-27, first version:
  one logistic regression per category), on the combined matrix of the active blocks.
  - Rationale: fast on sparse matrices with tens of thousands of columns, usable probabilities, interpretable coefficients (we can show which words or tickers push towards a category).
  - `class_weight="balanced"` so as not to sacrifice rare categories.
- **Tuning on an internal validation set**: the training set is split again (80/20, by near-duplicate group) to:
  - choose the regularisation strength `C` from a small grid (0.1 · 1 · 10);
  - (first version only) set a decision threshold per category: removed on 2026-10-27, the best-scored category is proposed.
- The final model is then retrained on the whole training set, with the selected parameter.
- **The test set is used only once**, for the final evaluation. It is never used for tuning.
- Category without any positive example in training: cannot be learned, reported in the report.

### 2. Prediction
- `predict(df) -> list[Prediction]`. For every ETF:
  - the **scores of all categories** (probabilities), sorted;
  - the **proposed category**: the best-scored one (scores sum to 1);
  - an **"uncertain"** flag if the best score is below a confidence threshold (configurable, 0.5 by default). This is the signal for the interface and for `llm-justify`: hard case or possible new category.
- `python -m etf_classifier.model.predict --model current --csv new_etfs.csv` writes a CSV of proposals (selected categories, scores, uncertain flag). Useful before the interface arrives.

### 3. Evaluation
Computed on the test set and written to `reports/model_<version>.md`:
- **accuracy** (share of ETFs given the right category);
- **macro F1** (every category counts equally: reveals poorly learned rare categories);
- the most frequent **confusions** (true category → proposed category); micro F1, per-ETF F1 and Hamming loss of the
  first version are dropped (amended on 2026-10-27);
- **per category**: precision, recall, F1, support, selected threshold;
- **comparison with the trivial baseline**, on the same test set;
- **most influential features per category** (10 most positive coefficients): the report shows for example that "gold", "NEM" and "miners" push towards "Gold".

### 4. Compared experiments
`python -m etf_classifier.model.compare --csv … --experiments local,local+embeddings` trains several
block configurations **on the same split** and produces a comparison table
(`reports/model_comparison.md`): micro and macro F1, training time and Azure cost of each configuration.

### 5. Versioning
- Every training produces `artifacts/models/<version>/` containing:
  - the model and **its own feature pipeline** (a single folder to load to predict);
  - `metrics.json` (metrics, parameters, thresholds, training-set fingerprint, blocks used);
  - a readable card `MODEL_CARD.md`.
- `artifacts/models/registry.json` indicates the **current** model (`current`) and the history.
- `train --promote` makes the new model current. The very first model is current automatically. Conditional automatic promotion will come with `retraining`.

### 6. Training command line
```
python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure          # local reference
python -m etf_classifier.model.train --csv data/raw/etfs.csv --yes --promote     # with embeddings
```
Embeddings go through the cost confirmation of the `features` module, except with `--yes`.

## Tech Stack
No new dependency: scikit-learn, numpy, scipy, pandas, joblib (already present).

## Structure
```
src/etf_classifier/model/
├── classifier.py      # training, C tuning, per-category thresholds, prediction
├── baseline.py        # trivial baseline (most frequent categories)
├── evaluate.py        # metrics and evaluation report
├── explain.py         # most influential features per category
├── registry.py        # saving, loading, current model
├── train.py           # training CLI
├── compare.py         # experiment comparison CLI
└── predict.py         # prediction CLI on a CSV
tests/model/
```

## Code style
Same as the previous modules. Fixed random seed everywhere (`random_state` from `settings.yaml`).

## Test strategy
- **Synthetic dataset with a known answer** (categories determined by certain words and tickers): the model must clearly beat the trivial baseline.
- **Thresholds**: on hand-built scores, the chosen threshold does maximise the category's F1.
- **"At least one category" rule** and "uncertain" flag.
- **Test set not used for tuning**: the tuning function only receives the training set (signature and behaviour test).
- **Reproducibility**: same data and same seed, same predictions.
- **Registry**: saving, reloading, identical predictions; designation of the current model.
- **Performance**: training on 10,000 synthetic ETFs (local blocks) in less than 5 minutes.

## Boundaries
- **Always:** tune on the internal validation set, evaluate only once on the test set; compare with the trivial baseline; save model and pipeline together.
- **Ask first:** changing the algorithm or adding a dependency; changing the main metric; changing the "at least one category" rule.
- **Never:** use the test set to choose a parameter or a threshold; commit `artifacts/`.

## Success criteria
1. On your real CSV, the local reference model clearly beats the trivial baseline in micro and macro F1.
2. The comparison report quantifies the contribution (or not) of the embeddings.
3. `predict` produces for every ETF at least one category, scores and the "uncertain" flag.
4. A saved model reloads and gives exactly the same predictions.
5. Tests pass without network.

A **numerical target** (for example micro F1 ≥ 0.85) cannot honestly be set without your data. I propose to define it with you after the first training on the real CSV.

## Decisions made
- **Main metric: macro F1** (every category counts equally), for the choice of `C` and model comparison; micro F1 is still displayed.
- **At most 5 categories proposed** per ETF; the scores of the other categories remain visible.
- **"Uncertain" flag threshold: 0.5** to start with, to be adjusted after the first training on the real CSV.
