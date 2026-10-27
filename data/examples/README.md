# Example data

Data to try the whole project on about 1,750 real ETFs, without the team's CSV.

| File | Rows | Content |
|---|---|---|
| `etfs_train.csv` | 1,492 | Reference dataset: train a model on it |
| `etfs_to_check.csv` | 263 | ETFs never seen in training, whose `Classification` must be checked: right (about 55 %), wrong (30 %), empty (10 %) or unknown to the grid (5 %) |
| `etfs_to_check_answers.csv` | 263 | Answer key: true classification, given classification, what was changed |

Format: the project's input format (`Ticker;Nom;Description;Classification;Composition`, UTF-8, `;`).
About 5 % of the rows have no description (composition only), as in a real file.

## What is real, what is not

- **Real**: ticker, name, description and classification of US-listed, active, non-leveraged ETFs, from
  [FinanceDatabase](https://github.com/JerBouma/FinanceDatabase) (MIT licence, see `LICENSE-FinanceDatabase.txt`).
  The classification combines its asset class and segment, e.g. `Fixed Income – Municipal Bonds` (23 classifications
  with at least 30 ETFs each, at most 150 per classification).
- **Fictitious**: the composition. The open database has no holdings, so `scripts/make_example_data.py` draws
  plausible top holdings (real security tickers, random weights) according to the classification, with the mega caps
  shared by most equity funds. They are **not** the ETFs' actual holdings, and they make the classification somewhat
  easier than real holdings would: compare with the text-only experiment below.

## Try it

```bash
python -m etf_classifier.data.report --csv data/examples/etfs_train.csv
python -m etf_classifier.model.train --csv data/examples/etfs_train.csv --no-azure
python -m etf_classifier.model.predict --csv data/examples/etfs_to_check.csv --output reports/checked.csv
python -m etf_classifier.model.compare --csv data/examples/etfs_train.csv --experiments local,text_only,holdings_only
```

Then compare `reports/checked.csv` (columns `PredictedCategory` and `Check`) with `etfs_to_check_answers.csv`, or
upload `etfs_to_check.csv` in the **Batch** page of the app: disagreements come first.

To use it as the app's reference dataset, set `feedback.reference_csv: data/examples/etfs_train.csv` in
`config/settings.yaml` (the default points to `data/raw/etfs.csv`).

## Results on 2026-10-27 (local blocks, no Azure)

| | Macro F1 | Accuracy |
|---|---|---|
| Trivial baseline | 0.006 | 0.070 |
| Text only (real data) | 0.769 | 0.802 |
| Composition only (fictitious) | 0.865 | 0.852 |
| Text + composition | 0.937 | 0.933 |

On `etfs_to_check.csv`: the model proposes the true classification for 94 % of the ETFs; it flags 77 of the 79
wrong classifications as disagreements (and proposes the true one for 74 of them), with 8 false alarms out of 141
right classifications, mostly between neighbouring bond segments (Blend, Corporate, Investment Grade).

## Rebuild

```bash
curl -L -o etfs.bz2 https://raw.githubusercontent.com/JerBouma/FinanceDatabase/main/compression/etfs.bz2
python scripts/make_example_data.py --source etfs.bz2 --output data/examples
```
The selection and the fictitious parts are seeded: the same source gives the same files.
