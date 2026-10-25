# Spec: `data` module

Status: **approved** (2026-10-25)

## Objective

Provide the other modules with a **reliable, clean and reproducible** dataset from the ETF CSV:
- read and convert the Composition and Categories columns into Python structures;
- detect and report invalid rows without ever modifying the source CSV;
- normalise holding names so that "Apple", "Apple Inc." and "APPLE INC" are recognised as a single security;
- group near-duplicates (ETFs tracking the same index);
- produce a train/test split that avoids leaks between these groups.

Users of the module: the `features`, `model`, `similarity`, `feedback` and `retraining` modules, and you, through a readable quality report.

## Input data (confirmed)

1. CSV encoded in **UTF-8**, **semicolon** separator (`;`). These settings remain editable in `settings.yaml`.
   Cells that themselves contain a `;`, for example in a description, must be quoted; otherwise the row is reported as malformed.
2. Columns: `Name`, `Description`, `Composition`, `Category`.
3. Composition: `[{'AAPL': 6.5}, {'MSFT': 6.1}]`. Holdings are **tickers**, with weights in percent and a decimal point.
4. Category: `['Global Equity', 'ESG']`, from 1 to N categories of different kinds per ETF.
5. **Variable coverage**: the composition is complete for ETFs with few holdings, and limited to the top holdings for the others. The sum of the weights can therefore range from about 10 to 100.

## Features

### 1. Loading and parsing
- `load_dataset(path) -> pd.DataFrame` returns the columns `name`, `description`, `composition`, `categories`, plus `row_id`, the position of the row in the source CSV.
- `composition` is converted to `dict[str, float]` with `ast.literal_eval`. `eval` is never used, and the format is not strict JSON. The list of one-key dictionaries becomes a single dictionary.
- `categories` is converted to `list[str]`. Spaces around each category are stripped and duplicates removed.

### 2. Quality checks
Every invalid row is **excluded from the usable dataset and listed in a report** with its reason, but it is never deleted from the CSV. The rules are:

| Rule | Handling |
|---|---|
| Composition or Categories unreadable | Excluded |
| No category | Excluded |
| Negative or non-numeric weight | Excluded |
| Sum of weights > 105 | Warning |
| Empty composition | Kept, with a warning (the description alone will be used) |
| Sum of weights < 100 | Normal (partial composition); the coverage is stored in a `coverage` column |
| Empty description | Kept, with a warning |
| Holding present twice in the same ETF | Weights added up, with a warning |
| Duplicate ETF name | Warning |

### 3. Ticker normalisation
- Cautious rules: upper-casing and removal of superfluous spaces.
- **Exchange suffixes are not stripped by default** (` US`, `.PA`, ` UW Equity`…). The same ticker can designate different companies depending on the exchange, and a wrong merge would be worse than no merge. Stripping suffixes remains an option to enable in `settings.yaml`.
- An optional file `config/holding_aliases.csv` (columns `raw;canonical`) lets you fix special cases by hand, for example `GOOG` and `GOOGL`.
- The report lists similar tickers (same root, different suffix) to help you fill in this file.
- The original ticker is kept; only the normalised key is used in computations.

### 4. Near-duplicates
- The overlap between two ETFs A and B is `Σ min(wA, wB) / min(ΣwA, ΣwB)`, computed over the common holdings.
  We divide by the **smallest coverage** rather than by 100 because of partial compositions. Two MSCI World trackers limited to their top 10 holdings (about 20 % of the weight) would otherwise have an overlap of about 0.20 and would never be grouped.
- Two ETFs whose overlap is ≥ **0.80** (approved threshold, configurable) belong to the same group. Groups are transitive (connected components).
- Safeguard: grouping requires at least 3 tickers in common, so that a single large shared holding is not enough to bring two poorly diversified ETFs together.
- For performance, only pairs sharing at least one holding are compared, thanks to an inverted index and sparse matrices, rather than the 50 million possible pairs.

### 5. Train/test split
- 80/20 by default, with a fixed random seed for reproducibility.
- **Group split**: all ETFs of the same near-duplicate group go to the same side.
- The aim is for every category with at least 2 groups to appear on both sides, without an extra library in v1.
- Categories that are too rare (fewer than 5 ETFs, configurable threshold) are listed in the report.

### 6. Quality report
`python -m etf_classifier.data.report` displays:
- the number of valid rows, rows with warnings and excluded rows, with the reason;
- the number of categories, the count per category and the average number of categories per ETF;
- the number of near-duplicate groups and the size of the largest one;
- the most frequent holdings before and after normalisation.

The report is also written to `reports/data_quality.md`.

## Tech Stack
- Python 3.11 or later
- pandas, numpy, scipy (sparse matrices), PyYAML
- pytest

## Commands
```
python -m venv .venv
source .venv/bin/activate        # Linux/macOS  (.venv\Scripts\activate on Windows)
pip install -r requirements.txt
pytest tests/data -v
python -m etf_classifier.data.report --csv data/raw/etfs.csv
```

## Project structure
```
etf-classifier/
├── docs/CAPABILITY-MAP.md
├── docs/specs/SPEC-data.md
├── config/
│   ├── settings.yaml          # column names, separator, thresholds
│   └── holding_aliases.csv    # manual fixes (optional)
├── data/raw/etfs.csv          # not committed (.gitignore)
├── reports/                   # generated reports (not committed)
├── src/etf_classifier/data/
│   ├── loader.py              # CSV reading + orchestration
│   ├── parsing.py             # Composition / Categories
│   ├── validation.py          # quality rules
│   ├── normalize.py           # tickers
│   ├── dedup.py               # near-duplicates
│   ├── split.py               # group split
│   └── report.py              # quality report (CLI)
└── tests/data/
    ├── fixtures/sample.csv    # ~20 fictitious rows covering the edge cases
    └── test_*.py
```

## Code style
Short typed functions, docstrings in English, no hidden side effects.
```python
def parse_composition(raw: str) -> dict[str, float]:
    """Convert "[{'Apple': 6.5}, ...]" into {'Apple': 6.5, ...}.

    Raise ParsingError if the cell is not readable.
    """
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ParsingError(f"Unreadable composition: {raw[:50]}") from exc
    ...
```
- Naming: `snake_case` for functions and variables, `PascalCase` for classes, constants in upper case.
- Configuration is read from `settings.yaml`; no threshold is hard-coded.
- **Operating-system independent** code (`pathlib`, no OS-specific path or command), since the server has not been chosen yet.

## Test strategy
- pytest, a fictitious CSV fixture covering every rule of the quality-check table.
- Unit tests for `parsing`, `validation`, `normalize`, `dedup` and `split`.
- Properties checked:
  - no group leak between training and test;
  - identical split for the same seed;
  - the source CSV is never modified (fingerprint comparison before and after).
- Performance test: near-duplicate detection on 10,000 synthetic ETFs takes less than 60 seconds on a standard machine.

## Boundaries
- **Always:** run the tests before proposing a commit; keep the source CSV read-only; log every exclusion.
- **Ask first:** adding a dependency; changing the quality rules or default thresholds; changing the CSV format.
- **Never:** commit `data/raw/` or real data; commit or push without explicit approval; use `eval`.

## Success criteria
1. Your full CSV loads, and every excluded row has an explicit reason.
2. The quality report is generated in less than 2 minutes for 10,000 ETFs.
3. Tests pass, including those on the absence of group leaks and on reproducibility.
4. Downstream modules can call `load_dataset()` and `make_split()` without any further processing.

## Decisions made
- Columns `Name;Description;Composition;Category`, `;` separator, UTF-8 encoding.
- Near-duplicates grouped by weighted overlap, threshold 0.80, relative to the smallest coverage.
- Holdings identified by ticker; exchange suffixes kept by default.
- Variable composition coverage, measured in the `coverage` column.
- Server and Git not chosen yet: portable code. The scheduler choice is deferred to the `retraining` module.

## Questions deferred to later modules
- **`features`**: the team has no ticker → sector/country table. In v1, composition features will rely only on tickers and their weights. An enrichment (exchange suffix as a country hint, or a table built later) will be studied in the `features` spec.
- **`retraining`**: the scheduler will depend on the operating system chosen for the server.
