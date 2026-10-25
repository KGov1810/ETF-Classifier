# Tasks: `data` module

Definition of "done" for every task: green tests (`pytest tests/data -v`), no regression,
code consistent with the style of `docs/specs/SPEC-data.md`, no commit without your approval.

---

## Task 1: project skeleton and configuration

**Description:** create the spec's folder tree, the Python environment, the configuration and a fictitious test CSV covering the edge cases.

**Acceptance criteria:**
- [x] `pip install -r requirements.txt` then `pytest` run without error
- [x] `config/settings.yaml` contains the column names, the `;` separator, the encoding and all the thresholds of the spec
- [x] `tests/data/fixtures/sample.csv` (≈ 20 fictitious rows) covers every rule of the quality table

**Verification:** `pytest -q` (0 failing test); review of `settings.yaml`

**Dependencies:** none

**Files:** `requirements.txt`, `pyproject.toml`, `.gitignore`, `config/settings.yaml`, `src/etf_classifier/__init__.py`, `src/etf_classifier/config.py`, `tests/data/fixtures/sample.csv`

**Size:** M

---

## Task 2: Composition/Category parsing and minimal `load_dataset()`

**Description:** read the CSV and convert Composition into `dict[str, float]` and Category into `list[str]` with `ast.literal_eval`. `load_dataset()` returns the normalised DataFrame (`row_id`, `name`, `description`, `composition`, `categories`).

**Acceptance criteria:**
- [x] `[{'AAPL': 6.5}, {'MSFT': 6.1}]` gives `{'AAPL': 6.5, 'MSFT': 6.1}`; a ticker present twice has its weights added up
- [x] An unreadable cell raises a `ParsingError` with an excerpt of the cell
- [x] `load_dataset()` loads the fixture with the `;` separator and UTF-8 encoding

**Verification:** `pytest tests/data/test_parsing.py tests/data/test_loader.py -v`

**Dependencies:** task 1

**Files:** `src/etf_classifier/data/parsing.py`, `src/etf_classifier/data/loader.py`, `tests/data/test_parsing.py`, `tests/data/test_loader.py`

**Size:** M

---

## Task 3: quality rules, justified exclusions, `coverage` column

**Description:** apply the rules of the quality table. Invalid rows are excluded from the usable dataset and logged with their reason, warnings are traced, and the coverage (sum of the weights) is computed.

**Acceptance criteria:**
- [x] Every rule of the spec has at least one test (exclusion or warning)
- [x] `load_dataset()` returns the valid dataset and a `QualityLog` object (row_id, rule, severity, excerpt)
- [x] The `coverage` column is present; a sum > 105 generates a warning

**Verification:** `pytest tests/data/test_validation.py -v`

**Dependencies:** task 2

**Files:** `src/etf_classifier/data/validation.py`, `src/etf_classifier/data/loader.py`, `tests/data/test_validation.py`

**Size:** S

---

## Task 4: minimal quality report (CLI + Markdown)

**Description:** command `python -m etf_classifier.data.report --csv <path>` that displays and writes to `reports/data_quality.md` the number of valid rows, rows with warnings and excluded rows (with reasons), the number of categories, the count per category and the average number of categories per ETF.

**Acceptance criteria:**
- [x] The command works on the fixture and produces the Markdown file
- [x] The report's figures match those expected for the fixture (test)
- [x] Clear error code if the file or a column is missing

**Verification:** `pytest tests/data/test_report.py -v`; manual run on the fixture

**Dependencies:** task 3

**Files:** `src/etf_classifier/data/report.py`, `tests/data/test_report.py`

**Size:** S

---

## Checkpoint 1: first pass on the real CSV
- [x] All tests pass
- [ ] Report generated on the real CSV
- [ ] Review with you: exclusion rate, unexpected formats, possible spec adjustments before continuing

---

## Task 5: ticker normalisation and alias file

**Description:** normalised key per ticker (upper case, spaces); exchange suffixes kept, but removable by option; application of `config/holding_aliases.csv`; detection of similar tickers for the report.

**Acceptance criteria:**
- [x] `' aapl '` → `'AAPL'`; `'AAPL US'` stays distinct from `'AAPL'` with the option disabled, and merges with the option enabled
- [x] An alias `GOOG;GOOGL` is applied; the original ticker stays accessible
- [x] Pairs of similar tickers (same root, different suffix) are listed

**Verification:** `pytest tests/data/test_normalize.py -v`

**Dependencies:** task 2

**Files:** `src/etf_classifier/data/normalize.py`, `config/holding_aliases.csv`, `tests/data/test_normalize.py`

**Size:** S

---

## Task 6: near-duplicate detection

**Description:** overlap `Σ min(wA, wB) / min(ΣwA, ΣwB)` computed with an inverted index and sparse matrices; two ETFs are linked if the overlap is ≥ 0.80 and they have at least 3 common tickers; groups are formed by connected components (`group_id`).

**Acceptance criteria:**
- [x] Two fictitious trackers of the same index, one complete and the other limited to the top 10 holdings, are in the same group
- [x] Two ETFs sharing a single large holding are not grouped
- [x] 10,000 synthetic ETFs are processed in less than 60 s; a warning is emitted if a group exceeds 2 % of the dataset

**Verification:** `pytest tests/data/test_dedup.py -v` (the performance test is marked `@pytest.mark.slow`)

**Dependencies:** tasks 3 and 5

**Files:** `src/etf_classifier/data/dedup.py`, `tests/data/test_dedup.py`

**Size:** M

---

## Checkpoint 2
- [x] All tests pass, including the performance test
- [ ] Review with you of the 10 largest near-duplicate groups on the real CSV; threshold confirmed or adjusted

---

## Task 7: train/test split by group

**Description:** `make_split(df, test_size=0.2, seed=42)`. A group is never split in two; we try to cover on both sides every category with at least 2 groups; categories below the rarity threshold are listed.

**Acceptance criteria:**
- [x] No `group_id` is present both in training and in test
- [x] Same seed, same split (reproducibility test)
- [x] Rare categories and categories missing from the test set are returned in a summary

**Verification:** `pytest tests/data/test_split.py -v`

**Dependencies:** task 6

**Files:** `src/etf_classifier/data/split.py`, `tests/data/test_split.py`

**Size:** S

---

## Task 8: full report and check that the source CSV is not modified

**Description:** add to the report the near-duplicate groups (number, largest group), the rare categories, the split summary, the similar tickers and the most frequent tickers. Check by fingerprint (hash) that the source CSV is never modified.

**Acceptance criteria:**
- [x] The report contains all the sections of the spec
- [x] A test checks that the CSV fingerprint is identical before and after a full run
- [x] The full report on 10,000 ETFs takes less than 2 minutes

**Verification:** `pytest tests/data -v`; manual run on the real CSV

**Dependencies:** tasks 4, 5, 6 and 7

**Files:** `src/etf_classifier/data/report.py`, `tests/data/test_report.py`, `tests/data/test_integrity.py`

**Size:** M

---

## Final checkpoint
- [ ] The 4 success criteria of `docs/specs/SPEC-data.md` are met
- [x] `load_dataset()` and `make_split()` can be used directly by the `features` module
- [ ] Review with you before moving on to the `features` spec
