"""Reading the ETF CSV and orchestrating the loading steps.

The source CSV is opened read-only and is never modified.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from etf_classifier.config import load_settings
from etf_classifier.data.dedup import DedupResult, add_groups
from etf_classifier.data.normalize import load_aliases, normalize_dataset
from etf_classifier.data.split import SplitResult, make_split
from etf_classifier.data.validation import QualityLog, validate_rows


class MissingColumnsError(ValueError):
    """One or more expected columns are missing from the CSV."""


@dataclass
class LoadResult:
    """Usable dataset and quality log produced by load_dataset()."""

    data: pd.DataFrame
    log: QualityLog
    total_rows: int


def read_raw_rows(path: str | Path, settings: dict,
                  require_categories: bool = True) -> tuple[list[dict], list[dict]]:
    """Reads the CSV row by row, keeping the line number in the file.

    Returns (structurally valid rows, malformed lines). A line is malformed when its number of
    fields differs from the header (for example an unquoted ';').
    With require_categories=False (new ETFs to predict), the Category column is optional.
    """
    csv_cfg = settings["csv"]
    expected = [column for key, column in csv_cfg["columns"].items()
                if require_categories or key != "category"]
    rows: list[dict] = []
    bad_lines: list[dict] = []
    with Path(path).open(encoding=csv_cfg["encoding"], newline="") as handle:
        reader = csv.reader(handle, delimiter=csv_cfg["separator"], quotechar='"')
        header = [column.strip().lstrip("﻿") for column in next(reader, [])]
        missing = [column for column in expected if column not in header]
        if missing:
            raise MissingColumnsError(f"Columns missing from the CSV: {', '.join(missing)}")
        record = 0
        previous_line = reader.line_num
        for fields in reader:
            line = previous_line + 1  # first physical line of the record
            previous_line = reader.line_num
            if not any(field.strip() for field in fields):
                continue
            if len(fields) != len(header):
                extract = csv_cfg["separator"].join(fields)[:60]
                bad_lines.append({"record": record, "line": line, "extract": extract})
            else:
                row = dict(zip(header, fields))
                row["__line__"] = line
                row["__record__"] = record
                rows.append(row)
            record += 1
    return rows, bad_lines


def _validation_rows(settings: dict, first_record: int) -> list[dict]:
    """Active validations from the journal, as CSV rows, with row_ids following the CSV ones."""
    from etf_classifier.feedback.store import FeedbackStore  # local import: the feedback module is optional

    store = FeedbackStore(settings)
    if not store.journal_path.exists():
        return []
    cols = settings["csv"]["columns"]
    rows = []
    for offset, (position, validation) in enumerate(store.active_validations().iterrows()):
        rows.append({
            cols["name"]: validation["Name"], cols["description"]: validation["Description"],
            cols["composition"]: validation["Composition"], cols["category"]: validation["Category"],
            "__line__": int(position) + 2,  # line in the journal (header = line 1)
            "__record__": first_record + offset,
        })
    return rows


def load_dataset(path: str | Path, settings: dict | None = None, require_categories: bool = True) -> LoadResult:
    """Loads, checks and normalises the reference dataset: original CSV + active validations from the journal.

    Columns of the usable dataset: row_id, line, source ("reference" or "validation"), name, description,
    composition_raw, composition (normalised tickers), categories, coverage, n_holdings.
    Excluded rows and warnings are in LoadResult.log.
    require_categories=False loads new ETFs to predict: the category is optional, and the validation
    journal is never merged.
    """
    settings = settings or load_settings()
    rows, bad_lines = read_raw_rows(path, settings, require_categories)
    n_reference = len(rows) + len(bad_lines)
    if require_categories and "feedback" in settings:
        rows = rows + _validation_rows(settings, first_record=n_reference)
    return load_records(rows, settings, require_categories, bad_lines=bad_lines, n_reference=n_reference)


def load_records(rows: list[dict], settings: dict, require_categories: bool = True,
                 bad_lines: list[dict] | None = None, n_reference: int | None = None) -> LoadResult:
    """Checks and normalises rows already in memory (keys = CSV column names).

    Same quality rules and ticker normalisation as load_dataset(): an ETF typed in the interface
    and the same ETF read from a CSV give the same record.
    Rows whose record number is >= n_reference are tagged as coming from the validation journal.
    """
    bad_lines = bad_lines or []
    if n_reference is None:
        n_reference = len(rows) + len(bad_lines)
    data, log = validate_rows(rows, bad_lines, settings, require_categories)
    for issue in log.issues:
        if issue.row_id is not None and issue.row_id >= n_reference:
            issue.source = "validation"
    data.insert(2, "source", ["validation" if r >= n_reference else "reference" for r in data["row_id"]])
    aliases = load_aliases(settings)
    data = normalize_dataset(data, aliases, settings["normalize"]["strip_exchange_suffix"])
    return LoadResult(data=data, log=log, total_rows=len(rows) + len(bad_lines))


@dataclass
class PreparedDataset:
    """Dataset ready for training: data with group_id, quality log, groups and split."""

    data: pd.DataFrame
    log: QualityLog
    total_rows: int
    dedup: DedupResult
    split: SplitResult


def prepare_dataset(path: str | Path, settings: dict | None = None) -> PreparedDataset:
    """Entry point for the downstream modules: loading, near-duplicates, then split."""
    settings = settings or load_settings()
    loaded = load_dataset(path, settings)
    data = loaded.data.reset_index(drop=True)
    dedup = add_groups(data, settings)
    split_cfg = settings["split"]
    split = make_split(data, split_cfg["test_size"], split_cfg["seed"], split_cfg["rare_category_min"])
    return PreparedDataset(data, loaded.log, loaded.total_rows, dedup, split)
