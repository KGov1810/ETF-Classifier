"""Quality rules applied to each CSV row.

An invalid row is excluded from the usable dataset and logged with its reason;
it is never removed from the source CSV.
"""

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from etf_classifier.data.parsing import (
    ParsingError,
    WeightError,
    parse_categories,
    parse_composition_items,
    to_weight,
)

EXCLUSION = "exclusion"
WARNING = "warning"

DATA_COLUMNS = ["row_id", "line", "name", "description", "composition", "categories", "coverage", "n_holdings"]

RULE_LABELS = {
    "malformed_line": "Malformed line (wrong number of fields, unquoted ';'?)",
    "unreadable_composition": "Unreadable composition",
    "unreadable_categories": "Unreadable categories",
    "no_category": "No category",
    "non_numeric_weight": "Non-numeric weight",
    "negative_weight": "Negative weight",
    "empty_composition": "Empty composition (the description alone will be used)",
    "empty_description": "Empty description",
    "duplicate_ticker": "Ticker listed several times (weights summed)",
    "weight_sum_too_high": "Sum of weights above the allowed maximum",
    "duplicate_name": "Duplicate ETF name",
}


@dataclass
class QualityIssue:
    """A quality issue attached to a CSV row."""

    row_id: int | None
    line: int | None
    rule: str
    severity: str
    message: str
    extract: str = ""
    source: str = "reference"  # "reference" (original CSV) or "validation" (validation journal)


@dataclass
class QualityLog:
    """Log of the exclusions and warnings produced while loading."""

    issues: list[QualityIssue] = field(default_factory=list)

    def add(self, row_id, line, rule, severity, message="", extract="") -> None:
        self.issues.append(QualityIssue(row_id, line, rule, severity, message or RULE_LABELS[rule], extract[:60]))

    def exclusions(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == EXCLUSION]

    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    def counts(self) -> Counter:
        """Number of issues per (severity, rule)."""
        return Counter((issue.severity, issue.rule) for issue in self.issues)


def _validate_row(row: dict, row_id: int, line: int, settings: dict, log: QualityLog,
                  require_categories: bool = True) -> dict | None:
    """Checks one row; returns the usable record, or None if the row is excluded."""
    cols = settings["csv"]["columns"]
    raw_composition = row[cols["composition"]]
    raw_categories = row.get(cols["category"], "")

    try:
        items = parse_composition_items(raw_composition)
    except ParsingError as exc:
        log.add(row_id, line, "unreadable_composition", EXCLUSION, str(exc), raw_composition)
        return None

    try:
        categories = parse_categories(raw_categories)
    except ParsingError as exc:
        log.add(row_id, line, "unreadable_categories", EXCLUSION, str(exc), raw_categories)
        return None
    if not categories and require_categories:
        log.add(row_id, line, "no_category", EXCLUSION, extract=raw_categories)
        return None

    composition: dict[str, float] = {}
    for ticker, raw_weight in items:
        try:
            weight = to_weight(raw_weight)
        except WeightError as exc:
            log.add(row_id, line, "non_numeric_weight", EXCLUSION, f"{ticker}: {exc}", raw_composition)
            return None
        if weight < 0:
            log.add(row_id, line, "negative_weight", EXCLUSION, f"{ticker}: {weight}", raw_composition)
            return None
        if ticker in composition:
            log.add(row_id, line, "duplicate_ticker", WARNING, f"Duplicate ticker: {ticker}", raw_composition)
        composition[ticker] = composition.get(ticker, 0.0) + weight

    coverage = sum(composition.values())
    if not composition:
        log.add(row_id, line, "empty_composition", WARNING, extract=raw_composition)
    if coverage > settings["quality"]["max_weight_sum"]:
        log.add(row_id, line, "weight_sum_too_high", WARNING, f"Sum of weights: {coverage:.2f}", raw_composition)

    description = row[cols["description"]].strip()
    if not description:
        log.add(row_id, line, "empty_description", WARNING, extract=row[cols["name"]])

    return {
        "row_id": row_id,
        "line": line,
        "name": row[cols["name"]].strip(),
        "description": description,
        "composition": composition,
        "categories": categories,
        "coverage": coverage,
        "n_holdings": len(composition),
    }


def validate_rows(rows: list[dict], bad_lines: list[dict], settings: dict,
                  require_categories: bool = True) -> tuple[pd.DataFrame, QualityLog]:
    """Applies every rule; returns (usable dataset, quality log)."""
    log = QualityLog()
    for bad in bad_lines:
        log.add(bad.get("record"), bad.get("line"), "malformed_line", EXCLUSION, extract=bad.get("extract", ""))

    records = []
    for position, row in enumerate(rows):
        row_id = row.get("__record__", position)
        record = _validate_row(row, row_id, row.get("__line__"), settings, log, require_categories)
        if record is not None:
            records.append(record)

    seen_names: set[str] = set()
    for record in records:
        if record["name"] in seen_names:
            log.add(record["row_id"], record["line"], "duplicate_name", WARNING, extract=record["name"])
        seen_names.add(record["name"])

    return pd.DataFrame(records, columns=DATA_COLUMNS), log
