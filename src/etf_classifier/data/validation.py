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
    normalize_etf_ticker,
    parse_category,
    parse_composition_items,
    to_weight,
)

EXCLUSION = "exclusion"
WARNING = "warning"

DATA_COLUMNS = ["row_id", "line", "ticker", "name", "description", "composition", "categories", "coverage",
                "n_holdings"]

RULE_LABELS = {
    "malformed_line": "Malformed line (wrong number of fields, unquoted ';'?)",
    "missing_ticker": "Empty ticker",
    "missing_name": "Empty name",
    "duplicate_etf_ticker": "Ticker present on several rows (every such row excluded)",
    "no_description_no_composition": "Neither description nor composition",
    "unreadable_composition": "Unreadable composition",
    "no_category": "No classification",
    "non_numeric_weight": "Non-numeric weight",
    "negative_weight": "Negative weight",
    "empty_composition": "Empty composition (the description alone will be used)",
    "empty_description": "Empty description",
    "duplicate_ticker": "Ticker listed several times (weights summed)",
    "weight_sum_too_high": "Sum of weights above the allowed maximum",
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
    ticker = normalize_etf_ticker(row.get(cols["ticker"]))
    name = str(row.get(cols["name"]) or "").strip()
    description = str(row.get(cols["description"]) or "").strip()
    raw_composition = row.get(cols["composition"]) or ""
    category = parse_category(row.get(cols["category"]))
    if not ticker:
        log.add(row_id, line, "missing_ticker", EXCLUSION, extract=name)
        return None
    if not name:
        log.add(row_id, line, "missing_name", EXCLUSION, extract=ticker)
        return None
    if not category and require_categories:
        log.add(row_id, line, "no_category", EXCLUSION, extract=ticker)
        return None

    try:
        items = parse_composition_items(raw_composition)
    except ParsingError as exc:
        log.add(row_id, line, "unreadable_composition", EXCLUSION, str(exc), raw_composition)
        return None
    if not items and not description:
        log.add(row_id, line, "no_description_no_composition", EXCLUSION, extract=ticker)
        return None

    composition: dict[str, float] = {}
    for holding, raw_weight in items:
        try:
            weight = to_weight(raw_weight)
        except WeightError as exc:
            log.add(row_id, line, "non_numeric_weight", EXCLUSION, f"{holding}: {exc}", raw_composition)
            return None
        if weight < 0:
            log.add(row_id, line, "negative_weight", EXCLUSION, f"{holding}: {weight}", raw_composition)
            return None
        if holding in composition:
            log.add(row_id, line, "duplicate_ticker", WARNING, f"Duplicate ticker: {holding}", raw_composition)
        composition[holding] = composition.get(holding, 0.0) + weight

    coverage = sum(composition.values())
    if not composition:
        log.add(row_id, line, "empty_composition", WARNING, extract=raw_composition)
    if coverage > settings["quality"]["max_weight_sum"]:
        log.add(row_id, line, "weight_sum_too_high", WARNING, f"Sum of weights: {coverage:.2f}", raw_composition)
    if not description:
        log.add(row_id, line, "empty_description", WARNING, extract=name)

    return {
        "row_id": row_id,
        "line": line,
        "ticker": ticker,
        "name": name,
        "description": description,
        "composition": composition,
        "categories": [category] if category else [],
        "coverage": coverage,
        "n_holdings": len(composition),
    }


def validate_rows(rows: list[dict], bad_lines: list[dict], settings: dict, require_categories: bool = True,
                  n_reference: int | None = None) -> tuple[pd.DataFrame, QualityLog]:
    """Applies every rule; returns (usable dataset, quality log).

    Rows whose record number is >= n_reference come from the validation journal: a ticker present on several
    reference rows excludes them all, while a validation replaces the reference row and the earlier validations
    of its ticker (rows are in journal order, so the latest wins).
    """
    log = QualityLog()
    for bad in bad_lines:
        log.add(bad.get("record"), bad.get("line"), "malformed_line", EXCLUSION, extract=bad.get("extract", ""))

    records = []
    for position, row in enumerate(rows):
        row_id = row.get("__record__", position)
        record = _validate_row(row, row_id, row.get("__line__"), settings, log, require_categories)
        if record is not None:
            records.append(record)

    if n_reference is None:
        n_reference = len(rows) + len(bad_lines)
    reference = [record for record in records if record["row_id"] < n_reference]
    counts = Counter(record["ticker"] for record in reference)
    for record in reference:
        if counts[record["ticker"]] > 1:
            log.add(record["row_id"], record["line"], "duplicate_etf_ticker", EXCLUSION, extract=record["ticker"])
    by_ticker = {record["ticker"]: record for record in reference if counts[record["ticker"]] == 1}
    for record in records[len(reference):]:
        by_ticker.pop(record["ticker"], None)  # re-inserted last: the dataset keeps the reference order first
        by_ticker[record["ticker"]] = record
    return pd.DataFrame(list(by_ticker.values()), columns=DATA_COLUMNS), log
