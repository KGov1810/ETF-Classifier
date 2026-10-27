"""Import of a batch of validations made outside the interface.

Input format: the input CSV format (`Ticker;Nom;Description;Classification;Composition`), where Classification is
the validated category; the output of `model.predict`, completed or corrected by the team, fits. Optional columns:
PredictedCategory (to record whether the proposal was accepted) and ModelVersion.
Every line goes through the same checks as a single validation.

Usage:
    python -m etf_classifier.feedback.importer --csv validated.csv --by "First Last" [--create-categories]
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from etf_classifier.common import markdown_table
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.feedback.checks import ValidationError
from etf_classifier.feedback.store import FeedbackError, FeedbackStore

EXIT_OK, EXIT_BAD_FILE, EXIT_STORE = 0, 2, 3


@dataclass
class ImportResult:
    accepted: list[tuple[int, str, str]] = field(default_factory=list)         # (line, ETF, decision)
    refused: list[tuple[int, str, list[str]]] = field(default_factory=list)    # (line, ETF, reasons)
    created: list[str] = field(default_factory=list)                           # categories created
    error: str = ""  # journal unavailable: the import stopped (lines already recorded stay recorded)


def read_rows(path: Path, columns: dict) -> list[tuple[int, dict]]:
    """(line number, row) of the validated CSV; raises ValueError if a required column is missing."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        missing = [c for c in columns.values() if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        return [(reader.line_num, row) for row in reader]


def import_rows(rows: list[tuple[int, dict]], store: FeedbackStore, validated_by: str, columns: dict,
                create_categories: bool = False) -> ImportResult:
    """Records every valid line with the same checks as a single validation; refused lines keep their reason.

    columns: header names of the input CSV (settings csv.columns).
    """
    result = ImportResult()
    created_before = set(store.registry.created()["category"])
    for line, row in rows:
        etf = {key: row.get(columns[key], "") for key in ("ticker", "name", "description", "composition")}
        label = (etf["ticker"] or "").strip() or (etf["name"] or "").strip()
        category = (row.get(columns["category"]) or "").strip()
        if not category:
            result.refused.append((line, label, [f"validated category missing (empty {columns['category']} column)"]))
            continue
        try:
            record = store.record_validation(
                etf, category, validated_by=validated_by,
                predicted_category=(row.get("PredictedCategory") or "").strip() or None,
                model_version=(row.get("ModelVersion") or "").strip() or None, create_category=create_categories,
            )
            result.accepted.append((line, record["Ticker"], record["Decision"]))
        except ValidationError as exc:
            result.refused.append((line, label, exc.errors))
        except FeedbackError as exc:
            result.error = str(exc)
            break
    result.created = sorted(set(store.registry.created()["category"]) - created_before)
    return result


def render_import_report(file_name: str, result: ImportResult) -> str:
    lines = [f"# Validation import: {file_name}", "",
             f"{len(result.accepted)} validation(s) recorded · {len(result.refused)} line(s) refused · "
             f"{len(result.created)} category(ies) created", ""]
    if result.created:
        lines += [f"Created categories: {', '.join(result.created)}", ""]
    if result.refused:
        rows = [[line, name, "; ".join(errors)] for line, name, errors in result.refused]
        lines += ["## Refused lines", "", markdown_table(["Line", "ETF", "Reason"], rows), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a batch of validations")
    parser.add_argument("--csv", required=True, help="CSV (input format; Classification = validated category)")
    parser.add_argument("--by", required=True, help="Name or e-mail of the validator")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--create-categories", action="store_true", help="Allow categories to be created")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    path = Path(args.csv)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return EXIT_BAD_FILE
    try:
        rows = read_rows(path, settings["csv"]["columns"])
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_BAD_FILE

    columns = settings["csv"]["columns"]
    result = import_rows(rows, FeedbackStore(settings), args.by, columns, args.create_categories)
    if result.error:
        print(f"Error: {result.error} ({len(result.accepted)} line(s) already recorded)", file=sys.stderr)
        return EXIT_STORE
    report = render_import_report(path.name, result)
    target = resolve_path(settings["model"]["reports_dir"]) / f"feedback_import_{datetime.now():%Y%m%d-%H%M%S}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"Summary: {target}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
