"""Import of a batch of validations made outside the interface.

Input format: the output of `model.predict`, completed by the team with a Category column
(validated categories). Every line goes through the same checks as a single validation.

Usage:
    python -m etf_classifier.feedback.importer --csv validated.csv --by "First Last" [--create-categories]
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.parsing import ParsingError, parse_categories
from etf_classifier.feedback.checks import ValidationError
from etf_classifier.feedback.store import FeedbackError, FeedbackStore

REQUIRED_COLUMNS = ["Name", "Description", "Composition", "Category"]
EXIT_OK, EXIT_BAD_FILE, EXIT_STORE = 0, 2, 3


def _created_categories(store: FeedbackStore) -> set[str]:
    """Categories created by the validators (excluding historical ones)."""
    return set(store.registry.created()["category"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import a batch of validations")
    parser.add_argument("--csv", required=True, help="Validated CSV (output of predict + Category column)")
    parser.add_argument("--by", required=True, help="Name or e-mail of the validator")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--create-categories", action="store_true", help="Allow categories to be created")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    path = Path(args.csv)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return EXIT_BAD_FILE
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"Error: missing columns: {', '.join(missing)}", file=sys.stderr)
            return EXIT_BAD_FILE
        rows = [(reader.line_num, row) for row in reader]

    store = FeedbackStore(settings)
    categories_before = _created_categories(store)
    accepted, refused = [], []
    for line, row in rows:
        name = (row.get("Name") or "").strip()
        try:
            categories = parse_categories(row.get("Category") or "")
            predicted_raw = (row.get("PredictedCategories") or "").strip()
            predicted = parse_categories(predicted_raw) if predicted_raw else None
        except ParsingError as exc:
            refused.append((line, name, [str(exc)]))
            continue
        if not categories:
            refused.append((line, name, ["validated category missing (empty Category column)"]))
            continue
        try:
            record = store.record_validation(
                {"name": name, "description": row.get("Description", ""), "composition": row.get("Composition", "")},
                categories, validated_by=args.by, predicted_categories=predicted,
                model_version=(row.get("ModelVersion") or "").strip() or None,
                create_categories=args.create_categories,
            )
            accepted.append((line, name, record["Decision"]))
        except ValidationError as exc:
            refused.append((line, name, exc.errors))
        except FeedbackError as exc:
            print(f"Error: {exc} ({len(accepted)} line(s) already recorded)", file=sys.stderr)
            return EXIT_STORE
    created = sorted(_created_categories(store) - categories_before)

    lines = [f"# Validation import: {path.name}", "",
             f"{len(accepted)} validation(s) recorded · {len(refused)} line(s) refused · "
             f"{len(created)} category(ies) created", ""]
    if created:
        lines += [f"Created categories: {', '.join(created)}", ""]
    if refused:
        lines += ["## Refused lines", "", "| Line | ETF | Reason |", "|---|---|---|"]
        lines += [f"| {line} | {name} | {'; '.join(errors)} |" for line, name, errors in refused]
        lines.append("")
    report = "\n".join(lines)
    target = resolve_path(settings["model"]["reports_dir"]) / f"feedback_import_{datetime.now():%Y%m%d-%H%M%S}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report)
    print(f"Summary: {target}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
