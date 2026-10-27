"""Data quality report (command line + Markdown file).

Usage:
    python -m etf_classifier.data.report --csv data/raw/etfs.csv
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from etf_classifier.common import markdown_table
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import MissingColumnsError, prepare_dataset
from etf_classifier.data.normalize import find_similar_tickers, normalize_ticker
from etf_classifier.data.validation import RULE_LABELS

MAX_DETAIL_ROWS = 50
EXIT_OK, EXIT_FILE_NOT_FOUND, EXIT_MISSING_COLUMNS = 0, 2, 3


@dataclass
class Report:
    markdown: str
    stats: dict


def _issues_section(title: str, issues: list, empty_text: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not issues:
        return lines + [empty_text, ""]
    counts = Counter(issue.rule for issue in issues)
    lines += [markdown_table(["Rule", "Count"], [[RULE_LABELS[r], n] for r, n in counts.most_common()]), ""]

    def where(issue) -> str:
        line = issue.line if issue.line is not None else "?"
        return f"journal {line}" if issue.source == "validation" else str(line)

    detail = [[where(i), RULE_LABELS[i.rule], i.message, i.extract]
              for i in sorted(issues, key=lambda i: (i.line or 0))[:MAX_DETAIL_ROWS]]
    lines += ["Details (line in the file, header = line 1):", "",
              markdown_table(["Line", "Rule", "Message", "Extract"], detail), ""]
    if len(issues) > MAX_DETAIL_ROWS:
        lines += [f"… and {len(issues) - MAX_DETAIL_ROWS} more.", ""]
    return lines


def build_report(csv_path: str | Path, settings: dict) -> Report:
    """Loads the CSV, computes the statistics and produces the Markdown report."""
    prepared = prepare_dataset(csv_path, settings)
    data, log, dedup, split = prepared.data, prepared.log, prepared.dedup, prepared.split

    category_counts = pd.Series([c for cats in data["categories"] for c in cats], dtype=object).value_counts()
    raw_tickers = Counter(t for comp in data["composition_raw"] for t in comp)
    tickers = Counter(t for comp in data["composition"] for t in comp)
    similar = find_similar_tickers(normalize_ticker(t) for t in raw_tickers)
    group_sizes = data.groupby("group_id").size().sort_values(ascending=False)

    stats = {
        "total_rows": prepared.total_rows,
        "valid_rows": len(data),
        "excluded_rows": len({(i.row_id, i.line) for i in log.exclusions()}),
        "rows_with_warnings": len({i.row_id for i in log.warnings()}),
        "n_categories": int(category_counts.size),
        "n_raw_tickers": len(raw_tickers),
        "n_tickers": len(tickers),
        "n_groups": dedup.n_groups,
        "largest_group_size": dedup.largest_group_size,
    }

    lines = [
        "# Data quality report",
        "",
        f"File: `{Path(csv_path).name}` · generated on {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Summary",
        "",
        markdown_table(["Indicator", "Value"], [
            ["Rows read", stats["total_rows"]],
            ["Usable rows", stats["valid_rows"]],
            ["Excluded rows", stats["excluded_rows"]],
            ["Rows with warnings", stats["rows_with_warnings"]],
        ]),
        "",
    ]
    lines += _issues_section("Exclusions", log.exclusions(), "No excluded row.")
    lines += _issues_section("Warnings", log.warnings(), "No warning.")

    lines += ["## Categories", "",
              f"{stats['n_categories']} distinct categories (one per ETF).", ""]
    if category_counts.size:
        lines += [markdown_table(["Category", "Number of ETFs"], [[c, int(n)] for c, n in category_counts.items()]), ""]

    coverage = data["coverage"] if len(data) else pd.Series([0.0])
    lines += ["## Composition coverage", "",
              markdown_table(["Indicator", "Value"], [
                  ["Minimum coverage (sum of weights)", f"{coverage.min():.1f}"],
                  ["Median coverage", f"{coverage.median():.1f}"],
                  ["Maximum coverage", f"{coverage.max():.1f}"],
                  ["Empty compositions", int((data["n_holdings"] == 0).sum()) if len(data) else 0],
                  ["Median number of holdings", f"{data['n_holdings'].median():.0f}" if len(data) else "0"],
              ]), ""]

    top_n = settings["report"]["top_holdings"]
    lines += ["## Most frequent tickers", "",
              f"{stats['n_raw_tickers']} distinct tickers before normalisation, "
              f"{stats['n_tickers']} after.", ""]
    if tickers:
        lines += [markdown_table(["Ticker", "Number of ETFs"], [[t, n] for t, n in tickers.most_common(top_n)]), ""]

    lines += ["## Similar tickers", "",
              "Same root, different suffix: check them, and add them to "
              "`config/holding_aliases.csv` if they are the same security.", ""]
    if similar:
        rows = [[root, ", ".join(variants)] for root, variants in list(similar.items())[:MAX_DETAIL_ROWS]]
        lines += [markdown_table(["Root", "Variants"], rows), ""]
    else:
        lines += ["None.", ""]

    lines += ["## Near-duplicates", "",
              markdown_table(["Indicator", "Value"], [
                  ["Groups", dedup.n_groups],
                  ["Links found", dedup.n_links],
                  ["Size of the largest group", dedup.largest_group_size],
              ]), ""]
    lines += [f"> ⚠️ {w}" for w in dedup.warnings] + ([""] if dedup.warnings else [])
    multi = group_sizes[group_sizes > 1].head(10)
    if len(multi):
        rows = [[g, int(n), ", ".join(data.loc[data["group_id"] == g, "name"].head(8))] for g, n in multi.items()]
        lines += ["Largest groups:", "", markdown_table(["Group", "Size", "ETFs (first 8)"], rows), ""]

    summary = split.summary
    lines += ["## Train/test split", "",
              markdown_table(["Indicator", "Value"], [
                  ["ETFs in training", summary.n_train],
                  ["ETFs in test", summary.n_test],
                  ["Test share", f"{summary.test_share:.1%}"],
                  ["Categories missing from test", ", ".join(summary.missing_in_test) or "none"],
                  ["Categories missing from training", ", ".join(summary.missing_in_train) or "none"],
              ]), "",
              f"Rare categories (fewer than {settings['split']['rare_category_min']} ETFs):", ""]
    if summary.rare_categories:
        rows = [[c, n] for c, n in summary.rare_categories.items()]
        lines += [markdown_table(["Category", "Number of ETFs"], rows), ""]
    else:
        lines += ["None.", ""]

    return Report(markdown="\n".join(lines), stats=stats)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns an exit code."""
    parser = argparse.ArgumentParser(description="Quality report of the ETF CSV")
    parser.add_argument("--csv", required=True, help="Path to the ETF CSV")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--output", help="Output Markdown file (default: reports/data_quality.md)")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        return EXIT_FILE_NOT_FOUND
    try:
        report = build_report(csv_path, settings)
    except MissingColumnsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_MISSING_COLUMNS

    output = Path(args.output) if args.output else resolve_path(settings["report"]["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"\nReport written to {output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
