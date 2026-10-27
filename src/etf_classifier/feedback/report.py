"""Validation monitoring report: volume, acceptance rate, bias alert, model weaknesses.

An acceptance-without-change rate close to 100% over many validations may mean that proposals are
validated without being reviewed: the model would then reinforce its own mistakes.

Usage:
    python -m etf_classifier.feedback.report [--since 2026-10-01]
"""

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date

import pandas as pd

from etf_classifier.common import markdown_table
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.feedback.store import ACCEPTED, CANCELLATION, CORRECTED, FeedbackStore


@dataclass
class FeedbackReport:
    markdown: str
    stats: dict


def _rate(group: pd.DataFrame) -> tuple[int, float | None]:
    proposed = group[group["Decision"].isin([ACCEPTED, CORRECTED])]
    if proposed.empty:
        return 0, None
    return len(proposed), float((proposed["Decision"] == ACCEPTED).mean())


def build_feedback_report(settings: dict, since: date | None = None) -> FeedbackReport:
    cfg = settings["feedback"]
    store = FeedbackStore(settings)
    journal = store.read_journal() if store.journal_path.exists() else pd.DataFrame(columns=["Decision"])
    if since is not None and not journal.empty:
        journal = journal[journal["ValidatedAt"].str[:10] >= since.isoformat()]
    lines = ["# Validation monitoring", ""]
    empty_stats = {"n_validations": 0, "n_cancelled": 0, "acceptance_rate": None, "by_validator": {},
                   "added": {}, "removed": {}, "bias_alerts": [], "created_categories": []}
    if journal.empty:
        return FeedbackReport("\n".join(lines + ["No validation recorded over this period.", ""]), empty_stats)

    active = FeedbackStore._active(journal)
    n_cancelled = int((journal["Decision"] == CANCELLATION).sum())
    n_proposed, rate = _rate(active)
    by_validator = {}
    for validator, group in active.groupby("ValidatedBy"):
        n, r = _rate(group)
        by_validator[validator] = {"n_validations": len(group), "n_with_proposal": n, "acceptance_rate": r}

    added, removed = Counter(), Counter()
    for row in active[active["Decision"] == CORRECTED].itertuples():
        added[row.Category] += 1           # category chosen by the validator…
        removed[row.PredictedCategory] += 1  # …instead of the model's proposal

    threshold, minimum = float(cfg["bias_acceptance_threshold"]), int(cfg["bias_min_validations"])
    alerts = [v for v, s in sorted(by_validator.items())
              if s["acceptance_rate"] is not None and s["n_with_proposal"] >= minimum
              and s["acceptance_rate"] >= threshold]
    if rate is not None and n_proposed >= minimum and rate >= threshold:
        alerts.append("global")

    created = store.registry.created()
    if since is not None:
        created = created[created["created_at"].str[:10] >= since.isoformat()]
    stats = {"n_validations": len(active), "n_cancelled": n_cancelled, "acceptance_rate": rate,
             "by_validator": by_validator, "added": dict(added), "removed": dict(removed),
             "bias_alerts": alerts, "created_categories": sorted(created["category"])}

    def pct(value):
        return f"{value:.0%}" if value is not None else "–"

    lines += [f"Period: since {since or 'the start'}", "",
              markdown_table(["Indicator", "Value"], [
                  ["Active validations", len(active)],
                  ["Cancellations", n_cancelled],
                  ["Acceptance rate without change", f"{pct(rate)} (over {n_proposed} proposal(s))"]]), ""]
    for who in alerts:
        label = "all validators" if who == "global" else who
        lines.append(f"> ⚠️ Bias risk for {label}: at least {threshold:.0%} of the proposals accepted without "
                     f"change (≥ {minimum} validations). Check that the proposals are actually reviewed.")
    if alerts:
        lines.append("")
    by_day = active["ValidatedAt"].str[:10].value_counts().sort_index()
    lines += ["## By validator", "",
              markdown_table(["Validator", "Validations", "With proposal", "Acceptance rate"],
                             [[v, s["n_validations"], s["n_with_proposal"], pct(s["acceptance_rate"])]
                              for v, s in sorted(by_validator.items())]),
              "", "## By day", "", markdown_table(["Day", "Validations"], [[d, n] for d, n in by_day.items()]),
              "", "## Validator corrections (model weaknesses)", "",
              markdown_table(["Category", "Added by validators", "Removed by validators"],
                             [[c, added.get(c, 0), removed.get(c, 0)] for c in sorted(set(added) | set(removed))])]
    lines += ["", "## Created categories", "", ", ".join(stats["created_categories"]) or "None.", ""]
    return FeedbackReport("\n".join(lines), stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation monitoring report")
    parser.add_argument("--since", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    report = build_feedback_report(settings, args.since)
    path = resolve_path(settings["model"]["reports_dir"]) / "feedback.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
