"""Azure OpenAI cost report built from the call log.

Usage:
    python -m etf_classifier.costs.report --since 2026-10-01
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from etf_classifier.common import markdown_table
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.costs.pricing import PricingTable, load_pricing
from etf_classifier.costs.tracker import LOG_SEPARATOR


@dataclass
class CostReport:
    markdown: str
    stats: dict


def _money(value: float) -> str:
    return f"${value:.4f}"


def _read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame()
    log = pd.read_csv(log_path, sep=LOG_SEPARATOR, encoding="utf-8", dtype={"note": str})
    log["day"] = log["timestamp_utc"].str[:10]
    return log


def build_cost_report(log_path: str | Path, pricing: PricingTable, since: date | None = None,
                      until: date | None = None, stale_after_days: int = 90,
                      today: date | None = None) -> CostReport:
    """Totals in USD per day, module, operation and deployment, over the requested period."""
    log = _read_log(Path(log_path))
    if not log.empty:
        if since:
            log = log[log["day"] >= since.isoformat()]
        if until:
            log = log[log["day"] <= until.isoformat()]

    period = f"{since or 'start'} → {until or 'today'}"
    lines = ["# Azure OpenAI costs", "", f"Period: {period} · currency: {pricing.currency}", ""]
    age = pricing.age_days(today)
    grid_line = f"Price list dated {pricing.as_of:%Y-%m-%d} ({age} days old): {pricing.source}"
    lines += [f"> ⚠️ {grid_line}. It is more than {stale_after_days} days old: check the prices."
              if age > stale_after_days else grid_line, ""]

    if log.empty:
        lines += ["No Azure call recorded over this period.", ""]
        return CostReport("\n".join(lines), {"n_calls": 0, "total_usd": 0.0, "by_day": {}, "by_module": {},
                                             "estimate_gap_pct": None})

    priced = log.dropna(subset=["cost_usd"])
    by = {key: priced.groupby(column)["cost_usd"].sum().sort_index().to_dict()
          for key, column in [("by_day", "day"), ("by_module", "module"),
                              ("by_operation", "operation"), ("by_deployment", "deployment")]}
    compared = log.dropna(subset=["cost_usd", "estimated_cost_usd"])
    gap = None
    if not compared.empty and compared["estimated_cost_usd"].sum() > 0:
        estimated = compared["estimated_cost_usd"].sum()
        gap = (compared["cost_usd"].sum() - estimated) / estimated * 100
    stats = {
        "n_calls": len(log),
        "n_unpriced": int(log["cost_usd"].isna().sum()),
        "total_usd": float(priced["cost_usd"].sum()),
        "input_tokens": int(log["input_tokens"].sum()),
        "output_tokens": int(log["output_tokens"].sum()),
        "estimate_gap_pct": gap,
        **by,
    }

    lines += ["## Total", "", markdown_table(["Indicator", "Value"], [
        ["Total cost", _money(stats["total_usd"])],
        ["Calls", stats["n_calls"]],
        ["Input tokens", f"{stats['input_tokens']:,}".replace(",", " ")],
        ["Output tokens", f"{stats['output_tokens']:,}".replace(",", " ")],
        ["Actual vs estimate gap", f"{gap:+.1f}%" if gap is not None else "no estimate"],
    ]), ""]
    if stats["n_unpriced"]:
        lines += [f"> ⚠️ {stats['n_unpriced']} call(s) without a cost: deployment missing from the price list.", ""]
    for title, key, header in [("By day", "by_day", "Day"), ("By module", "by_module", "Module"),
                               ("By operation", "by_operation", "Operation"),
                               ("By deployment", "by_deployment", "Deployment")]:
        rows = [[k, _money(v)] for k, v in stats[key].items()]
        lines += [f"## {title}", "", markdown_table([header, "Cost"], rows), ""]
    return CostReport("\n".join(lines), stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Azure OpenAI cost report")
    parser.add_argument("--since", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    parser.add_argument("--log", help="Call log (default: logs/azure_usage.csv)")
    parser.add_argument("--pricing", help="Price list (default: config/pricing.yaml)")
    parser.add_argument("--output", help="Markdown file (default: reports/azure_costs.md)")
    args = parser.parse_args(argv)

    cfg = load_settings()["costs"]
    pricing = load_pricing(args.pricing)
    log_path = Path(args.log) if args.log else resolve_path(cfg["usage_log_path"])
    report = build_cost_report(log_path, pricing, args.since, args.until, cfg["stale_after_days"])
    output = Path(args.output) if args.output else resolve_path(cfg["report_output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"\nReport written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
