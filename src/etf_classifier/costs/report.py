"""Rapport des coûts Azure OpenAI à partir du journal des appels.

Usage :
    python -m etf_classifier.costs.report --since 2026-10-01
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.costs.pricing import PricingTable, load_pricing
from etf_classifier.costs.tracker import LOG_SEPARATOR


@dataclass
class CostReport:
    markdown: str
    stats: dict


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _money(value: float) -> str:
    return f"{value:.4f} $"


def _read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return pd.DataFrame()
    log = pd.read_csv(log_path, sep=LOG_SEPARATOR, encoding="utf-8", dtype={"note": str})
    log["day"] = log["timestamp_utc"].str[:10]
    return log


def build_cost_report(log_path: str | Path, pricing: PricingTable, since: date | None = None,
                      until: date | None = None, stale_after_days: int = 90,
                      today: date | None = None) -> CostReport:
    """Totaux en USD par jour, module, opération et déploiement, sur la période demandée."""
    log = _read_log(Path(log_path))
    if not log.empty:
        if since:
            log = log[log["day"] >= since.isoformat()]
        if until:
            log = log[log["day"] <= until.isoformat()]

    period = f"{since or 'début'} → {until or 'aujourd’hui'}"
    lines = ["# Coûts Azure OpenAI", "", f"Période : {period} · devise : {pricing.currency}", ""]
    age = pricing.age_days(today)
    grid_line = f"Grille tarifaire du {pricing.as_of:%d/%m/%Y} ({age} jours) : {pricing.source}"
    lines += [f"> ⚠️ {grid_line}. Elle a plus de {stale_after_days} jours : vérifier les prix."
              if age > stale_after_days else grid_line, ""]

    if log.empty:
        lines += ["Aucun appel Azure enregistré sur cette période.", ""]
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

    lines += ["## Total", "", _table(["Indicateur", "Valeur"], [
        ["Coût total", _money(stats["total_usd"])],
        ["Appels", stats["n_calls"]],
        ["Tokens d'entrée", f"{stats['input_tokens']:,}".replace(",", " ")],
        ["Tokens de sortie", f"{stats['output_tokens']:,}".replace(",", " ")],
        ["Écart réel / estimation", f"{gap:+.1f} %" if gap is not None else "pas d'estimation"],
    ]), ""]
    if stats["n_unpriced"]:
        lines += [f"> ⚠️ {stats['n_unpriced']} appel(s) sans coût : déploiement absent de la grille tarifaire.", ""]
    for title, key, header in [("Par jour", "by_day", "Jour"), ("Par module", "by_module", "Module"),
                               ("Par opération", "by_operation", "Opération"),
                               ("Par déploiement", "by_deployment", "Déploiement")]:
        lines += [f"## {title}", "", _table([header, "Coût"], [[k, _money(v)] for k, v in stats[key].items()]), ""]
    return CostReport("\n".join(lines), stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rapport des coûts Azure OpenAI")
    parser.add_argument("--since", type=date.fromisoformat, help="Date de début (AAAA-MM-JJ)")
    parser.add_argument("--until", type=date.fromisoformat, help="Date de fin (AAAA-MM-JJ)")
    parser.add_argument("--log", help="Journal des appels (défaut : logs/azure_usage.csv)")
    parser.add_argument("--pricing", help="Grille tarifaire (défaut : config/pricing.yaml)")
    parser.add_argument("--output", help="Fichier Markdown (défaut : reports/azure_costs.md)")
    args = parser.parse_args(argv)

    cfg = load_settings()["costs"]
    pricing = load_pricing(args.pricing)
    log_path = Path(args.log) if args.log else resolve_path(cfg["usage_log_path"])
    report = build_cost_report(log_path, pricing, args.since, args.until, cfg["stale_after_days"])
    output = Path(args.output) if args.output else resolve_path(cfg["report_output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"\nRapport écrit dans {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
