"""Comparison of experiments (block combinations) on exactly the same split.

Usage:
    python -m etf_classifier.model.compare --csv data/raw/etfs.csv --experiments local,local+embeddings
"""

import argparse
import copy
import logging
import sys
from dataclasses import dataclass
from typing import Callable

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import PreparedDataset, prepare_dataset
from etf_classifier.features.build import announce_embedding_plan, confirm_spending
from etf_classifier.features.embeddings import create_embedder
from etf_classifier.features.pipeline import training_fingerprint
from etf_classifier.model.evaluate import Metrics
from etf_classifier.model.train import train_and_evaluate

EXIT_OK, EXIT_CANCELLED, EXIT_UNKNOWN = 0, 1, 2


@dataclass
class ExperimentResult:
    name: str
    test_fingerprint: str
    metrics: Metrics | None = None
    baseline: Metrics | None = None
    seconds: float = 0.0
    azure_cost_usd: float = 0.0
    n_features: int = 0
    skipped_reason: str | None = None


def _experiment_settings(settings: dict, blocks: dict[str, bool]) -> dict:
    exp = copy.deepcopy(settings)
    for name, enabled in blocks.items():
        exp["features"]["blocks"][name]["enabled"] = bool(enabled)
    return exp


def run_comparison(prepared: PreparedDataset, settings: dict, names: list[str], embedder=None) -> list[ExperimentResult]:
    """Trains and evaluates each experiment on the same training set and the same test set."""
    fingerprint = training_fingerprint(prepared.split.test)
    results = []
    for name in names:
        blocks = settings["model"]["experiments"][name]
        if blocks.get("embeddings") and embedder is None:
            results.append(ExperimentResult(name, fingerprint, skipped_reason="not run (Azure not configured)"))
            continue
        outcome = train_and_evaluate(prepared, _experiment_settings(settings, blocks),
                                     embedder if blocks.get("embeddings") else None)
        n_features = sum(b["n_features"] for b in outcome.result.pipeline.metadata["blocks"].values())
        results.append(ExperimentResult(name, training_fingerprint(prepared.split.test), outcome.metrics,
                                        outcome.baseline, outcome.seconds, outcome.azure_cost_usd, n_features))
    return results


def render_comparison(results: list[ExperimentResult]) -> str:
    rows = ["| Experiment | Macro F1 | Micro F1 | Features | Time | Azure cost |", "|---|---|---|---|---|---|"]
    baseline = next((r.baseline for r in results if r.baseline is not None), None)
    if baseline:
        rows.append(f"| Trivial baseline | {baseline.f1_macro:.3f} | {baseline.f1_micro:.3f} | 0 | – | $0 |")
    for r in results:
        if r.skipped_reason:
            rows.append(f"| {r.name} | {r.skipped_reason} | | | | |")
        else:
            rows.append(f"| {r.name} | {r.metrics.f1_macro:.3f} | {r.metrics.f1_micro:.3f} | {r.n_features} | "
                        f"{r.seconds:.1f} s | ${r.azure_cost_usd:.4f} |")
    fingerprint = results[0].test_fingerprint if results else "–"
    return "\n".join([
        "# Experiment comparison", "",
        f"All experiments are evaluated on the same test set (fingerprint `{fingerprint}`).",
        "Main metric: macro F1. The Azure cost is marginal: embeddings already cached are not paid for again.",
        "", *rows, "",
    ])


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Compare experiments on the same split")
    parser.add_argument("--csv", required=True, help="Path to the categorised ETF CSV")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--experiments", default="local,local+embeddings", help="Comma-separated list")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending on Azure")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings(args.config)
    available = settings["model"]["experiments"]
    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"Error: unknown experiment(s): {', '.join(unknown)}. "
              f"Available: {', '.join(available)}", file=sys.stderr)
        return EXIT_UNKNOWN

    prepared = prepare_dataset(args.csv, settings)
    embedder = None
    if any(available[n].get("embeddings") for n in names):
        embedder = create_embedder(settings, client=client)
        if embedder is not None:
            plan, cost = announce_embedding_plan(settings, prepared.data, embedder)
            if not confirm_spending(plan, cost, args.yes, input_fn):
                return EXIT_CANCELLED

    report = render_comparison(run_comparison(prepared, settings, names, embedder))
    path = resolve_path(settings["model"]["reports_dir"]) / "model_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
