"""Already-categorised ETFs closest to new ETFs, with explanations and a near-duplicate alert.

Usage:
    python -m etf_classifier.similarity.search --csv new_etfs.csv [--model current] [--k 5]
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from etf_classifier.common import markdown_table
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import load_dataset
from etf_classifier.features.embeddings import CommandStopped, embedder_for_command
from etf_classifier.model.registry import NoModelError, load_bundle

EXIT_OK, EXIT_NO_MODEL, EXIT_NO_INDEX = 0, 2, 4  # 1 = cancelled, 3 = Azure not configured


def _summary(labels: list[str], results, version: str) -> str:
    lines = [f"# Closest ETFs (model {version})", ""]
    for label, result in zip(labels, results):
        lines += [f"## {label}", ""]
        if result.near_duplicate:
            dup = result.near_duplicate
            lines += [f"> ⚠️ Near-duplicate of **{dup.ticker} · {dup.name}** ({dup.overlap_pct:.0f}% of the weight in "
                      f"common), category: {', '.join(dup.categories)}", ""]
        if result.text_only:
            lines += ["No composition: matched on the text only.", ""]
        rows = [[rank, f"{n.ticker} · {n.name}", ", ".join(n.categories), f"{n.score:.2f}", f"{n.overlap_pct:.0f}%",
                 ", ".join(t for t, _ in n.shared_holdings) or "–"]
                for rank, n in enumerate(result.neighbors, start=1)]
        lines += [markdown_table(["Rank", "ETF", "Category", "Score", "Weight in common", "Shared tickers"], rows),
                  ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="ETFs closest to new ETFs")
    parser.add_argument("--csv", required=True, help="CSV of the new ETFs (Ticker;Nom;Description;Composition)")
    parser.add_argument("--model", default="current", help="Model version (default: current model)")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--k", type=int, help="Number of neighbours (default: settings.yaml)")
    parser.add_argument("--output", help="Output CSV (a .md summary is written next to it)")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending on Azure")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings(args.config)
    try:
        bundle = load_bundle(args.model, settings)
    except NoModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_NO_MODEL
    if bundle.similarity is None:
        print(f"Error: model {bundle.version} has no similarity index: retrain it "
              "with `python -m etf_classifier.model.train`.", file=sys.stderr)
        return EXIT_NO_INDEX

    data = load_dataset(args.csv, settings, require_categories=False).data.reset_index(drop=True)
    try:
        embedder = embedder_for_command(settings, data, args.yes, input_fn, client,
                                        enabled=bundle.pipeline.uses_embeddings, required=True)
    except CommandStopped as stop:
        return stop.exit_code
    if embedder is not None:
        bundle.pipeline.attach_embedder(embedder)

    k = args.k or settings["similarity"]["n_neighbors"]
    results = bundle.similarity.query(data, bundle.pipeline, k=k)
    rows = []
    for ticker, name, result in zip(data["ticker"], data["name"], results):
        duplicate = result.near_duplicate.ticker if result.near_duplicate else ""
        for rank, n in enumerate(result.neighbors, start=1):
            rows.append({
                "Ticker": ticker, "Name": name, "Rank": rank, "NeighborTicker": n.ticker, "NeighborName": n.name,
                "NeighborCategory": ", ".join(n.categories),
                "Score": round(n.score, 4), "CompositionSimilarity": round(n.composition_similarity, 4),
                "TextSimilarity": round(n.text_similarity, 4), "OverlapPct": round(n.overlap_pct, 1),
                "SharedHoldings": str([t for t, _ in n.shared_holdings]), "NearDuplicate": duplicate,
            })
    if args.output:
        output = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = resolve_path(settings["model"]["reports_dir"]) / f"neighbors_{bundle.version}_{stamp}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep=";", index=False, encoding="utf-8")
    labels = [f"{t} · {n}" for t, n in zip(data["ticker"], data["name"])]
    output.with_suffix(".md").write_text(_summary(labels, results, bundle.version), encoding="utf-8")

    n_duplicates = sum(1 for r in results if r.near_duplicate)
    print(f"{len(data)} ETFs processed · {k} neighbours each · {n_duplicates} near-duplicate(s) flagged")
    print(f"Neighbours: {output}\nSummary: {output.with_suffix('.md')}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
