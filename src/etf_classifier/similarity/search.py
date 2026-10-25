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

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import load_dataset
from etf_classifier.features.build import announce_embedding_plan, confirm_spending
from etf_classifier.features.embeddings import create_embedder
from etf_classifier.model.registry import NoModelError, load_bundle

EXIT_OK, EXIT_CANCELLED, EXIT_NO_MODEL, EXIT_NO_AZURE, EXIT_NO_INDEX = 0, 1, 2, 3, 4


def _summary(names: list[str], results, version: str) -> str:
    lines = [f"# Closest ETFs (model {version})", ""]
    for name, result in zip(names, results):
        lines += [f"## {name}", ""]
        if result.near_duplicate:
            dup = result.near_duplicate
            lines += [f"> ⚠️ Near-duplicate of **{dup.name}** ({dup.overlap_pct:.0f}% of the weight in common), "
                      f"categories: {', '.join(dup.categories)}", ""]
        if result.text_only:
            lines += ["No composition: matched on the text only.", ""]
        lines += ["| Rank | ETF | Categories | Score | Weight in common | Shared tickers |", "|---|---|---|---|---|---|"]
        for rank, n in enumerate(result.neighbors, start=1):
            shared = ", ".join(t for t, _ in n.shared_holdings) or "–"
            lines.append(f"| {rank} | {n.name} | {', '.join(n.categories)} | {n.score:.2f} | "
                         f"{n.overlap_pct:.0f}% | {shared} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="ETFs closest to new ETFs")
    parser.add_argument("--csv", required=True, help="CSV of the new ETFs (Name;Description;Composition)")
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
    if bundle.pipeline.uses_embeddings:
        embedder = create_embedder(settings, client=client)
        if embedder is None:
            print("Error: this model uses Azure embeddings, but Azure OpenAI is not configured "
                  "(see .env.example).", file=sys.stderr)
            return EXIT_NO_AZURE
        bundle.pipeline.attach_embedder(embedder)
        plan, cost = announce_embedding_plan(settings, data, embedder)
        if not confirm_spending(plan, cost, args.yes, input_fn):
            return EXIT_CANCELLED

    k = args.k or settings["similarity"]["n_neighbors"]
    results = bundle.similarity.query(data, bundle.pipeline, k=k)
    rows = []
    for name, result in zip(data["name"], results):
        duplicate = result.near_duplicate.name if result.near_duplicate else ""
        for rank, n in enumerate(result.neighbors, start=1):
            rows.append({
                "Name": name, "Rank": rank, "NeighborName": n.name, "NeighborCategories": str(n.categories),
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
    output.with_suffix(".md").write_text(_summary(list(data["name"]), results, bundle.version), encoding="utf-8")

    n_duplicates = sum(1 for r in results if r.near_duplicate)
    print(f"{len(data)} ETFs processed · {k} neighbours each · {n_duplicates} near-duplicate(s) flagged")
    print(f"Neighbours: {output}\nSummary: {output.with_suffix('.md')}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
