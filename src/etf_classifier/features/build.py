"""Feature building: cost estimate, confirmation, fitting and saving of the pipeline.

Usage:
    python -m etf_classifier.features.build --csv data/raw/etfs.csv --estimate-only
    python -m etf_classifier.features.build --csv data/raw/etfs.csv          # asks for confirmation
    python -m etf_classifier.features.build --csv data/raw/etfs.csv --yes    # no confirmation (daily job)
"""

import argparse
import logging
import sys
from typing import Callable

from etf_classifier.config import load_settings
from etf_classifier.data.loader import prepare_dataset
from etf_classifier.features.embeddings import EmbeddingPlan, EmbeddingsBlock, create_embedder
from etf_classifier.features.pipeline import FeaturePipeline

EXIT_OK, EXIT_CANCELLED = 0, 1


def _density(matrix) -> float:
    cells = matrix.shape[0] * matrix.shape[1]
    return matrix.nnz / cells if cells else 0.0


def announce_embedding_plan(settings: dict, data, embedder) -> tuple[EmbeddingPlan, str]:
    """Shows what computing the embeddings would cost (cache deducted), without calling Azure."""
    texts = EmbeddingsBlock(None, embedder.dimensions,
                            settings["features"]["embeddings"]["include_top_tickers"]).texts(data)
    plan = embedder.plan(texts)
    cost = f"${plan.estimate.cost_usd:.4f}" if plan.estimate else "unknown (deployment missing from the price list)"
    approx = " (approximate)" if plan.estimate and plan.estimate.approximate else ""
    tokens = plan.estimate.input_tokens if plan.estimate else "?"
    print(f"Embeddings: {plan.n_to_compute} text(s) to compute, {plan.n_cached} already cached · "
          f"{tokens} tokens · estimated cost {cost}{approx}")
    return plan, cost


def confirm_spending(plan: EmbeddingPlan, cost: str, assume_yes: bool, input_fn: Callable[[str], str]) -> bool:
    """Asks for confirmation before spending on Azure (unless there is nothing to compute or assume_yes)."""
    if not plan.n_to_compute or assume_yes:
        return True
    answer = input_fn(f"Confirm the estimated spending of {cost}? [y/N] ").strip().lower()
    if answer in {"y", "yes"}:
        return True
    print("Cancelled: no Azure call made.")
    return False


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Build the ETF features")
    parser.add_argument("--csv", required=True, help="Path to the ETF CSV")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--estimate-only", action="store_true", help="Show the estimated cost without computing")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending")
    parser.add_argument("--no-azure", action="store_true", help="Build the local blocks only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings(args.config)
    prepared = prepare_dataset(args.csv, settings)
    embeddings_enabled = settings["features"]["blocks"]["embeddings"]["enabled"] and not args.no_azure
    embedder = create_embedder(settings, client=client) if embeddings_enabled else None
    print(f"{len(prepared.data)} usable ETFs ({len(prepared.split.train)} in training, "
          f"{len(prepared.split.test)} in test).")

    if embedder is None:
        print("Embeddings: block inactive (Azure not configured or disabled).")
    else:
        plan, cost = announce_embedding_plan(settings, prepared.data, embedder)
        if args.estimate_only:
            return EXIT_OK
        if not confirm_spending(plan, cost, args.yes, input_fn):
            return EXIT_CANCELLED
    if args.estimate_only:
        return EXIT_OK

    pipeline = FeaturePipeline(settings, embedder=embedder).fit(prepared.split.train)
    train_features = pipeline.transform(prepared.split.train)
    pipeline.transform(prepared.split.test)
    path = pipeline.save()

    print("\nBlocks built:")
    for name, matrix in train_features.blocks.items():
        print(f"  - {name}: {matrix.shape[1]} features, density {_density(matrix):.1%}")
    print(f"Categories: {len(pipeline.labels.classes)}")
    if embedder is not None:
        print(f"Actual cost of this run: ${embedder.run_cost_usd:.4f} ({embedder.n_api_calls} Azure call(s))")
    print(f"Pipeline saved to {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
