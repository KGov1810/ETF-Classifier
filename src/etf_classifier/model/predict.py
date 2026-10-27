"""Category proposals for new ETFs, and check of the classifications already established.

Input: the input CSV format (Ticker;Nom;Description;Composition), Classification optional. When a Classification
is given, the model checks it: agreement, disagreement or unknown category, with the model's score for it.
The output keeps the input columns, so that once Classification is completed or corrected it can be imported as
validations (feedback.importer).

Usage:
    python -m etf_classifier.model.predict --csv new_etfs.csv [--model current] [--output …]
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
from etf_classifier.data.parsing import format_composition
from etf_classifier.features.embeddings import CommandStopped, embedder_for_command
from etf_classifier.model.check import AGREEMENT, DISAGREEMENT, UNKNOWN, check_established
from etf_classifier.model.registry import NoModelError, load_bundle

EXIT_OK, EXIT_NO_MODEL = 0, 2  # 1 = cancelled, 3 = Azure not configured (see features.embeddings)
N_SCORES = 3  # best categories listed with their score


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Category proposals for new ETFs")
    parser.add_argument("--csv", required=True,
                        help="CSV of the ETFs (Ticker;Nom;Description;Composition, Classification optional)")
    parser.add_argument("--model", default="current", help="Model version (default: current model)")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--output", help="Output CSV (default: reports/predictions_<version>_<date>.csv)")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending on Azure")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings(args.config)
    try:
        bundle = load_bundle(args.model, settings)
    except NoModelError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_NO_MODEL

    loaded = load_dataset(args.csv, settings, require_categories=False)
    data = loaded.data.reset_index(drop=True)
    try:
        embedder = embedder_for_command(settings, data, args.yes, input_fn, client,
                                        enabled=bundle.pipeline.uses_embeddings, required=True)
    except CommandStopped as stop:
        return stop.exit_code
    if embedder is not None:
        bundle.pipeline.attach_embedder(embedder)

    cfg = settings["model"]
    predictions = bundle.model.predict(bundle.pipeline.transform(data).matrix, cfg["uncertain_threshold"])
    checks = [check_established(p, cats[0] if cats else "", bundle.model.classes)
              for p, cats in zip(predictions, data["categories"])]
    cols = settings["csv"]["columns"]
    # Self-contained file: the team completes or corrects Classification, then imports it (feedback.importer)
    output = pd.DataFrame({
        cols["ticker"]: data["ticker"],
        cols["name"]: data["name"],
        cols["description"]: data["description"],
        cols["composition"]: [format_composition(comp) for comp in data["composition_raw"]],
        cols["category"]: [c.established for c in checks],
        "PredictedCategory": [p.categories[0] for p in predictions],
        "Scores": [str([{c: round(s, 4)} for c, s in p.scores[:N_SCORES]]) for p in predictions],
        "Uncertain": [p.uncertain for p in predictions],
        "EstablishedScore": [f"{c.established_score:.4f}" if c.established_score is not None else "" for c in checks],
        "Check": [c.verdict or "" for c in checks],
        "ModelVersion": bundle.version,
    })
    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = resolve_path(cfg["reports_dir"]) / f"predictions_{bundle.version}_{stamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, sep=";", index=False, encoding="utf-8")

    n_uncertain = int(output["Uncertain"].sum())
    print(f"{len(output)} ETFs processed with model {bundle.version} · {n_uncertain} uncertain · "
          f"{len(loaded.log.exclusions())} excluded row(s)")
    verdicts = [c.verdict for c in checks]
    if any(verdicts):
        print(f"Established classifications: {verdicts.count(AGREEMENT)} agreement · "
              f"{verdicts.count(DISAGREEMENT)} disagreement · {verdicts.count(UNKNOWN)} unknown")
    print(f"Proposals: {output_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
