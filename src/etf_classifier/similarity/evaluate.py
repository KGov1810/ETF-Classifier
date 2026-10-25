"""Evaluation of the neighbour vote against the model and the trivial baseline, on the same split.

The evaluation index is built on the training set only: no test ETF can vote.

Usage:
    python -m etf_classifier.similarity.evaluate --csv data/raw/etfs.csv
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from etf_classifier.common import markdown_table
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import PreparedDataset, prepare_dataset
from etf_classifier.features.embeddings import CommandStopped, embedder_for_command
from etf_classifier.features.pipeline import training_fingerprint
from etf_classifier.model.baseline import FrequencyBaseline
from etf_classifier.model.classifier import decide, tune
from etf_classifier.model.evaluate import Metrics, evaluate
from etf_classifier.similarity.index import SimilarityIndex
from etf_classifier.similarity.vote import vote_matrix

EXIT_OK = 0  # 1 = cancelled (see features.embeddings)
VOTE_THRESHOLD = 0.5  # a category is selected when the neighbours holding it weigh at least half of the vote


@dataclass
class EvaluationOutcome:
    rows: list[tuple[str, Metrics]]
    index_row_ids: list[int]
    test_fingerprint: str


def run_similarity_evaluation(prepared: PreparedDataset, settings: dict, embedder=None) -> EvaluationOutcome:
    train, test = prepared.split.train, prepared.split.test
    tuned = tune(train, settings, embedder)
    pipeline, classes = tuned.pipeline, tuned.model.classes
    y_test = pipeline.transform_labels(test)
    max_categories = settings["model"]["max_categories"]

    rows = [("Trivial baseline",
             evaluate(y_test, FrequencyBaseline().fit(pipeline.transform_labels(train)).predict(len(y_test)), classes))]
    model_pred = tuned.model.decide(tuned.model.predict_proba(pipeline.transform(test).matrix), max_categories)
    rows.append(("Model (logistic regression)", evaluate(y_test, model_pred, classes)))

    index = SimilarityIndex.build(train, pipeline, settings)
    for k in settings["similarity"]["eval_k"]:
        scores = vote_matrix(index.query(test, pipeline, k=k), classes)
        votes = decide(scores, np.full(len(classes), VOTE_THRESHOLD), max_categories)
        rows.append((f"Neighbour vote (k = {k})", evaluate(y_test, votes, classes)))
    return EvaluationOutcome(rows, [int(r) for r in index.row_ids], training_fingerprint(test))


def render(outcome: EvaluationOutcome) -> str:
    lines = ["# Neighbour vote evaluation", "",
             f"Same test set for every method (fingerprint `{outcome.test_fingerprint}`). "
             "The index only contains training ETFs.", "",
             markdown_table(["Method", "Macro F1", "Micro F1", "Per-ETF F1"],
                            [[label, f"{m.f1_macro:.3f}", f"{m.f1_micro:.3f}", f"{m.f1_samples:.3f}"]
                             for label, m in outcome.rows])]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Neighbour vote evaluation")
    parser.add_argument("--csv", required=True, help="Path to the categorised ETF CSV")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending on Azure")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings(args.config)
    prepared = prepare_dataset(args.csv, settings)
    try:
        embedder = embedder_for_command(settings, prepared.data, args.yes, input_fn, client,
                                        enabled=settings["features"]["blocks"]["embeddings"]["enabled"])
    except CommandStopped as stop:
        return stop.exit_code

    report = render(run_similarity_evaluation(prepared, settings, embedder))
    path = resolve_path(settings["model"]["reports_dir"]) / "similarity_evaluation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
