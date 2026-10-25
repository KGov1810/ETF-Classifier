"""Full training: tuning, single evaluation on the test set, report, saving, promotion.

Usage:
    python -m etf_classifier.model.train --csv data/raw/etfs.csv --no-azure        # local baseline
    python -m etf_classifier.model.train --csv data/raw/etfs.csv --yes --promote   # with embeddings
"""

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import PreparedDataset, prepare_dataset
from etf_classifier.features.embeddings import CommandStopped, embedder_for_command
from etf_classifier.features.pipeline import dataset_fingerprint, training_fingerprint
from etf_classifier.model.baseline import FrequencyBaseline
from etf_classifier.model.classifier import TuningResult, tune
from etf_classifier.model.evaluate import Metrics, evaluate, render_report
from etf_classifier.model.explain import top_features
from etf_classifier.model.registry import TRAIN_NAMES_FILE, current_version, promote, save_bundle
from etf_classifier.similarity.index import SimilarityIndex

EXIT_OK = 0  # 1 = cancelled (see features.embeddings)


@dataclass
class TrainingOutcome:
    result: TuningResult
    metrics: Metrics
    baseline: Metrics
    top: dict
    unknown_categories: dict[str, int]
    seconds: float
    azure_cost_usd: float


def train_and_evaluate(prepared: PreparedDataset, settings: dict, embedder=None) -> TrainingOutcome:
    """Tunes the model on the training set only, then evaluates it once on the test set."""
    start = time.perf_counter()
    cost_before = embedder.run_cost_usd if embedder else 0.0
    result = tune(prepared.split.train, settings, embedder)
    cfg = settings["model"]
    test_features = result.pipeline.transform(prepared.split.test)
    y_test = result.pipeline.transform_labels(prepared.split.test)
    unknown = dict(result.pipeline.labels.last_unknown)
    y_pred = result.model.decide(result.model.predict_proba(test_features.matrix), cfg["max_categories"])
    classes = result.model.classes
    baseline = FrequencyBaseline().fit(result.pipeline.transform_labels(prepared.split.train))
    return TrainingOutcome(
        result=result,
        metrics=evaluate(y_test, y_pred, classes),
        baseline=evaluate(y_test, baseline.predict(len(y_test)), classes),
        top=top_features(result.model, test_features.feature_names, k=10),
        unknown_categories=unknown,
        seconds=time.perf_counter() - start,
        azure_cost_usd=(embedder.run_cost_usd if embedder else 0.0) - cost_before,
    )


def _model_card(version: str, outcome: TrainingOutcome, info: dict) -> str:
    m, b = outcome.metrics, outcome.baseline
    lines = [f"# Model card {version}", "",
             "Multi-label ETF classifier: one logistic regression per category, thresholds tuned per category.", ""]
    lines += [f"- **{k}**: {v}" for k, v in info.items()]
    lines += ["", "## Performance (test set)", "",
              f"- Macro F1: {m.f1_macro:.3f} (trivial baseline: {b.f1_macro:.3f})",
              f"- Micro F1: {m.f1_micro:.3f} (trivial baseline: {b.f1_micro:.3f})", "",
              "## Limitations", "",
              "- Cannot predict a category that is missing from the training set.",
              "- Rare categories use a default threshold when the validation set has too few examples.",
              "- Performance depends on the coverage of the compositions (often partial)."]
    return "\n".join(lines) + "\n"


@dataclass
class TrainedModel:
    version: str
    path: Path
    outcome: TrainingOutcome
    report_path: Path


def train_and_save(prepared: PreparedDataset, settings: dict, embedder=None, promote_model: bool = False,
                   make_current_if_first: bool = True) -> TrainedModel:
    """Trains, evaluates once, saves the model folder (model, pipeline, index, training names, card) and the report.

    Shared by the `model.train` command and the daily retraining. The very first model becomes current
    (unless make_current_if_first=False); later ones only with promote_model=True.
    """
    outcome = train_and_evaluate(prepared, settings, embedder)
    result = outcome.result
    first_model = current_version(settings) is None
    fingerprint = dataset_fingerprint(prepared.data)
    info = {
        "Training / test ETFs": f"{len(prepared.split.train)} / {len(prepared.split.test)}",
        "Training set fingerprint": training_fingerprint(prepared.split.train),
        "Dataset fingerprint": fingerprint,
        "Feature blocks": ", ".join(result.pipeline.active_blocks),
        "Selected C (validation)": result.best_c,
        "Validation macro F1 per C": ", ".join(f"{c} → {s:.3f}" for c, s in result.validation_scores.items()),
        "Learned categories": len(result.model.classes),
        "Training time": f"{outcome.seconds:.1f} s",
        "Azure cost of training": f"${outcome.azure_cost_usd:.4f}",
    }
    thresholds = dict(zip(result.model.classes, result.model.thresholds.tolist()))
    metrics = {**outcome.metrics.to_dict(), "baseline": outcome.baseline.to_dict(), "best_c": result.best_c,
               "validation_scores": result.validation_scores, "thresholds": thresholds,
               "blocks": result.pipeline.active_blocks, "unknown_categories": outcome.unknown_categories,
               "n_train": len(prepared.split.train), "n_test": len(prepared.split.test),
               "azure_cost_usd": outcome.azure_cost_usd, "dataset_fingerprint": fingerprint}
    # similarity index over all usable rows, stored with the model (always consistent)
    similarity = SimilarityIndex.build(prepared.data, result.pipeline, settings)
    info["Similarity index"] = f"{len(prepared.data)} ETFs, {similarity.size_mb:.1f} MB"
    path = save_bundle(result.model, result.pipeline, metrics, settings, "", similarity=similarity,
                       make_current_if_first=make_current_if_first)
    version = path.name
    # names of the training ETFs: the retraining compares models on ETFs the current model never saw
    (path / TRAIN_NAMES_FILE).write_text(json.dumps(sorted(prepared.split.train["name"]), ensure_ascii=False),
                                         encoding="utf-8")
    (path / "MODEL_CARD.md").write_text(_model_card(version, outcome, info), encoding="utf-8")
    report = render_report(version, outcome.metrics, outcome.baseline, outcome.top, thresholds,
                           outcome.unknown_categories, info)
    report_path = resolve_path(settings["model"]["reports_dir"]) / f"model_{version}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    if promote_model and not first_model:
        promote(version, settings)
    return TrainedModel(version, path, outcome, report_path)


def main(argv: list[str] | None = None, client=None, input_fn: Callable[[str], str] = input) -> int:
    parser = argparse.ArgumentParser(description="Train the multi-label classifier")
    parser.add_argument("--csv", required=True, help="Path to the categorised ETF CSV")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--no-azure", action="store_true", help="Local blocks only (no embeddings)")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before spending on Azure")
    parser.add_argument("--promote", action="store_true", help="Make this model the current model")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    settings = load_settings(args.config)
    if args.no_azure:
        settings = copy.deepcopy(settings)
        settings["features"]["blocks"]["embeddings"]["enabled"] = False
    prepared = prepare_dataset(args.csv, settings)
    try:
        embedder = embedder_for_command(settings, prepared.data, args.yes, input_fn, client,
                                        enabled=settings["features"]["blocks"]["embeddings"]["enabled"])
    except CommandStopped as stop:
        return stop.exit_code

    trained = train_and_save(prepared, settings, embedder, promote_model=args.promote)
    outcome, version, report_path = trained.outcome, trained.version, trained.report_path
    result = outcome.result

    m, b = outcome.metrics, outcome.baseline
    print(f"\nModel {version} · blocks: {', '.join(result.pipeline.active_blocks)} · C = {result.best_c}")
    print(f"Macro F1: {m.f1_macro:.3f} (trivial baseline {b.f1_macro:.3f})")
    print(f"Micro F1: {m.f1_micro:.3f} (trivial baseline {b.f1_micro:.3f})")
    if embedder is not None:
        print(f"Azure cost of training: ${outcome.azure_cost_usd:.4f}")
    status = "current model" if current_version(settings) == version else "not promoted (use --promote)"
    print(f"Status: {status}")
    print(f"Report: {report_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
