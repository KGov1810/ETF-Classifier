"""Daily retraining: detect changes, train a candidate, compare it fairly, promote it only if it is not worse.

Usage (what the Windows scheduler runs every day):
    python -m etf_classifier.retraining.run [--force] [--dry-run] [--config …]

Exit codes: 0 = promoted, not promoted or skipped (normal outcomes); 1 = failure.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass

import pandas as pd

from etf_classifier.common import markdown_table, utc_timestamp
from etf_classifier.config import load_settings, resolve_path
from etf_classifier.data.loader import prepare_dataset
from etf_classifier.features.embeddings import create_embedder
from etf_classifier.features.pipeline import dataset_fingerprint
from etf_classifier.model.registry import current_version, load_bundle, promote
from etf_classifier.model.train import train_and_save
from etf_classifier.retraining.compare import Scores, comparison_set, decide, score, shared_categories
from etf_classifier.retraining.history import (
    FAILED, NOT_PROMOTED, PROMOTED, SKIPPED, RetrainingBusy, append_run, job_lock, last_trained_fingerprint,
)

EXIT_OK, EXIT_FAILED = 0, 1
logger = logging.getLogger("etf_classifier.retraining")


def _setup_logging(settings: dict) -> logging.Handler:
    path = resolve_path(settings["retraining"]["log_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def _empty_scores() -> Scores:
    return Scores(0.0, pd.DataFrame(columns=["category", "f1", "support"]))


def _report(started: str, outcome: str, info: dict, checks: list, per_category: pd.DataFrame | None,
            new_categories: list[str]) -> str:
    lines = [f"# Retraining {started[:10]}", "", f"**Outcome: {outcome.replace('_', ' ')}**", ""]
    lines += [f"- **{key}**: {value}" for key, value in info.items()]
    if checks:
        lines += ["", "## Promotion checks", ""]
        lines += [f"- {'✅' if ok else '❌'} {text}" for ok, text in checks]
    if per_category is not None:  # categories absent from the comparison set and never predicted say nothing
        per_category = per_category[(per_category["support_current"] > 0) | (per_category["f1_current"] > 0)
                                    | (per_category["f1_candidate"] > 0)]
    if per_category is not None and not per_category.empty:
        rows = [[r.category, int(r.support_current), f"{r.f1_current:.2f}", f"{r.f1_candidate:.2f}"]
                for r in per_category.itertuples()]
        lines += ["", "## Per category (comparison set)", "",
                  markdown_table(["Category", "Support", "Current F1", "Candidate F1"], rows)]
    if new_categories:
        lines += ["", "## New categories (not compared)", "", ", ".join(new_categories)]
    return "\n".join(lines) + "\n"


@dataclass
class _Comparison:
    checks: list[tuple[bool, str]]
    per_category: pd.DataFrame
    new_categories: list[str]
    description: str   # how the comparison set was built, for the report
    record: dict       # history columns filled by the comparison


def _compare_with_current(trained, prepared, current_bundle, settings: dict, embedder) -> _Comparison:
    """Scores both models on the test ETFs the current model never saw, then applies the promotion rule."""
    candidate_bundle = load_bundle(trained.version, settings, embedder)
    comparison = comparison_set(prepared.split.test, current_bundle)
    shared, new_categories, _ = shared_categories(current_bundle, candidate_bundle)
    n = len(comparison.data)
    if n and shared:
        current_scores = score(current_bundle, comparison.data, shared)
        candidate_scores = score(candidate_bundle, comparison.data, shared)
    else:
        current_scores, candidate_scores = _empty_scores(), _empty_scores()
    decision = decide(current_scores, candidate_scores, trained.outcome.metrics.f1_macro,
                      trained.outcome.baseline.f1_macro, n, settings["retraining"])
    if comparison.whole_test_set:
        description = (f"{n} ETF(s): whole test set (the current model predates this module and may have seen "
                       "some of them, which favours it)")
    else:
        description = (f"{n} ETF(s); {comparison.n_seen_excluded} test ETF(s) excluded because the current model "
                       "trained on them")
    per_category = current_scores.per_category.merge(candidate_scores.per_category, on="category",
                                                     suffixes=("_current", "_candidate"))
    record = {"comparison_etfs": n, "current_f1_macro": f"{current_scores.f1_macro:.3f}",
              "candidate_f1_macro": f"{candidate_scores.f1_macro:.3f}"}
    return _Comparison(decision.checks, per_category, new_categories, description, record)


def retrain(settings: dict, force: bool = False, dry_run: bool = False, client=None) -> dict:
    """One run under the job lock; returns the history record (raises on failure)."""
    started = utc_timestamp()
    t0 = time.perf_counter()
    record = {"started_at": started}
    with job_lock(settings):
        prepared = prepare_dataset(settings["feedback"]["reference_csv"], settings)
        fingerprint = dataset_fingerprint(prepared.data)
        current = current_version(settings)
        record["current_version"] = current or ""
        embedder = None
        if settings["features"]["blocks"]["embeddings"]["enabled"]:
            embedder = create_embedder(settings, client=client)  # paid without confirmation (approved)
        current_bundle = load_bundle(current, settings, embedder) if current else None

        record["dataset_fingerprint"] = fingerprint
        # nothing changed since the current model, or since the last candidate (even a refused one)
        trained_on = {last_trained_fingerprint(settings)}
        if current_bundle is not None:
            trained_on.add(current_bundle.metrics.get("dataset_fingerprint"))
        if current_bundle is not None and not force and fingerprint in trained_on:
            record.update(outcome=SKIPPED, reasons="No new or cancelled validation since the last training.",
                          duration_seconds=f"{time.perf_counter() - t0:.1f}")
            logger.info("Skipped: dataset unchanged (%s)", fingerprint)
            return record

        trained = train_and_save(prepared, settings, embedder, promote_model=False,
                                 make_current_if_first=not dry_run)
        record.update(candidate_version=trained.version, azure_cost_usd=f"{trained.outcome.azure_cost_usd:.4f}")
        n_validations = int((prepared.data["source"] == "validation").sum())
        info = {"Candidate": trained.version, "Current model": current or "none",
                "Usable ETFs": f"{len(prepared.data)} (of which {n_validations} from validations)",
                "Candidate macro F1 on its test set": f"{trained.outcome.metrics.f1_macro:.3f} "
                                                      f"(trivial baseline {trained.outcome.baseline.f1_macro:.3f})",
                "Azure cost": f"${trained.outcome.azure_cost_usd:.4f}"}
        per_category, new_categories = None, []
        if current_bundle is None:  # a first model is already current (make_current_if_first), unless dry run
            checks = [] if dry_run else [(True, "First model: it becomes the current model.")]
        else:
            comparison = _compare_with_current(trained, prepared, current_bundle, settings, embedder)
            checks, per_category, new_categories = comparison.checks, comparison.per_category, comparison.new_categories
            info["Comparison set"] = comparison.description
            record.update(comparison.record)
        if dry_run:
            checks = [*checks, (False, "Dry run: never promoted.")]
        promoted = all(ok for ok, _ in checks)
        if promoted and current_bundle is not None:
            promote(trained.version, settings)
        failed = [text for ok, text in checks if not ok]
        outcome = PROMOTED if promoted else NOT_PROMOTED
        reasons = "; ".join(failed) if failed else (checks[0][1] if current_bundle is None
                                                    else "All promotion checks passed.")

        record.update(outcome=outcome, reasons=reasons, duration_seconds=f"{time.perf_counter() - t0:.1f}")
        report_path = resolve_path(settings["model"]["reports_dir"]) / f"retraining_{trained.version}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_report(started, outcome, info, checks, per_category,
                                       new_categories), encoding="utf-8")
        logger.info("%s: candidate %s (current %s) · %s", outcome, trained.version, current, reasons)
        return record


def main(argv: list[str] | None = None, client=None) -> int:
    parser = argparse.ArgumentParser(description="Daily retraining with fair comparison and conditional promotion")
    parser.add_argument("--config", help="Configuration file (default: config/settings.yaml)")
    parser.add_argument("--force", action="store_true", help="Retrain even if nothing changed")
    parser.add_argument("--dry-run", action="store_true", help="Train and compare, never promote")
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    handler = _setup_logging(settings)
    started = utc_timestamp()
    try:
        record = retrain(settings, force=args.force, dry_run=args.dry_run, client=client)
    except RetrainingBusy as exc:
        print(str(exc))
        logger.warning(str(exc))
        append_run(settings, {"started_at": started, "outcome": SKIPPED, "reasons": str(exc)})
        return EXIT_OK
    except Exception as exc:  # any failure must be visible: log, history, exit code 1
        logger.exception("Retraining failed")
        append_run(settings, {"started_at": started, "outcome": FAILED, "reasons": f"{type(exc).__name__}: {exc}"})
        print(f"Retraining failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    finally:
        logger.removeHandler(handler)
        handler.close()
    append_run(settings, record)
    print(f"{record['outcome']}: {record.get('reasons', '')}")
    if record.get("candidate_version"):
        report = resolve_path(settings["model"]["reports_dir"]) / f"retraining_{record['candidate_version']}.md"
        print(f"Report: {report}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
