"""Fair comparison of the current model and the candidate, and the promotion rule.

Both models are scored on the same test ETFs, which the current model never saw during training,
and only on the categories both of them know.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from etf_classifier.model.evaluate import evaluate
from etf_classifier.model.train import TRAIN_NAMES_FILE


@dataclass
class ComparisonSet:
    data: pd.DataFrame
    n_seen_excluded: int      # test ETFs removed because the current model trained on them
    whole_test_set: bool      # True for a current model trained before this module (no training names)


@dataclass
class Scores:
    f1_macro: float
    per_category: pd.DataFrame  # columns: category, f1, support


@dataclass
class Decision:
    promote: bool
    checks: list[tuple[bool, str]] = field(default_factory=list)  # (passed, explanation), in order

    @property
    def reasons(self) -> list[str]:
        """Explanations of the failed checks (empty when promoted)."""
        return [text for ok, text in self.checks if not ok]


def comparison_set(candidate_test: pd.DataFrame, current_bundle) -> ComparisonSet:
    """Candidate's test ETFs that the current model did not train on."""
    names_file = current_bundle.path / TRAIN_NAMES_FILE
    if not names_file.exists():
        return ComparisonSet(candidate_test.reset_index(drop=True), 0, True)
    seen = set(json.loads(names_file.read_text(encoding="utf-8")))
    keep = ~candidate_test["name"].isin(seen)
    return ComparisonSet(candidate_test[keep].reset_index(drop=True), int((~keep).sum()), False)


def shared_categories(current_bundle, candidate_bundle) -> tuple[list[str], list[str], list[str]]:
    """(categories known by both, new in the candidate, unknown to the candidate)."""
    current, candidate = set(current_bundle.model.classes), set(candidate_bundle.model.classes)
    return sorted(current & candidate), sorted(candidate - current), sorted(current - candidate)


def score(bundle, data: pd.DataFrame, categories: list[str], settings: dict) -> Scores:
    """Macro F1 and per-category F1 of a model on `data`, restricted to `categories`."""
    model = bundle.model
    probas = model.predict_proba(bundle.pipeline.transform(data.reset_index(drop=True)).matrix)
    decided = model.decide(probas, settings["model"]["max_categories"])
    columns = [list(model.classes).index(c) for c in categories]
    y_pred = np.asarray(decided)[:, columns]
    y_true = np.array([[int(c in etf_categories) for c in categories] for etf_categories in data["categories"]],
                      dtype=int).reshape(len(data), len(categories))
    metrics = evaluate(y_true, y_pred, categories)
    return Scores(metrics.f1_macro, metrics.per_category[["category", "f1", "support"]].reset_index(drop=True))


def decide(current: Scores, candidate: Scores, candidate_test_f1: float, baseline_test_f1: float,
           n_etfs: int, cfg: dict) -> Decision:
    """Promotion rule (approved on 2026-10-25). Every check is reported, passed or not."""
    checks = []
    minimum = int(cfg["min_comparison_etfs"])
    checks.append((n_etfs >= minimum,
                   f"Comparison set: {n_etfs} ETF(s) (minimum {minimum})."))

    tolerance = float(cfg["f1_tolerance"])
    checks.append((candidate.f1_macro >= current.f1_macro - tolerance,
                   f"Comparison macro F1: candidate {candidate.f1_macro:.3f}, current {current.f1_macro:.3f} "
                   f"(tolerance {tolerance:.2f})."))

    drop_max, min_support = float(cfg["category_drop_max"]), int(cfg["category_min_support"])
    merged = current.per_category.merge(candidate.per_category, on="category", suffixes=("_current", "_candidate"))
    collapsed = merged[(merged["support_current"] >= min_support)
                       & (merged["f1_current"] - merged["f1_candidate"] > drop_max)]
    if collapsed.empty:
        checks.append((True, f"No category loses more than {drop_max:.2f} of F1 "
                             f"(categories with at least {min_support} test examples)."))
    else:
        details = ", ".join(f"{r.category} {r.f1_current:.2f} → {r.f1_candidate:.2f}" for r in collapsed.itertuples())
        checks.append((False, f"Category F1 collapses: {details}."))

    checks.append((candidate_test_f1 > baseline_test_f1,
                   f"Candidate macro F1 on its test set {candidate_test_f1:.3f} vs trivial baseline "
                   f"{baseline_test_f1:.3f}."))
    return Decision(all(ok for ok, _ in checks), checks)
