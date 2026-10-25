"""Multi-label metrics, identical for the trivial baseline and for the classifier."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


@dataclass
class Metrics:
    f1_micro: float
    f1_macro: float
    f1_samples: float
    subset_accuracy: float
    hamming_loss: float
    per_category: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "f1_micro": self.f1_micro,
            "f1_macro": self.f1_macro,
            "f1_samples": self.f1_samples,
            "subset_accuracy": self.subset_accuracy,
            "hamming_loss": self.hamming_loss,
            "per_category": self.per_category.to_dict(orient="records"),
        }


def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]
    return "\n".join(lines)


def render_report(version: str, metrics: Metrics, baseline: Metrics, top: dict, thresholds: dict[str, float],
                  unknown_categories: dict[str, int], info: dict) -> str:
    """Markdown evaluation report of a model (test set, evaluated only once)."""
    rows = [["Macro F1 (main metric)", f"{metrics.f1_macro:.3f}", f"{baseline.f1_macro:.3f}"],
            ["Micro F1", f"{metrics.f1_micro:.3f}", f"{baseline.f1_micro:.3f}"],
            ["Per-ETF F1", f"{metrics.f1_samples:.3f}", f"{baseline.f1_samples:.3f}"],
            ["Exact match", f"{metrics.subset_accuracy:.3f}", f"{baseline.subset_accuracy:.3f}"],
            ["Hamming loss (lower is better)", f"{metrics.hamming_loss:.3f}", f"{baseline.hamming_loss:.3f}"]]
    lines = [
        f"# Evaluation report: model {version}", "",
        _table(["Information", "Value"], [[k, v] for k, v in info.items()]), "",
        "## Comparison with the trivial baseline", "",
        "The trivial baseline predicts the most frequent categories, without looking at the ETF.", "",
        _table(["Metric (test set)", "Model", "Trivial baseline"], rows), "",
        "## Details per category", "",
    ]
    details = metrics.per_category.sort_values("support", ascending=False)
    lines += [_table(["Category", "Precision", "Recall", "F1", "Support (test)", "Threshold"],
                     [[r.category, f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", int(r.support),
                       f"{thresholds.get(r.category, 0.5):.2f}"] for r in details.itertuples()]), ""]
    lines += ["## Most influential features", "",
              "Features that push most towards each category (positive coefficients).", ""]
    lines += [_table(["Category", "Features"],
                     [[c, ", ".join(name for name, _ in feats) or "none (constant category)"]
                      for c, feats in sorted(top.items())]), ""]
    lines += ["## Categories that cannot be learned", ""]
    if unknown_categories:
        lines += ["Present in the test set but missing from the training set: the model cannot predict them.", "",
                  _table(["Category", "ETFs concerned (test)"], [[c, n] for c, n in unknown_categories.items()]), ""]
    else:
        lines += ["None.", ""]
    return "\n".join(lines)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> Metrics:
    """Computes the metrics.

    Macro F1: average over the categories present in the true labels OR in the predictions.
    A category missing on both sides has nothing to measure and is not counted; a category
    predicted wrongly but missing from the true labels counts (F1 = 0), as it is a real error.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    active = np.flatnonzero((y_true.sum(axis=0) + y_pred.sum(axis=0)) > 0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(classes)), average=None, zero_division=0
    )
    per_category = pd.DataFrame({
        "category": classes, "precision": precision, "recall": recall, "f1": f1, "support": support,
    })
    if y_true.size == 0:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, per_category)
    return Metrics(
        f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        f1_macro=float(f1[active].mean()) if active.size else 0.0,
        f1_samples=float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        subset_accuracy=float((y_true == y_pred).all(axis=1).mean()),
        hamming_loss=float(hamming_loss(y_true, y_pred)),
        per_category=per_category,
    )
