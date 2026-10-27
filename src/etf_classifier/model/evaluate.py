"""Single-category metrics, identical for the trivial baseline and for the classifier.

Labels are one-hot matrices (one row per ETF); a row of zeros is an ETF whose category is unknown to training.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from etf_classifier.common import markdown_table


MAX_CONFUSIONS = 10


@dataclass
class Metrics:
    accuracy: float
    f1_macro: float
    per_category: pd.DataFrame
    confusions: list[tuple[str, str, int]]  # (true category, predicted category, number of ETFs), most frequent first

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "per_category": self.per_category.to_dict(orient="records"),
            "confusions": [list(c) for c in self.confusions],
        }


def render_report(version: str, metrics: Metrics, baseline: Metrics, top: dict,
                  unknown_categories: dict[str, int], info: dict) -> str:
    """Markdown evaluation report of a model (test set, evaluated only once)."""
    rows = [["Macro F1 (main metric)", f"{metrics.f1_macro:.3f}", f"{baseline.f1_macro:.3f}"],
            ["Accuracy", f"{metrics.accuracy:.3f}", f"{baseline.accuracy:.3f}"]]
    lines = [
        f"# Evaluation report: model {version}", "",
        markdown_table(["Information", "Value"], [[k, v] for k, v in info.items()]), "",
        "## Comparison with the trivial baseline", "",
        "The trivial baseline predicts the most frequent category, without looking at the ETF. "
        "Accuracy: share of ETFs given the right category; macro F1: average over the categories, "
        "so that rare categories weigh as much as frequent ones.", "",
        markdown_table(["Metric (test set)", "Model", "Trivial baseline"], rows), "",
        "## Details per category", "",
    ]
    details = metrics.per_category.sort_values("support", ascending=False)
    lines += [markdown_table(["Category", "Precision", "Recall", "F1", "Support (test)"],
                     [[r.category, f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", int(r.support)]
                      for r in details.itertuples()]), ""]
    lines += ["## Most frequent confusions", "", "True category of test ETFs given another category.", ""]
    lines += ([markdown_table(["True category", "Predicted category", "ETFs"], [list(c) for c in metrics.confusions]),
               ""] if metrics.confusions else ["None.", ""])
    lines += ["## Most influential features", "",
              "Features that push most towards each category (positive coefficients).", ""]
    lines += [markdown_table(["Category", "Features"],
                     [[c, ", ".join(name for name, _ in feats) or "none (constant category)"]
                      for c, feats in sorted(top.items())]), ""]
    lines += ["## Categories that cannot be learned", ""]
    if unknown_categories:
        lines += ["Present in the test set but missing from the training set: the model cannot predict them.", "",
                  markdown_table(["Category", "ETFs concerned (test)"],
                                 [[c, n] for c, n in unknown_categories.items()]), ""]
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
        return Metrics(0.0, 0.0, per_category, [])
    known = y_true.sum(axis=1) == 1
    pairs = Counter((classes[t], classes[p]) for t, p in zip(y_true[known].argmax(axis=1), y_pred[known].argmax(axis=1))
                    if t != p)
    confusions = sorted(((t, p, n) for (t, p), n in pairs.items()), key=lambda c: (-c[2], c[0], c[1]))
    return Metrics(
        accuracy=float((y_true == y_pred).all(axis=1).mean()),
        f1_macro=float(f1[active].mean()) if active.size else 0.0,
        per_category=per_category,
        confusions=confusions[:MAX_CONFUSIONS],
    )
