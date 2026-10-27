"""Single-category classifier: one multinomial logistic regression (softmax) over the categories.

Every ETF gets exactly one category, the one with the highest score; the scores of an ETF sum to 1.
Tuning (regularisation strength C) is done on an internal validation set drawn from the training set.
tune() never receives the test set: leakage is impossible by construction.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from etf_classifier.data.split import make_split
from etf_classifier.features.pipeline import FeaturePipeline
from etf_classifier.model.evaluate import evaluate

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Proposal for an ETF: the category (one-element list), every score sorted, "uncertain" flag."""

    categories: list[str]
    scores: list[tuple[str, float]]
    uncertain: bool


def decide(probas: np.ndarray) -> np.ndarray:
    """One-hot matrix of the best category of each row."""
    probas = np.asarray(probas)
    decided = np.zeros(probas.shape, dtype=np.int8)
    if probas.size:
        decided[np.arange(len(probas)), probas.argmax(axis=1)] = 1
    return decided


class SingleLabelClassifier:
    def __init__(self, classes: list[str], C: float, seed: int, max_iter: int):
        self.classes = list(classes)
        self.C = C
        self._model = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed)
        self._constant: int | None = None  # index of the only category seen in training, if so

    def fit(self, x, y: np.ndarray) -> "SingleLabelClassifier":
        """y: one-hot matrix (one row per ETF); rows without a known category are ignored."""
        y = np.asarray(y)
        if (y.sum(axis=1) > 1).any() or (y > 1).any():
            raise ValueError("Exactly one category per ETF is expected.")
        known = y.sum(axis=1) == 1
        target = y[known].argmax(axis=1)
        if len(np.unique(target)) == 1:
            self._constant = int(target[0])
        else:
            self._constant = None
            self._model.fit(x[np.flatnonzero(known)], target)
        return self

    def predict_proba(self, x) -> np.ndarray:
        """Scores of every category (columns in the order of `classes`), each row summing to 1."""
        probas = np.zeros((x.shape[0], len(self.classes)))
        if self._constant is not None:
            probas[:, self._constant] = 1.0
        else:
            probas[:, self._model.classes_] = self._model.predict_proba(x)
        return probas

    def coefficients(self) -> dict[str, np.ndarray | None]:
        """Coefficient vector of each category (None when it cannot be learned: absent or the only one)."""
        result: dict[str, np.ndarray | None] = {c: None for c in self.classes}
        if self._constant is not None:
            return result
        coef = np.asarray(self._model.coef_)
        learned = list(self._model.classes_)
        if len(learned) == 2:  # scikit-learn stores one vector for two classes: the second one's
            coef = np.vstack([-coef[0], coef[0]])
        for row, index in enumerate(learned):
            result[self.classes[index]] = coef[row]
        return result

    def decide(self, probas: np.ndarray) -> np.ndarray:
        return decide(probas)

    def predict(self, x, uncertain_threshold: float) -> list[Prediction]:
        predictions = []
        for scores in self.predict_proba(x):
            order = np.argsort(-scores, kind="stable")
            predictions.append(Prediction(
                categories=[self.classes[order[0]]],
                scores=[(self.classes[j], float(scores[j])) for j in order],
                uncertain=bool(scores[order[0]] < uncertain_threshold),
            ))
        return predictions


@dataclass
class TuningResult:
    model: SingleLabelClassifier
    pipeline: FeaturePipeline
    best_c: float
    validation_scores: dict[float, float]
    n_validation: int


def tune(train: pd.DataFrame, settings: dict, embedder=None) -> TuningResult:
    """Chooses C (macro F1) on an internal validation set, then refits on the whole training set."""
    cfg = settings["model"]
    inner = make_split(train.reset_index(drop=True), cfg["validation_size"], cfg["seed"],
                       settings["split"]["rare_category_min"])
    inner_pipeline = FeaturePipeline(settings, embedder).fit(inner.train)
    x_fit, x_val = inner_pipeline.transform(inner.train).matrix, inner_pipeline.transform(inner.test).matrix
    y_fit, y_val = inner_pipeline.transform_labels(inner.train), inner_pipeline.transform_labels(inner.test)
    classes = inner_pipeline.labels.classes

    scores = {}
    for c in cfg["c_grid"]:
        model = SingleLabelClassifier(classes, c, cfg["seed"], cfg["max_iter"]).fit(x_fit, y_fit)
        scores[c] = evaluate(y_val, model.decide(model.predict_proba(x_val)), classes).f1_macro
    best_c = max(cfg["c_grid"], key=lambda c: (scores[c], -c))  # on a tie, the strongest regularisation
    logger.info("Validation (macro F1) per C: %s → selected C = %s", scores, best_c)

    pipeline = FeaturePipeline(settings, embedder).fit(train)
    final = SingleLabelClassifier(pipeline.labels.classes, best_c, cfg["seed"], cfg["max_iter"])
    final.fit(pipeline.transform(train).matrix, pipeline.transform_labels(train))
    return TuningResult(final, pipeline, best_c, scores, len(inner.test))
