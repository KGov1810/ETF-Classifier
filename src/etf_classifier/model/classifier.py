"""Classifieur multi-étiquettes : une régression logistique par catégorie, seuils réglés par catégorie.

Le réglage (force de régularisation C et seuils) se fait sur une validation interne tirée du jeu
d'entraînement. tune() ne reçoit jamais le jeu de test : la fuite est impossible par construction.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from etf_classifier.data.split import make_split
from etf_classifier.features.pipeline import FeaturePipeline
from etf_classifier.model.evaluate import evaluate

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Proposition pour un ETF : catégories retenues, scores triés, indicateur « incertain »."""

    categories: list[str]
    scores: list[tuple[str, float]]
    uncertain: bool


def best_threshold(y_true: np.ndarray, scores: np.ndarray, min_positives: int, default: float) -> float:
    """Seuil qui maximise le F1 d'une catégorie ; valeur par défaut s'il y a trop peu d'exemples positifs."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if y_true.sum() < min_positives:
        return default
    best, best_f1 = default, -1.0
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        tp = np.sum(predicted & (y_true == 1))
        f1 = 2 * tp / (predicted.sum() + y_true.sum()) if (predicted.sum() + y_true.sum()) else 0.0
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best


def decide(probas: np.ndarray, thresholds: np.ndarray, max_categories: int) -> np.ndarray:
    """Catégories retenues : au-dessus de leur seuil, au plus max_categories, au moins la meilleure."""
    probas = np.asarray(probas)
    decided = np.zeros(probas.shape, dtype=np.int8)
    for row, scores in enumerate(probas):
        order = np.argsort(-scores, kind="stable")
        kept = [j for j in order if scores[j] >= thresholds[j]][:max_categories]
        decided[row, kept or [order[0]]] = 1
    return decided


class MultiLabelClassifier:
    def __init__(self, classes: list[str], C: float, seed: int, max_iter: int):
        self.classes = list(classes)
        self.C = C
        self.thresholds = np.full(len(self.classes), 0.5)
        self._model = OneVsRestClassifier(LogisticRegression(
            C=C, class_weight="balanced", solver="liblinear", max_iter=max_iter, random_state=seed))

    def fit(self, x, y: np.ndarray) -> "MultiLabelClassifier":
        self._model.fit(x, np.asarray(y))
        return self

    def predict_proba(self, x) -> np.ndarray:
        return np.asarray(self._model.predict_proba(x))

    @property
    def estimators(self) -> list:
        return self._model.estimators_

    def decide(self, probas: np.ndarray, max_categories: int = 5) -> np.ndarray:
        return decide(probas, self.thresholds, max_categories)

    def predict(self, x, max_categories: int, uncertain_threshold: float) -> list[Prediction]:
        probas = self.predict_proba(x)
        decided = self.decide(probas, max_categories)
        predictions = []
        for scores, kept in zip(probas, decided):
            order = np.argsort(-scores, kind="stable")
            predictions.append(Prediction(
                categories=[self.classes[j] for j in order if kept[j]],
                scores=[(self.classes[j], float(scores[j])) for j in order],
                uncertain=bool(scores[order[0]] < uncertain_threshold),
            ))
        return predictions


@dataclass
class TuningResult:
    model: MultiLabelClassifier
    pipeline: FeaturePipeline
    best_c: float
    validation_scores: dict[float, float]
    n_validation: int


def _thresholds(model, y_val, probas, cfg) -> np.ndarray:
    return np.array([best_threshold(y_val[:, j], probas[:, j], cfg["min_positives_for_threshold"],
                                    cfg["default_threshold"]) for j in range(y_val.shape[1])])


def tune(train: pd.DataFrame, settings: dict, embedder=None) -> TuningResult:
    """Choisit C (F1 macro) et les seuils sur une validation interne, puis réentraîne sur tout l'entraînement."""
    cfg = settings["model"]
    inner = make_split(train.reset_index(drop=True), cfg["validation_size"], cfg["seed"],
                       settings["split"]["rare_category_min"])
    inner_pipeline = FeaturePipeline(settings, embedder).fit(inner.train)
    x_fit, x_val = inner_pipeline.transform(inner.train).matrix, inner_pipeline.transform(inner.test).matrix
    y_fit, y_val = inner_pipeline.transform_labels(inner.train), inner_pipeline.transform_labels(inner.test)
    classes = inner_pipeline.labels.classes

    scores, thresholds_by_c = {}, {}
    for c in cfg["c_grid"]:
        model = MultiLabelClassifier(classes, c, cfg["seed"], cfg["max_iter"]).fit(x_fit, y_fit)
        probas = model.predict_proba(x_val)
        model.thresholds = _thresholds(model, y_val, probas, cfg)
        scores[c] = evaluate(y_val, model.decide(probas, cfg["max_categories"]), classes).f1_macro
        thresholds_by_c[c] = dict(zip(classes, model.thresholds))
    best_c = max(cfg["c_grid"], key=lambda c: (scores[c], -c))  # à égalité, la régularisation la plus forte
    logger.info("Validation (F1 macro) par C : %s → C retenu = %s", scores, best_c)

    pipeline = FeaturePipeline(settings, embedder).fit(train)
    final = MultiLabelClassifier(pipeline.labels.classes, best_c, cfg["seed"], cfg["max_iter"])
    final.fit(pipeline.transform(train).matrix, pipeline.transform_labels(train))
    chosen = thresholds_by_c[best_c]
    final.thresholds = np.array([chosen.get(c, cfg["default_threshold"]) for c in final.classes])
    return TuningResult(final, pipeline, best_c, scores, len(inner.test))
