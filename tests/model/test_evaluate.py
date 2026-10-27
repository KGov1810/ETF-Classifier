import numpy as np
import pytest

from etf_classifier.model.evaluate import evaluate

CLASSES = ["A", "B", "C"]


def _one_hot(indices):
    matrix = np.zeros((len(indices), len(CLASSES)), dtype=int)
    for row, index in enumerate(indices):
        if index is not None:
            matrix[row, index] = 1
    return matrix


def test_metrics_computed_by_hand():
    metrics = evaluate(_one_hot([0, 0, 1, 1]), _one_hot([0, 1, 1, 1]), CLASSES)
    assert metrics.accuracy == pytest.approx(3 / 4)
    # A: TP=1 FN=1 → F1 2/3; B: TP=2 FP=1 → F1 0.8; C absent on both sides, not counted
    assert metrics.f1_macro == pytest.approx((2 / 3 + 0.8) / 2)


def test_category_predicted_but_absent_is_penalised():
    metrics = evaluate(_one_hot([0, 0]), _one_hot([0, 2]), CLASSES)
    assert metrics.f1_macro == pytest.approx((2 / 3 + 0.0) / 2)


def test_most_frequent_confusions_are_listed():
    metrics = evaluate(_one_hot([0, 0, 0, 1]), _one_hot([1, 1, 0, 2]), CLASSES)
    assert metrics.confusions == [("A", "B", 2), ("B", "C", 1)]


def test_etf_with_a_category_unknown_to_training_counts_as_an_error():
    metrics = evaluate(_one_hot([None, 0]), _one_hot([0, 0]), CLASSES)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.confusions == []


def test_per_category_details():
    details = evaluate(_one_hot([0, 0, 1]), _one_hot([0, 1, 1]), CLASSES).per_category.set_index("category")
    assert details.loc["A", "recall"] == pytest.approx(0.5)
    assert details.loc["B", "precision"] == pytest.approx(0.5)
    assert details.loc["A", "support"] == 2


def test_metrics_can_be_exported_as_a_dict():
    exported = evaluate(_one_hot([0]), _one_hot([0]), CLASSES).to_dict()
    assert exported["f1_macro"] == 1.0 and exported["accuracy"] == 1.0
    assert "per_category" in exported and "confusions" in exported
