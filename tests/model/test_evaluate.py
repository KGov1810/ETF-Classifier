import numpy as np
import pytest

from etf_classifier.model.evaluate import evaluate

CLASSES = ["A", "B", "C"]


def test_metrics_computed_by_hand():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    metrics = evaluate(y_true, y_pred, CLASSES)
    # A: TP=2 FP=1 FN=0 → F1 0.8; B: TP=1 FP=0 FN=1 → F1 2/3; C ignored (absent on both sides)
    assert metrics.f1_macro == pytest.approx((0.8 + 2 / 3) / 2)
    # micro: TP=3 FP=1 FN=1 → F1 0.75
    assert metrics.f1_micro == pytest.approx(0.75)
    assert metrics.subset_accuracy == pytest.approx(1 / 3)
    assert metrics.hamming_loss == pytest.approx(2 / 9)


def test_category_absent_on_both_sides_does_not_lower_macro_f1():
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[1, 0, 0], [0, 1, 0]])
    assert evaluate(y_true, y_pred, CLASSES).f1_macro == pytest.approx(1.0)


def test_category_predicted_but_absent_is_penalised():
    y_true = np.array([[1, 0, 0], [1, 0, 0]])
    y_pred = np.array([[1, 0, 1], [1, 0, 0]])
    assert evaluate(y_true, y_pred, CLASSES).f1_macro == pytest.approx(0.5)


def test_per_category_details():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]])
    details = evaluate(y_true, y_pred, CLASSES).per_category.set_index("category")
    assert details.loc["A", "precision"] == pytest.approx(2 / 3)
    assert details.loc["B", "recall"] == pytest.approx(0.5)
    assert details.loc["A", "support"] == 2


def test_metrics_can_be_exported_as_a_dict():
    y = np.array([[1, 0, 0]])
    exported = evaluate(y, y, CLASSES).to_dict()
    assert exported["f1_macro"] == 1.0
    assert "per_category" in exported
