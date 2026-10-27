import numpy as np
import pytest

from etf_classifier.model.evaluate import evaluate, render_report


@pytest.fixture
def report():
    classes = ["Gold", "ESG"]
    y_true = np.array([[1, 0], [0, 1], [0, 1]])
    metrics = evaluate(y_true, np.array([[1, 0], [0, 1], [1, 0]]), classes)
    baseline = evaluate(y_true, np.array([[1, 0], [1, 0], [1, 0]]), classes)
    top = {"Gold": [("text:gold", 2.1), ("holding:NEM", 1.7)], "ESG": []}
    return render_report("20261025-120000", metrics, baseline, top, {"Crypto": 2}, {"Selected C (validation)": 1.0})


@pytest.mark.parametrize("section", [
    "## Comparison with the trivial baseline", "## Details per category",
    "## Most frequent confusions", "## Most influential features", "## Categories that cannot be learned",
])
def test_report_contains_every_section(report, section):
    assert section in report


def test_report_shows_accuracy_and_macro_f1_against_the_baseline(report):
    assert "| Accuracy | 0.667 | 0.333 |" in report
    assert "Macro F1 (main metric)" in report
    assert "Threshold" not in report and "Hamming" not in report


def test_report_lists_influential_features_and_confusions(report):
    assert "text:gold, holding:NEM" in report
    assert "| ESG | Gold | 1 |" in report


def test_report_lists_categories_unknown_to_training(report):
    assert "| Crypto | 2 |" in report


def test_constant_category_is_explained(report):
    assert "none (constant category)" in report
