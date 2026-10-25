import numpy as np
import pytest

from etf_classifier.model.evaluate import evaluate, render_report


@pytest.fixture
def report():
    classes = ["Gold", "ESG"]
    y_true = np.array([[1, 0], [0, 1]])
    metrics = evaluate(y_true, np.array([[1, 0], [0, 1]]), classes)
    baseline = evaluate(y_true, np.array([[1, 0], [1, 0]]), classes)
    top = {"Gold": [("text:gold", 2.1), ("holding:NEM", 1.7)], "ESG": []}
    return render_report("20261025-120000", metrics, baseline, top, {"Gold": 0.42, "ESG": 0.5},
                         {"Crypto": 2}, {"C retenu (validation)": 1.0})


@pytest.mark.parametrize("section", [
    "## Comparison with the trivial baseline", "## Details per category",
    "## Most influential features", "## Categories that cannot be learned",
])
def test_report_contains_every_section(report, section):
    assert section in report


def test_report_lists_influential_features_and_thresholds(report):
    assert "text:gold, holding:NEM" in report
    assert "0.42" in report


def test_report_lists_categories_unknown_to_training(report):
    assert "| Crypto | 2 |" in report


def test_constant_category_is_explained(report):
    assert "none (constant category)" in report
