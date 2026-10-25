import numpy as np
import pytest

from etf_classifier.model.classifier import MultiLabelClassifier, best_threshold, decide


def test_threshold_maximises_the_f1_of_the_category():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.35, 0.4, 0.7, 0.9])
    assert best_threshold(y_true, scores, min_positives=3, default=0.5) == pytest.approx(0.4)


def test_threshold_falls_back_to_default_with_too_few_positives():
    y_true = np.array([0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3])
    assert best_threshold(y_true, scores, min_positives=3, default=0.5) == 0.5


def test_at_least_one_category_is_always_kept():
    probas = np.array([[0.2, 0.3, 0.1]])
    assert decide(probas, thresholds=np.full(3, 0.5), max_categories=5).tolist() == [[0, 1, 0]]


def test_never_more_than_max_categories():
    probas = np.array([[0.9, 0.8, 0.7, 0.95]])
    assigned = decide(probas, thresholds=np.full(4, 0.5), max_categories=2)
    assert assigned.tolist() == [[1, 0, 0, 1]]


def test_each_category_uses_its_own_threshold():
    probas = np.array([[0.6, 0.6]])
    assert decide(probas, thresholds=np.array([0.5, 0.7]), max_categories=5).tolist() == [[1, 0]]


def _fit_toy():
    x = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=float)
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]])
    model = MultiLabelClassifier(classes=["Gold", "ESG"], C=10.0, seed=0, max_iter=1000)
    return model.fit(x, y), x


def test_predictions_list_sorted_scores_and_kept_categories():
    model, x = _fit_toy()
    prediction = model.predict(x[:1], max_categories=5, uncertain_threshold=0.5)[0]
    assert prediction.categories == ["Gold"]
    assert [c for c, _ in prediction.scores] == ["Gold", "ESG"]
    assert prediction.scores[0][1] >= prediction.scores[1][1]


def test_uncertain_flag_when_best_score_is_low():
    model, x = _fit_toy()
    confident = model.predict(x[:1], max_categories=5, uncertain_threshold=0.5)[0]
    doubtful = model.predict(x[:1], max_categories=5, uncertain_threshold=0.999)[0]
    assert confident.uncertain is False
    assert doubtful.uncertain is True


def test_category_present_in_every_training_example_is_handled():
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[1, 0], [1, 1], [1, 0]])
    model = MultiLabelClassifier(classes=["Always", "Sometimes"], C=1.0, seed=0, max_iter=1000).fit(x, y)
    assert model.predict_proba(x).shape == (3, 2)
