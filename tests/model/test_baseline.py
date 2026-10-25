import numpy as np

from etf_classifier.model.baseline import FrequencyBaseline


def test_baseline_predicts_the_most_frequent_categories():
    y_train = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0], [0, 0, 1]])
    baseline = FrequencyBaseline().fit(y_train)
    assert baseline.predict(2).tolist() == [[1, 0, 0], [1, 0, 0]]


def test_baseline_uses_the_median_number_of_categories():
    y_train = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1]])
    baseline = FrequencyBaseline().fit(y_train)
    assert baseline.n_predicted == 2
    assert baseline.predict(1).tolist() == [[1, 1, 0]]


def test_baseline_always_predicts_at_least_one_category():
    baseline = FrequencyBaseline().fit(np.array([[0, 1], [0, 0]]))
    assert baseline.predict(1).sum() == 1
