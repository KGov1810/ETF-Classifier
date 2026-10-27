import numpy as np

from etf_classifier.model.baseline import FrequencyBaseline


def test_baseline_predicts_the_most_frequent_category_for_every_etf():
    y_train = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert FrequencyBaseline().fit(y_train).predict(2).tolist() == [[0, 1, 0], [0, 1, 0]]


def test_baseline_breaks_ties_by_category_order():
    assert FrequencyBaseline().fit(np.array([[0, 1], [1, 0]])).predict(1).tolist() == [[1, 0]]
