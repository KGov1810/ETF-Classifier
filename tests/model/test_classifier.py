import numpy as np
import pytest

from etf_classifier.model.classifier import SingleLabelClassifier, decide


def test_decide_keeps_exactly_the_best_category():
    probas = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    assert decide(probas).tolist() == [[0, 1, 0], [1, 0, 0]]


def _fit_toy():
    x = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    y = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
    model = SingleLabelClassifier(classes=["Gold", "ESG", "Bonds"], C=10.0, seed=0, max_iter=1000)
    return model.fit(x, y), x


def test_scores_form_a_distribution_over_the_categories():
    model, x = _fit_toy()
    probas = model.predict_proba(x)
    assert probas.shape == (6, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_prediction_has_one_category_and_sorted_scores():
    model, x = _fit_toy()
    prediction = model.predict(x[:1], uncertain_threshold=0.5)[0]
    assert prediction.categories == ["Gold"]
    assert [c for c, _ in prediction.scores][0] == "Gold"
    assert [s for _, s in prediction.scores] == sorted((s for _, s in prediction.scores), reverse=True)


def test_uncertain_flag_when_best_score_is_low():
    model, x = _fit_toy()
    assert model.predict(x[:1], uncertain_threshold=0.3)[0].uncertain is False
    assert model.predict(x[:1], uncertain_threshold=0.999)[0].uncertain is True


def test_two_categories_give_two_complementary_scores():
    x = np.array([[0.0], [0.1], [1.0], [1.1]])
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    model = SingleLabelClassifier(classes=["Low", "High"], C=10.0, seed=0, max_iter=1000).fit(x, y)
    assert model.predict(np.array([[1.2]]), uncertain_threshold=0.5)[0].categories == ["High"]
    assert set(model.coefficients()) == {"Low", "High"}


def test_a_single_training_category_is_always_predicted():
    x = np.array([[1.0], [2.0]])
    y = np.array([[0, 1], [0, 1]])
    model = SingleLabelClassifier(classes=["Never", "Always"], C=1.0, seed=0, max_iter=1000).fit(x, y)
    assert model.predict_proba(x).tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert model.coefficients() == {"Never": None, "Always": None}


def test_rows_without_a_known_category_are_ignored_in_training():
    x = np.array([[0.0], [1.0], [5.0]])
    y = np.array([[1, 0], [0, 1], [0, 0]])
    model = SingleLabelClassifier(classes=["A", "B"], C=1.0, seed=0, max_iter=1000).fit(x, y)
    assert model.predict_proba(x).shape == (3, 2)


@pytest.mark.parametrize("bad_y", [np.array([[1, 1]]), np.array([[0, 2]])])
def test_more_than_one_category_per_etf_is_refused(bad_y):
    with pytest.raises(ValueError, match="one category"):
        SingleLabelClassifier(classes=["A", "B"], C=1.0, seed=0, max_iter=1000).fit(np.array([[0.0]]), bad_y)
