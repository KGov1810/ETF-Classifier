import inspect

import numpy as np
import pytest

from etf_classifier.data.split import make_split
from etf_classifier.model.baseline import FrequencyBaseline
from etf_classifier.model.classifier import tune
from etf_classifier.model.evaluate import evaluate


def test_tune_only_receives_the_training_set():
    assert list(inspect.signature(tune).parameters) == ["train", "settings", "embedder"]


@pytest.fixture
def tuned(synthetic_data, isolated_settings):
    split = make_split(synthetic_data, test_size=0.2, seed=1)
    return split, tune(split.train, isolated_settings)


def test_tuned_model_clearly_beats_the_baseline(tuned):
    split, result = tuned
    x_test = result.pipeline.transform(split.test).matrix
    y_test = result.pipeline.transform_labels(split.test)
    y_train = result.pipeline.transform_labels(split.train)
    model = evaluate(y_test, result.model.decide(result.model.predict_proba(x_test)), result.model.classes)
    baseline = evaluate(y_test, FrequencyBaseline().fit(y_train).predict(len(y_test)), result.model.classes)
    assert model.f1_macro > 0.85 and model.accuracy > 0.85
    assert model.f1_macro > baseline.f1_macro + 0.4


def test_best_c_comes_from_the_grid_with_validation_scores(tuned, isolated_settings):
    _, result = tuned
    assert result.best_c in isolated_settings["model"]["c_grid"]
    assert set(result.validation_scores) == set(isolated_settings["model"]["c_grid"])
    assert result.validation_scores[result.best_c] == max(result.validation_scores.values())


def test_final_model_is_refit_on_the_whole_training_set(tuned):
    split, result = tuned
    assert result.pipeline.metadata["n_train"] == len(split.train)
    assert result.n_validation > 0


def test_every_test_etf_gets_exactly_one_category(tuned):
    split, result = tuned
    decided = result.model.decide(result.model.predict_proba(result.pipeline.transform(split.test).matrix))
    assert np.all(decided.sum(axis=1) == 1)


def test_same_data_and_seed_give_the_same_model(synthetic_data, isolated_settings):
    split = make_split(synthetic_data, test_size=0.2, seed=1)
    first, second = tune(split.train, isolated_settings), tune(split.train, isolated_settings)
    x = first.pipeline.transform(split.test).matrix
    assert first.best_c == second.best_c
    assert np.allclose(first.model.predict_proba(x), second.model.predict_proba(x))
