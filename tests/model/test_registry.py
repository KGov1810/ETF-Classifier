import json

import numpy as np
import pytest

from etf_classifier.data.split import make_split
from etf_classifier.model.classifier import tune
from etf_classifier.model.registry import NoModelError, current_version, load_bundle, promote, save_bundle


@pytest.fixture
def trained(synthetic_data, isolated_settings):
    split = make_split(synthetic_data, test_size=0.2, seed=1)
    return split, tune(split.train, isolated_settings)


def _save(result, settings):
    return save_bundle(result.model, result.pipeline, {"f1_macro": 0.9, "accuracy": 0.92}, settings, "# Fiche")


def test_reloaded_model_gives_identical_predictions(trained, isolated_settings):
    split, result = trained
    path = _save(result, isolated_settings)
    bundle = load_bundle(path.name, isolated_settings)
    x_before = result.pipeline.transform(split.test).matrix
    x_after = bundle.pipeline.transform(split.test).matrix
    assert np.array_equal(result.model.predict_proba(x_before), bundle.model.predict_proba(x_after))
    assert bundle.model.classes == result.model.classes


def test_bundle_folder_contains_metrics_and_model_card(trained, isolated_settings):
    _, result = trained
    path = _save(result, isolated_settings)
    assert json.loads((path / "metrics.json").read_text(encoding="utf-8"))["f1_macro"] == 0.9
    assert (path / "MODEL_CARD.md").read_text(encoding="utf-8") == "# Fiche"


def test_first_model_becomes_current_and_later_ones_only_when_promoted(trained, isolated_settings):
    _, result = trained
    first = _save(result, isolated_settings)
    second = _save(result, isolated_settings)
    assert current_version(isolated_settings) == first.name
    promote(second.name, isolated_settings)
    assert current_version(isolated_settings) == second.name
    assert load_bundle("current", isolated_settings).version == second.name


def test_two_saves_in_the_same_second_get_distinct_versions(trained, isolated_settings):
    _, result = trained
    assert _save(result, isolated_settings).name != _save(result, isolated_settings).name


def test_loading_current_without_any_model_gives_a_clear_error(isolated_settings):
    with pytest.raises(NoModelError, match="No current model"):
        load_bundle("current", isolated_settings)


def test_promoting_an_unknown_version_fails(isolated_settings):
    with pytest.raises(NoModelError):
        promote("20990101-000000", isolated_settings)
