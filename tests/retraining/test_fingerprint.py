import json

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.features.pipeline import dataset_fingerprint
from etf_classifier.feedback.store import FeedbackStore
from etf_classifier.model.registry import current_version, load_bundle
from etf_classifier.model.train import train_and_save


def test_trained_model_saves_its_training_tickers_and_the_dataset_fingerprint(isolated_settings, sample_csv):
    prepared = prepare_dataset(sample_csv, isolated_settings)
    trained = train_and_save(prepared, isolated_settings)
    tickers = json.loads((trained.path / "train_tickers.json").read_text(encoding="utf-8"))
    assert sorted(tickers) == sorted(prepared.split.train["ticker"])
    metrics = json.loads((trained.path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset_fingerprint"] == dataset_fingerprint(prepared.data)


def test_first_model_becomes_current_unless_asked_otherwise(isolated_settings, sample_csv):
    prepared = prepare_dataset(sample_csv, isolated_settings)
    train_and_save(prepared, isolated_settings, make_current_if_first=False)
    assert current_version(isolated_settings) is None
    trained = train_and_save(prepared, isolated_settings)
    assert current_version(isolated_settings) == trained.version
    assert load_bundle("current", isolated_settings).version == trained.version


def test_fingerprint_is_stable_and_changes_with_validations(isolated_settings, sample_csv):
    first = dataset_fingerprint(prepare_dataset(sample_csv, isolated_settings).data)
    assert dataset_fingerprint(prepare_dataset(sample_csv, isolated_settings).data) == first
    store = FeedbackStore(isolated_settings)
    record = store.record_validation({"ticker": "NGLD", "name": "New Gold", "description": "d",
                                      "composition": {"NEM": 5.0}}, "Gold", "Alice Martin")
    with_validation = dataset_fingerprint(prepare_dataset(sample_csv, isolated_settings).data)
    assert with_validation != first
    store.cancel_validation(record["ValidationId"], "Alice Martin", "mistake")
    assert dataset_fingerprint(prepare_dataset(sample_csv, isolated_settings).data) == first
