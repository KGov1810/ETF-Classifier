import pytest

from etf_classifier.feedback.categories import CategoryRegistry
from etf_classifier.feedback.checks import ValidationError


@pytest.mark.parametrize("change, message", [
    ({"name": "  "}, "name"),
    ({"ticker": " "}, "ticker"),
    ({"composition": "[{'AAPL': 5.0}, {'MSFT'"}, "Composition"),
    ({"description": "", "composition": {}}, "description or a composition"),
])
def test_invalid_etf_is_refused_with_a_clear_message(store, new_etf, change, message):
    with pytest.raises(ValidationError, match=message):
        store.record_validation({**new_etf, **change}, "Clean Energy", validated_by="Kevin")


def test_description_alone_is_accepted(store, new_etf):
    record = store.record_validation({**new_etf, "composition": {}}, "Clean Energy", validated_by="Kevin")
    assert record["Composition"] == "[]"


def test_a_category_is_required(store, new_etf):
    with pytest.raises(ValidationError, match="category"):
        store.record_validation(new_etf, " ", validated_by="Kevin")


def test_journal_holds_the_normalised_ticker_and_one_category(store, new_etf):
    record = store.record_validation({**new_etf, "ticker": " ncle "}, "Clean Energy", validated_by="Kevin",
                                     predicted_category="Global Equity")
    assert (record["Ticker"], record["Category"], record["PredictedCategory"]) == \
        ("NCLE", "Clean Energy", "Global Equity")


def test_validator_identity_is_required(store, new_etf):
    with pytest.raises(ValidationError, match="validator"):
        store.record_validation(new_etf, "Clean Energy", validated_by=" ")


def test_unknown_category_is_refused_without_explicit_creation(store, new_etf):
    with pytest.raises(ValidationError, match="Hydrogen"):
        store.record_validation(new_etf, "Hydrogen", validated_by="Kevin")


def test_created_category_is_registered_with_author_and_description(store, new_etf, isolated_settings):
    store.record_validation(new_etf, "Hydrogen", validated_by="Kevin", create_category=True,
                            category_description="Hydrogen value chain")
    row = CategoryRegistry(isolated_settings).read().set_index("category").loc["Hydrogen"]
    assert row["created_by"] == "Kevin" and row["source"] == "created" and row["created_at"]
    assert row["description"] == "Hydrogen value chain"


def test_registry_starts_with_the_historical_categories(store, new_etf, isolated_settings):
    store.record_validation(new_etf, "Clean Energy", validated_by="Kevin")
    registry = CategoryRegistry(isolated_settings).read()
    historical = set(registry.loc[registry["source"] == "historical", "category"])
    assert {"Global Equity", "Gold", "Clean Energy"} <= historical


def test_known_ticker_can_be_validated_again(store, new_etf):
    record = store.record_validation({**new_etf, "ticker": "GDX", "name": "Gold Miners"}, "Gold",
                                     validated_by="Kevin")
    again = store.record_validation({**new_etf, "ticker": "GDX", "name": "Gold Miners"}, "Gold",
                                    validated_by="Kevin")
    assert record["Ticker"] == again["Ticker"] == "GDX"


@pytest.mark.parametrize("predicted, validated, decision", [
    ("Clean Energy", "Clean Energy", "accepted"),
    ("Global Equity", "Clean Energy", "corrected"),
    (None, "Clean Energy", "no_proposal"),
])
def test_decision_is_deduced_from_the_proposal(store, new_etf, predicted, validated, decision):
    record = store.record_validation(new_etf, validated, validated_by="Kevin", predicted_category=predicted,
                                     model_version="20261025-120000")
    assert record["Decision"] == decision


def test_reference_csv_is_read_once_until_it_changes(isolated_settings, tmp_path, sample_csv):
    import shutil

    from etf_classifier.feedback.store import FeedbackStore, _read_reference

    reference = tmp_path / "reference.csv"
    shutil.copy(sample_csv, reference)
    isolated_settings["feedback"]["reference_csv"] = str(reference)
    _read_reference.cache_clear()
    FeedbackStore(isolated_settings).known_categories()
    FeedbackStore(isolated_settings).known_categories()  # a new store (as in every app rerun): no new read
    assert _read_reference.cache_info().misses == 1
    with reference.open("a", encoding="utf-8") as handle:
        handle.write('RKT;Space ETF;Space companies;Space;"[{\'RKLB\': 5.0}]"\n')
    assert "Space" in FeedbackStore(isolated_settings).known_categories()
    assert _read_reference.cache_info().misses == 2
