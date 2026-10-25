import pytest

from etf_classifier.feedback.categories import CategoryRegistry
from etf_classifier.feedback.checks import ValidationError


@pytest.mark.parametrize("change, message", [
    ({"name": "  "}, "name"),
    ({"composition": "[{'AAPL': 5.0}, {'MSFT'"}, "Composition"),
])
def test_invalid_etf_is_refused_with_a_clear_message(store, new_etf, change, message):
    with pytest.raises(ValidationError, match=message):
        store.record_validation({**new_etf, **change}, ["Clean Energy"], validated_by="Kevin")


def test_at_least_one_category_is_required(store, new_etf):
    with pytest.raises(ValidationError, match="category"):
        store.record_validation(new_etf, [], validated_by="Kevin")


def test_validator_identity_is_required(store, new_etf):
    with pytest.raises(ValidationError, match="validator"):
        store.record_validation(new_etf, ["Clean Energy"], validated_by=" ")


def test_unknown_category_is_refused_without_explicit_creation(store, new_etf):
    with pytest.raises(ValidationError, match="Hydrogen"):
        store.record_validation(new_etf, ["Hydrogen"], validated_by="Kevin")


def test_created_category_is_registered_with_author_and_date(store, new_etf, fb_settings):
    store.record_validation(new_etf, ["Hydrogen"], validated_by="Kevin", create_categories=True)
    registry = CategoryRegistry(fb_settings).read()
    row = registry.set_index("category").loc["Hydrogen"]
    assert row["created_by"] == "Kevin" and row["source"] == "created" and row["created_at"]


def test_registry_starts_with_the_historical_categories(store, new_etf, fb_settings):
    store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    registry = CategoryRegistry(fb_settings).read()
    historical = set(registry.loc[registry["source"] == "historical", "category"])
    assert {"Global Equity", "Gold", "Clean Energy"} <= historical


def test_etf_already_in_the_reference_is_refused_by_default(store, new_etf):
    with pytest.raises(ValidationError, match="already present"):
        store.record_validation({**new_etf, "name": "Gold Miners"}, ["Gold"], validated_by="Kevin")


def test_etf_already_validated_is_refused_by_default(store, new_etf):
    store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    with pytest.raises(ValidationError, match="already present"):
        store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")


def test_duplicate_name_can_be_confirmed_as_a_different_etf(store, new_etf):
    record = store.record_validation({**new_etf, "name": "Gold Miners"}, ["Gold"], validated_by="Kevin",
                                     allow_duplicate_name=True)
    assert record["Name"] == "Gold Miners"


@pytest.mark.parametrize("predicted, validated, decision", [
    (["Clean Energy", "Global Equity"], ["Global Equity", "Clean Energy"], "accepted"),
    (["Global Equity"], ["Global Equity", "Clean Energy"], "corrected"),
    (None, ["Clean Energy"], "no_proposal"),
])
def test_decision_is_deduced_from_the_proposal(store, new_etf, predicted, validated, decision):
    record = store.record_validation(new_etf, validated, validated_by="Kevin", predicted_categories=predicted,
                                     model_version="20261025-120000")
    assert record["Decision"] == decision
