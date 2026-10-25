import pytest

from etf_classifier.feedback.report import build_feedback_report


def _etf(name):
    return {"name": name, "description": "d", "composition": {"AAPL": 5.0}}


@pytest.fixture
def journal(store):
    """Alice : 3 acceptées + 1 corrigée ; Bob : 1 acceptée + 1 acceptée annulée ; 1 sans proposition."""
    for i in range(3):
        store.record_validation(_etf(f"A{i}"), ["Actions US"], "Alice", predicted_categories=["Actions US"])
    store.record_validation(_etf("A3"), ["Actions US", "ESG"], "Alice", predicted_categories=["Actions US", "Or"])
    store.record_validation(_etf("B0"), ["Or"], "Bob", predicted_categories=["Or"])
    cancelled = store.record_validation(_etf("B1"), ["Or"], "Bob", predicted_categories=["Or"])
    store.cancel_validation(cancelled["ValidationId"], "Bob")
    store.record_validation(_etf("C0"), ["Or"], "Carla")
    return store


def test_acceptance_rates_computed_by_hand(journal, fb_settings):
    stats = build_feedback_report(fb_settings).stats
    assert stats["n_validations"] == 6  # hors validation annulée
    assert stats["acceptance_rate"] == pytest.approx(4 / 5)
    assert stats["by_validator"]["Alice"]["acceptance_rate"] == pytest.approx(3 / 4)
    assert stats["by_validator"]["Bob"]["acceptance_rate"] == pytest.approx(1.0)
    assert stats["n_cancelled"] == 1


def test_categories_added_and_removed_by_validators(journal, fb_settings):
    stats = build_feedback_report(fb_settings).stats
    assert stats["added"] == {"ESG": 1}
    assert stats["removed"] == {"Or": 1}


def test_bias_alert_above_the_threshold(journal, fb_settings):
    fb_settings["feedback"].update({"bias_acceptance_threshold": 0.75, "bias_min_validations": 3})
    report = build_feedback_report(fb_settings)
    assert report.stats["bias_alerts"] == ["Alice", "global"]
    assert "⚠️" in report.markdown and "Alice" in report.markdown


def test_no_bias_alert_below_the_minimum_number_of_validations(journal, fb_settings):
    fb_settings["feedback"].update({"bias_acceptance_threshold": 0.5, "bias_min_validations": 100})
    assert build_feedback_report(fb_settings).stats["bias_alerts"] == []


def test_created_categories_are_listed(store, fb_settings):
    store.record_validation(_etf("H"), ["Hydrogène"], "Alice", create_categories=True)
    report = build_feedback_report(fb_settings)
    assert report.stats["created_categories"] == ["Hydrogène"]


def test_missing_journal_gives_a_clear_message(fb_settings):
    report = build_feedback_report(fb_settings)
    assert report.stats["n_validations"] == 0
    assert "Aucune validation" in report.markdown
