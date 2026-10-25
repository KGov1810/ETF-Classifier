from datetime import date, timedelta

import pandas as pd
import pytest

from etf_classifier.app import services
from etf_classifier.feedback.categories import CategoryRegistry
from etf_classifier.feedback.store import FeedbackStore


def _proposal(settings, name="New Gold", holdings=(("NEM", 11.0), ("GOLD", 9.0), ("AEM", 8.0))):
    bundle = services.load_current_model(settings)
    checked = services.etf_from_inputs(name, "Global gold miners",
                                       pd.DataFrame(list(holdings), columns=["Ticker", "Weight (%)"]), settings)
    return services.propose(checked.data, bundle, settings)[0]


def _journal(settings) -> pd.DataFrame:
    return FeedbackStore(settings).read_journal()


def test_validation_is_recorded_with_validator_model_version_and_proposal(trained_settings):
    proposal = _proposal(trained_settings)
    record = services.validate(proposal, proposal.categories, "Alice Martin", trained_settings)
    journal = _journal(trained_settings)
    assert len(journal) == 1
    row = journal.iloc[0]
    assert row["ValidatedBy"] == "Alice Martin"
    assert row["ModelVersion"] == proposal.model_version
    assert row["PredictedCategories"] == str(proposal.categories)
    assert row["Decision"] == record["Decision"] == "accepted"


def test_changed_categories_give_a_corrected_decision(trained_settings):
    proposal = _proposal(trained_settings)
    other = "ESG" if "ESG" not in proposal.categories else "Dividend"
    record = services.validate(proposal, proposal.categories + [other], "Bruno Leroy", trained_settings)
    assert record["Decision"] == "corrected"


def test_no_validator_is_refused_without_writing(trained_settings):
    proposal = _proposal(trained_settings)
    with pytest.raises(services.AppError, match="validator"):
        services.validate(proposal, proposal.categories, "", trained_settings)
    assert _journal(trained_settings).empty


def test_unknown_category_without_creation_is_refused_with_the_reason(trained_settings):
    proposal = _proposal(trained_settings)
    with pytest.raises(services.AppError, match="Unknown category"):
        services.validate(proposal, ["Space"], "Alice Martin", trained_settings)
    assert _journal(trained_settings).empty


def test_new_category_is_created_with_its_description(trained_settings):
    proposal = _proposal(trained_settings)
    services.validate(proposal, ["Space"], "Alice Martin", trained_settings,
                      new_categories={"Space": "Space exploration and satellites"})
    created = CategoryRegistry(trained_settings).created()
    assert created.iloc[0]["category"] == "Space"
    assert created.iloc[0]["description"] == "Space exploration and satellites"
    assert created.iloc[0]["created_by"] == "Alice Martin"


def test_second_validation_of_the_same_etf_is_refused(trained_settings):
    proposal = _proposal(trained_settings)
    services.validate(proposal, proposal.categories, "Alice Martin", trained_settings)
    with pytest.raises(services.AppError, match="already"):
        services.validate(proposal, proposal.categories, "Alice Martin", trained_settings)
    assert len(_journal(trained_settings)) == 1


def test_locked_journal_gives_a_readable_message(trained_settings):
    from filelock import FileLock

    proposal = _proposal(trained_settings)
    store = FeedbackStore(trained_settings)
    store.journal_path.parent.mkdir(parents=True, exist_ok=True)
    trained_settings["feedback"]["lock_timeout_seconds"] = 0.1
    with FileLock(str(store.journal_path) + ".lock"):
        with pytest.raises(services.AppError, match="try again"):
            services.validate(proposal, proposal.categories, "Alice Martin", trained_settings)


def test_known_categories_include_the_historical_ones(trained_settings):
    categories = services.known_categories(trained_settings)
    assert {"Gold", "Global Equity", "ESG"} <= set(categories)
    assert categories == sorted(categories)


def test_cancellation_requires_a_comment(trained_settings):
    proposal = _proposal(trained_settings)
    record = services.validate(proposal, proposal.categories, "Alice Martin", trained_settings)
    with pytest.raises(services.AppError, match="comment"):
        services.cancel(record["ValidationId"], "Alice Martin", "  ", trained_settings)
    services.cancel(record["ValidationId"], "Alice Martin", "Wrong ETF", trained_settings)
    assert list(_journal(trained_settings)["Decision"]) == ["accepted", "cancellation"]


def test_history_flags_cancelled_validations_and_hides_cancellation_lines(trained_settings):
    first = services.validate(_proposal(trained_settings, "Gold A"), ["Gold"], "Alice Martin", trained_settings)
    services.validate(_proposal(trained_settings, "Gold B"), ["Gold"], "Bruno Leroy", trained_settings)
    services.cancel(first["ValidationId"], "Alice Martin", "Duplicate", trained_settings)
    history = services.history(trained_settings)
    assert list(history["Name"]) == ["Gold B", "Gold A"]  # newest first
    assert list(history["Cancelled"]) == [False, True]
    assert history.loc[history["Name"] == "Gold A", "CancellationComment"].iloc[0] == "Duplicate"


def test_history_filters(trained_settings):
    services.validate(_proposal(trained_settings, "Gold A"), ["Gold"], "Alice Martin", trained_settings)
    services.validate(_proposal(trained_settings, "Gold B"), ["Gold", "ESG"], "Bruno Leroy", trained_settings)
    assert list(services.history(trained_settings, validator="Bruno Leroy")["Name"]) == ["Gold B"]
    assert list(services.history(trained_settings, name="gold a")["Name"]) == ["Gold A"]
    tomorrow = date.today() + timedelta(days=1)
    assert services.history(trained_settings, since=tomorrow).empty


def test_history_of_an_empty_journal_is_empty(app_settings):
    assert services.history(app_settings).empty


def test_monitoring_reports_match_the_cli_reports(trained_settings):
    from etf_classifier.costs.report import build_cost_report
    from etf_classifier.costs.pricing import load_pricing
    from etf_classifier.config import resolve_path
    from etf_classifier.feedback.report import build_feedback_report

    services.validate(_proposal(trained_settings, "Gold A"), ["Gold"], "Alice Martin", trained_settings)
    feedback, costs = services.monitoring(trained_settings)
    assert feedback.stats == build_feedback_report(trained_settings).stats
    cfg = trained_settings["costs"]
    expected = build_cost_report(resolve_path(cfg["usage_log_path"]), load_pricing(resolve_path(cfg["pricing_path"])),
                                 stale_after_days=cfg["stale_after_days"])
    assert costs.stats == expected.stats


def test_monitoring_without_journal_or_cost_log(app_settings):
    feedback, costs = services.monitoring(app_settings)
    assert feedback.stats["n_validations"] == 0
    assert "No Azure call" in costs.markdown


def test_retraining_runs_newest_first(app_settings):
    from etf_classifier.retraining.history import append_run

    append_run(app_settings, {"started_at": "2026-10-25T10:00:00Z", "outcome": "promoted", "candidate_version": "v1"})
    append_run(app_settings, {"started_at": "2026-10-26T10:00:00Z", "outcome": "skipped", "reasons": "No change"})
    runs = services.retraining_runs(app_settings)
    assert list(runs["outcome"]) == ["skipped", "promoted"]


def test_no_retraining_run_yet(app_settings):
    assert services.retraining_runs(app_settings).empty
