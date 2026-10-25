import pytest

from etf_classifier.retraining.history import HISTORY_COLUMNS, RetrainingBusy, append_run, job_lock, read_history


def test_second_run_cannot_take_the_lock(rt_settings):
    with job_lock(rt_settings):
        with pytest.raises(RetrainingBusy, match="already running"):
            with job_lock(rt_settings):
                pass


def test_lock_is_released_after_the_run(rt_settings):
    with job_lock(rt_settings):
        pass
    with job_lock(rt_settings):
        pass


def test_runs_are_appended_and_read_back(rt_settings):
    append_run(rt_settings, {"started_at": "2026-10-25T10:00:00Z", "outcome": "skipped", "reasons": "No change"})
    append_run(rt_settings, {"started_at": "2026-10-26T10:00:00Z", "outcome": "promoted",
                             "candidate_version": "v2", "reasons": "a; b"})
    history = read_history(rt_settings)
    assert list(history.columns) == HISTORY_COLUMNS
    assert list(history["outcome"]) == ["skipped", "promoted"]
    assert history.iloc[1]["reasons"] == "a; b"


def test_missing_history_is_an_empty_table(rt_settings):
    history = read_history(rt_settings)
    assert history.empty and list(history.columns) == HISTORY_COLUMNS
