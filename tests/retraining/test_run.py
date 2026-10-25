from pathlib import Path

import yaml

from etf_classifier.feedback.store import FeedbackStore
from etf_classifier.model.registry import current_version
from etf_classifier.retraining.history import job_lock, read_history
from etf_classifier.retraining.run import EXIT_FAILED, EXIT_OK, main


def _config(settings, tmp_path) -> str:
    path = tmp_path / "rt_settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return str(path)


def _run(settings, tmp_path, *extra, client=None) -> int:
    return main(["--config", _config(settings, tmp_path), *extra], client=client)


def _validate(settings, name="New Gold Miners"):
    FeedbackStore(settings).record_validation(
        {"name": name, "description": "Gold mining companies", "composition": {"NEM": 10.0, "GOLD": 8.0, "AEM": 7.0}},
        ["Commodities", "Gold"], "Alice Martin")


def test_first_run_without_a_current_model_promotes(rt_settings, tmp_path):
    assert _run(rt_settings, tmp_path) == EXIT_OK
    history = read_history(rt_settings)
    assert list(history["outcome"]) == ["promoted"]
    assert current_version(rt_settings) == history.iloc[0]["candidate_version"]
    report = Path(rt_settings["model"]["reports_dir"]) / f"retraining_{history.iloc[0]['candidate_version']}.md"
    assert "First model" in report.read_text(encoding="utf-8")


def test_second_run_without_change_is_skipped(rt_settings, tmp_path):
    _run(rt_settings, tmp_path)
    version = current_version(rt_settings)
    models_before = sorted(p.name for p in Path(rt_settings["model"]["artifacts_dir"]).iterdir() if p.is_dir())
    assert _run(rt_settings, tmp_path) == EXIT_OK
    history = read_history(rt_settings)
    assert list(history["outcome"]) == ["promoted", "skipped"]
    assert current_version(rt_settings) == version
    models_after = sorted(p.name for p in Path(rt_settings["model"]["artifacts_dir"]).iterdir() if p.is_dir())
    assert models_after == models_before  # nothing trained


def test_force_retrains_without_change(rt_settings, tmp_path):
    _run(rt_settings, tmp_path)
    _run(rt_settings, tmp_path, "--force")
    assert read_history(rt_settings).iloc[1]["outcome"] in {"promoted", "not_promoted"}


def test_new_validation_triggers_a_compared_candidate(rt_settings, tmp_path):
    rt_settings["retraining"]["min_comparison_etfs"] = 0
    _run(rt_settings, tmp_path)
    _validate(rt_settings)
    assert _run(rt_settings, tmp_path) == EXIT_OK
    last = read_history(rt_settings).iloc[-1]
    assert last["outcome"] in {"promoted", "not_promoted"}
    assert last["candidate_version"] != last["current_version"]
    report = Path(rt_settings["model"]["reports_dir"]) / f"retraining_{last['candidate_version']}.md"
    text = report.read_text(encoding="utf-8")
    assert "Promotion checks" in text and "Comparison macro F1" in text
    if last["outcome"] == "promoted":
        assert current_version(rt_settings) == last["candidate_version"]


def test_too_small_comparison_set_is_not_promoted(rt_settings, tmp_path):
    rt_settings["retraining"]["min_comparison_etfs"] = 1000
    _run(rt_settings, tmp_path)
    first = current_version(rt_settings)
    _validate(rt_settings)
    _run(rt_settings, tmp_path)
    last = read_history(rt_settings).iloc[-1]
    assert last["outcome"] == "not_promoted"
    assert "minimum 1000" in last["reasons"]
    assert current_version(rt_settings) == first


def test_dry_run_never_promotes_even_a_first_model(rt_settings, tmp_path):
    assert _run(rt_settings, tmp_path, "--dry-run") == EXIT_OK
    assert current_version(rt_settings) is None
    assert read_history(rt_settings).iloc[0]["outcome"] == "not_promoted"


def test_failure_is_recorded_and_returns_an_error_code(rt_settings, tmp_path):
    rt_settings["feedback"]["reference_csv"] = str(tmp_path / "missing.csv")
    assert _run(rt_settings, tmp_path) == EXIT_FAILED
    last = read_history(rt_settings).iloc[-1]
    assert last["outcome"] == "failed"
    assert "missing.csv" in last["reasons"]
    assert "missing.csv" in Path(rt_settings["retraining"]["log_path"]).read_text(encoding="utf-8")


def test_run_stops_when_another_one_is_in_progress(rt_settings, tmp_path, capsys):
    with job_lock(rt_settings):
        assert _run(rt_settings, tmp_path) == EXIT_OK
    assert "already running" in capsys.readouterr().out
    assert current_version(rt_settings) is None


def test_embeddings_are_paid_without_confirmation(rt_settings, tmp_path, fake_client):
    rt_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    assert _run(rt_settings, tmp_path, client=fake_client) == EXIT_OK
    assert fake_client.calls
    assert float(read_history(rt_settings).iloc[0]["azure_cost_usd"]) >= 0


def test_rerun_after_a_refused_candidate_is_skipped(rt_settings, tmp_path):
    rt_settings["retraining"]["min_comparison_etfs"] = 1000  # every candidate refused
    _run(rt_settings, tmp_path)
    _validate(rt_settings)
    _run(rt_settings, tmp_path)
    assert read_history(rt_settings).iloc[-1]["outcome"] == "not_promoted"
    _run(rt_settings, tmp_path)  # nothing changed since the refused candidate: no new training
    assert read_history(rt_settings).iloc[-1]["outcome"] == "skipped"
