from pathlib import Path

import yaml

from etf_classifier.feedback.store import FeedbackStore
from etf_classifier.model.registry import current_version
from etf_classifier.retraining.history import job_lock, read_history
from etf_classifier.retraining.run import EXIT_FAILED, EXIT_OK, main


def _config(settings, tmp_path) -> str:
    path = tmp_path / "isolated_settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return str(path)


def _run(settings, tmp_path, *extra, client=None) -> int:
    return main(["--config", _config(settings, tmp_path), *extra], client=client)


def _validate(settings, name="New Gold Miners"):
    FeedbackStore(settings).record_validation(
        {"name": name, "description": "Gold mining companies", "composition": {"NEM": 10.0, "GOLD": 8.0, "AEM": 7.0}},
        ["Commodities", "Gold"], "Alice Martin")


def test_first_run_without_a_current_model_promotes(isolated_settings, tmp_path):
    assert _run(isolated_settings, tmp_path) == EXIT_OK
    history = read_history(isolated_settings)
    assert list(history["outcome"]) == ["promoted"]
    assert current_version(isolated_settings) == history.iloc[0]["candidate_version"]
    report = Path(isolated_settings["model"]["reports_dir"]) / f"retraining_{history.iloc[0]['candidate_version']}.md"
    assert "First model" in report.read_text(encoding="utf-8")


def test_second_run_without_change_is_skipped(isolated_settings, tmp_path):
    _run(isolated_settings, tmp_path)
    version = current_version(isolated_settings)
    models_before = sorted(p.name for p in Path(isolated_settings["model"]["artifacts_dir"]).iterdir() if p.is_dir())
    assert _run(isolated_settings, tmp_path) == EXIT_OK
    history = read_history(isolated_settings)
    assert list(history["outcome"]) == ["promoted", "skipped"]
    assert current_version(isolated_settings) == version
    models_after = sorted(p.name for p in Path(isolated_settings["model"]["artifacts_dir"]).iterdir() if p.is_dir())
    assert models_after == models_before  # nothing trained


def test_force_retrains_without_change(isolated_settings, tmp_path):
    _run(isolated_settings, tmp_path)
    _run(isolated_settings, tmp_path, "--force")
    assert read_history(isolated_settings).iloc[1]["outcome"] in {"promoted", "not_promoted"}


def test_new_validation_triggers_a_compared_candidate(isolated_settings, tmp_path):
    isolated_settings["retraining"]["min_comparison_etfs"] = 0
    _run(isolated_settings, tmp_path)
    _validate(isolated_settings)
    assert _run(isolated_settings, tmp_path) == EXIT_OK
    last = read_history(isolated_settings).iloc[-1]
    assert last["outcome"] in {"promoted", "not_promoted"}
    assert last["candidate_version"] != last["current_version"]
    report = Path(isolated_settings["model"]["reports_dir"]) / f"retraining_{last['candidate_version']}.md"
    text = report.read_text(encoding="utf-8")
    assert "Promotion checks" in text and "Comparison macro F1" in text
    if last["outcome"] == "promoted":
        assert current_version(isolated_settings) == last["candidate_version"]


def test_too_small_comparison_set_is_not_promoted(isolated_settings, tmp_path):
    isolated_settings["retraining"]["min_comparison_etfs"] = 1000
    _run(isolated_settings, tmp_path)
    first = current_version(isolated_settings)
    _validate(isolated_settings)
    _run(isolated_settings, tmp_path)
    last = read_history(isolated_settings).iloc[-1]
    assert last["outcome"] == "not_promoted"
    assert "minimum 1000" in last["reasons"]
    assert current_version(isolated_settings) == first


def test_dry_run_never_promotes_even_a_first_model(isolated_settings, tmp_path):
    assert _run(isolated_settings, tmp_path, "--dry-run") == EXIT_OK
    assert current_version(isolated_settings) is None
    assert read_history(isolated_settings).iloc[0]["outcome"] == "not_promoted"


def test_failure_is_recorded_and_returns_an_error_code(isolated_settings, tmp_path):
    isolated_settings["feedback"]["reference_csv"] = str(tmp_path / "missing.csv")
    assert _run(isolated_settings, tmp_path) == EXIT_FAILED
    last = read_history(isolated_settings).iloc[-1]
    assert last["outcome"] == "failed"
    assert "missing.csv" in last["reasons"]
    assert "missing.csv" in Path(isolated_settings["retraining"]["log_path"]).read_text(encoding="utf-8")


def test_run_stops_when_another_one_is_in_progress(isolated_settings, tmp_path, capsys):
    with job_lock(isolated_settings):
        assert _run(isolated_settings, tmp_path) == EXIT_OK
    assert "already running" in capsys.readouterr().out
    assert current_version(isolated_settings) is None


def test_embeddings_are_paid_without_confirmation(isolated_settings, tmp_path, fake_client):
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    assert _run(isolated_settings, tmp_path, client=fake_client) == EXIT_OK
    assert fake_client.calls
    assert float(read_history(isolated_settings).iloc[0]["azure_cost_usd"]) >= 0


def test_rerun_after_a_refused_candidate_is_skipped(isolated_settings, tmp_path):
    isolated_settings["retraining"]["min_comparison_etfs"] = 1000  # every candidate refused
    _run(isolated_settings, tmp_path)
    _validate(isolated_settings)
    _run(isolated_settings, tmp_path)
    assert read_history(isolated_settings).iloc[-1]["outcome"] == "not_promoted"
    _run(isolated_settings, tmp_path)  # nothing changed since the refused candidate: no new training
    assert read_history(isolated_settings).iloc[-1]["outcome"] == "skipped"
