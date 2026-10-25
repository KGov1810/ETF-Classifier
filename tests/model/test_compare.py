from pathlib import Path

import yaml

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.model.compare import main, run_comparison


def _config(tmp_path, settings) -> str:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_every_experiment_uses_the_same_test_set(sample_csv, isolated_settings):
    prepared = prepare_dataset(sample_csv, isolated_settings)
    results = run_comparison(prepared, isolated_settings, ["local", "text_only"], embedder=None)
    fingerprints = {r.test_fingerprint for r in results}
    assert len(fingerprints) == 1


def test_table_contains_the_baseline_and_each_experiment(sample_csv, isolated_settings, tmp_path, capsys):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, isolated_settings),
                 "--experiments", "local,text_only"])
    report = (Path(isolated_settings["model"]["reports_dir"]) / "model_comparison.md").read_text(encoding="utf-8")
    assert code == 0
    for label in ["Trivial baseline", "| local |", "| text_only |"]:
        assert label in report


def test_unknown_experiment_lists_the_available_ones(sample_csv, isolated_settings, tmp_path, capsys):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, isolated_settings), "--experiments", "magic"])
    assert code == 2
    assert "local+embeddings" in capsys.readouterr().err


def test_embedding_experiment_is_skipped_without_azure(sample_csv, isolated_settings, monkeypatch):
    for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]:
        monkeypatch.delenv(name, raising=False)
    prepared = prepare_dataset(sample_csv, isolated_settings)
    results = run_comparison(prepared, isolated_settings, ["local", "local+embeddings"], embedder=None)
    skipped = next(r for r in results if r.name == "local+embeddings")
    assert skipped.skipped_reason and "Azure" in skipped.skipped_reason


def test_embedding_experiment_reports_its_cost(sample_csv, isolated_settings, fake_client, tmp_path):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, isolated_settings),
                 "--experiments", "local,local+embeddings", "--yes"], client=fake_client)
    report = (Path(isolated_settings["model"]["reports_dir"]) / "model_comparison.md").read_text(encoding="utf-8")
    assert code == 0
    assert fake_client.calls
    row = next(line for line in report.splitlines() if line.startswith("| local+embeddings |"))
    assert "$" in row
