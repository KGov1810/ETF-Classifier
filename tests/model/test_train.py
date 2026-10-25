import time
from pathlib import Path

import pytest
import yaml

from etf_classifier.data.split import make_split
from etf_classifier.model.classifier import tune
from etf_classifier.model.registry import current_version
from etf_classifier.model.train import main


def _config(tmp_path, settings) -> str:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return str(path)


def test_end_to_end_training_without_azure(sample_csv, isolated_settings, tmp_path, capsys):
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, isolated_settings), "--no-azure"])
    output = capsys.readouterr().out
    assert code == 0
    version = current_version(isolated_settings)
    assert version is not None
    assert (Path(isolated_settings["model"]["artifacts_dir"]) / version / "bundle.joblib").exists()
    report = Path(isolated_settings["model"]["reports_dir"]) / f"model_{version}.md"
    assert "## Comparison with the trivial baseline" in report.read_text(encoding="utf-8")
    assert "Macro F1" in output


def test_training_with_embeddings_reports_the_azure_cost(sample_csv, isolated_settings, fake_client, tmp_path, capsys):
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, isolated_settings), "--yes"],
                client=fake_client)
    output = capsys.readouterr().out
    assert code == 0
    assert fake_client.calls
    assert "Azure cost" in output


def test_refusing_the_embedding_cost_cancels_training(sample_csv, isolated_settings, fake_client, tmp_path):
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    code = main(["--csv", str(sample_csv), "--config", _config(tmp_path, isolated_settings)],
                client=fake_client, input_fn=lambda prompt: "n")
    assert code == 1
    assert current_version(isolated_settings) is None


def test_second_training_is_not_promoted_without_the_flag(sample_csv, isolated_settings, tmp_path):
    config = _config(tmp_path, isolated_settings)
    main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    first = current_version(isolated_settings)
    main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    assert current_version(isolated_settings) == first
    main(["--csv", str(sample_csv), "--config", config, "--no-azure", "--promote"])
    assert current_version(isolated_settings) != first


@pytest.mark.slow
def test_training_on_10000_etfs_in_less_than_5_minutes(make_synthetic, isolated_settings):
    data = make_synthetic(10_000, seed=3)
    split = make_split(data, test_size=0.2, seed=1)
    start = time.perf_counter()
    tune(split.train, isolated_settings)
    assert time.perf_counter() - start < 300
