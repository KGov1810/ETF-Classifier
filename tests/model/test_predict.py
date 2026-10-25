import ast
import csv

import pytest
import yaml

from etf_classifier.model.predict import main as predict_main
from etf_classifier.model.train import main as train_main


def _config(tmp_path, settings) -> str:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture
def new_etfs_csv(tmp_path):
    path = tmp_path / "new_etfs.csv"
    path.write_text(
        "Name;Description;Composition\n"
        "New World;Tracks the MSCI World index;\"[{'AAPL': 5.0}, {'MSFT': 4.4}, {'NVDA': 4.0}]\"\n"
        "New Gold;Gold miners worldwide;\"[{'NEM': 11.0}, {'GOLD': 9.5}]\"\n",
        encoding="utf-8",
    )
    return path


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_predictions_are_written_for_new_etfs(sample_csv, new_etfs_csv, model_settings, tmp_path, capsys):
    config = _config(tmp_path, model_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    output = tmp_path / "predictions.csv"
    code = predict_main(["--csv", str(new_etfs_csv), "--config", config, "--output", str(output)])
    rows = _rows(output)
    assert code == 0
    assert [r["Name"] for r in rows] == ["New World", "New Gold"]
    for row in rows:
        assert 1 <= len(ast.literal_eval(row["PredictedCategories"])) <= 5
        assert len(ast.literal_eval(row["Scores"])) <= 5
        assert row["Uncertain"] in {"True", "False"}
    assert "2 ETF" in capsys.readouterr().out


def test_without_any_model_a_clear_message_is_shown(new_etfs_csv, model_settings, tmp_path, capsys):
    code = predict_main(["--csv", str(new_etfs_csv), "--config", _config(tmp_path, model_settings)])
    assert code == 2
    assert "No current model" in capsys.readouterr().err


def test_model_with_embeddings_requires_azure(sample_csv, new_etfs_csv, model_settings, fake_client, tmp_path,
                                               capsys, monkeypatch):
    model_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    config = _config(tmp_path, model_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--yes"], client=fake_client)
    for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]:
        monkeypatch.delenv(name, raising=False)
    assert predict_main(["--csv", str(new_etfs_csv), "--config", config]) == 3
    assert "Azure" in capsys.readouterr().err


def test_model_with_embeddings_predicts_with_azure(sample_csv, new_etfs_csv, model_settings, fake_client, tmp_path):
    model_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    config = _config(tmp_path, model_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--yes"], client=fake_client)
    output = tmp_path / "predictions.csv"
    code = predict_main(["--csv", str(new_etfs_csv), "--config", config, "--output", str(output), "--yes"],
                        client=fake_client)
    assert code == 0
    assert len(_rows(output)) == 2
