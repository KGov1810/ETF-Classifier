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
        "Ticker;Nom;Description;Composition\n"
        "NWLD;New World;Tracks the MSCI World index;\"[{'AAPL': 5.0}, {'MSFT': 4.4}, {'NVDA': 4.0}]\"\n"
        "NGLD;New Gold;Gold miners worldwide;\"[{'NEM': 12.0}, {'GOLD': 10.0}, {'AEM': 9.0}]\"\n",
        encoding="utf-8",
    )
    return path


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_predictions_are_written_for_new_etfs(sample_csv, new_etfs_csv, isolated_settings, tmp_path, capsys):
    config = _config(tmp_path, isolated_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    output = tmp_path / "predictions.csv"
    code = predict_main(["--csv", str(new_etfs_csv), "--config", config, "--output", str(output)])
    rows = _rows(output)
    assert code == 0
    assert [(r["Ticker"], r["Nom"]) for r in rows] == [("NWLD", "New World"), ("NGLD", "New Gold")]
    for row in rows:
        assert row["PredictedCategory"]
        assert len(ast.literal_eval(row["Scores"])) == 3
        assert row["Uncertain"] in {"True", "False"}
        assert row["Classification"] == row["Check"] == row["EstablishedScore"] == ""
    assert "2 ETF" in capsys.readouterr().out


@pytest.fixture
def established_csv(tmp_path):
    path = tmp_path / "established.csv"
    path.write_text(
        "Ticker;Nom;Description;Classification;Composition\n"
        "AGRE;Gold Agree;Gold miners worldwide;Gold;\"[{'NEM': 12.0}, {'GOLD': 10.0}, {'AEM': 9.0}]\"\n"
        "DISA;Gold Disagree;Gold miners worldwide;Technology;\"[{'NEM': 12.0}, {'GOLD': 10.0}, {'AEM': 9.0}]\"\n"
        "UNKN;Gold Unknown;Gold miners worldwide;Crypto;\"[{'NEM': 12.0}, {'GOLD': 10.0}, {'AEM': 9.0}]\"\n"
        "NONE;Gold None;Gold miners worldwide;;\"[{'NEM': 12.0}, {'GOLD': 10.0}, {'AEM': 9.0}]\"\n",
        encoding="utf-8",
    )
    return path


def test_established_classifications_are_checked(sample_csv, established_csv, isolated_settings, tmp_path, capsys):
    config = _config(tmp_path, isolated_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    output = tmp_path / "checked.csv"
    assert predict_main(["--csv", str(established_csv), "--config", config, "--output", str(output)]) == 0
    rows = {r["Ticker"]: r for r in _rows(output)}
    assert {t: r["PredictedCategory"] for t, r in rows.items()} == dict.fromkeys(rows, "Gold")
    assert {t: r["Check"] for t, r in rows.items()} == {"AGRE": "agreement", "DISA": "disagreement",
                                                         "UNKN": "unknown", "NONE": ""}
    assert float(rows["DISA"]["EstablishedScore"]) < float(rows["AGRE"]["EstablishedScore"])
    assert "1 agreement · 1 disagreement · 1 unknown" in capsys.readouterr().out


def test_output_can_be_imported_back_as_validations(sample_csv, established_csv, isolated_settings, tmp_path):
    from etf_classifier.feedback.importer import main as import_main
    from etf_classifier.feedback.store import FeedbackStore

    config = _config(tmp_path, isolated_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    output = tmp_path / "checked.csv"
    predict_main(["--csv", str(established_csv), "--config", config, "--output", str(output)])
    assert import_main(["--csv", str(output), "--by", "Kevin", "--config", config]) == 0
    journal = FeedbackStore(isolated_settings).read_journal().set_index("Ticker")
    assert journal.loc["AGRE", "Decision"] == "accepted" and journal.loc["DISA", "Decision"] == "corrected"


def test_without_any_model_a_clear_message_is_shown(new_etfs_csv, isolated_settings, tmp_path, capsys):
    code = predict_main(["--csv", str(new_etfs_csv), "--config", _config(tmp_path, isolated_settings)])
    assert code == 2
    assert "No current model" in capsys.readouterr().err


def test_model_with_embeddings_requires_azure(sample_csv, new_etfs_csv, isolated_settings, fake_client, tmp_path,
                                               capsys, monkeypatch):
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    config = _config(tmp_path, isolated_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--yes"], client=fake_client)
    for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]:
        monkeypatch.delenv(name, raising=False)
    assert predict_main(["--csv", str(new_etfs_csv), "--config", config]) == 3
    assert "Azure" in capsys.readouterr().err


def test_model_with_embeddings_predicts_with_azure(sample_csv, new_etfs_csv, isolated_settings, fake_client, tmp_path):
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    config = _config(tmp_path, isolated_settings)
    train_main(["--csv", str(sample_csv), "--config", config, "--yes"], client=fake_client)
    output = tmp_path / "predictions.csv"
    code = predict_main(["--csv", str(new_etfs_csv), "--config", config, "--output", str(output), "--yes"],
                        client=fake_client)
    assert code == 0
    assert len(_rows(output)) == 2
