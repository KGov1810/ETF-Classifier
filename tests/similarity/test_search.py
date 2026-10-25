import csv

import pytest
import yaml

from etf_classifier.model.registry import load_bundle
from etf_classifier.model.train import main as train_main
from etf_classifier.similarity.search import main as search_main


@pytest.fixture
def config(sim_settings, tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(sim_settings, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture
def new_etfs_csv(tmp_path):
    path = tmp_path / "new_etfs.csv"
    path.write_text(
        "Name;Description;Composition\n"
        "World Clone;Tracks the MSCI World index;"
        "\"[{'AAPL': 5.0}, {'MSFT': 4.5}, {'NVDA': 4.0}, {'AMZN': 3.0}, {'GOOGL': 2.5}]\"\n"
        "New Gold;Gold miners worldwide;\"[{'NEM': 11.0}, {'GOLD': 9.5}]\"\n",
        encoding="utf-8",
    )
    return path


def _rows(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_k_neighbors_are_written_for_each_new_etf(sample_csv, new_etfs_csv, config, tmp_path):
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    output = tmp_path / "neighbors.csv"
    assert search_main(["--csv", str(new_etfs_csv), "--config", config, "--k", "3", "--output", str(output)]) == 0
    rows = _rows(output)
    assert len(rows) == 6
    assert rows[0]["Rank"] == "1"
    first_gold = next(r for r in rows if r["Name"] == "New Gold" and r["Rank"] == "1")
    assert first_gold["NeighborName"] == "Gold Miners"
    assert "NEM" in first_gold["SharedHoldings"]


def test_near_duplicate_is_reported_in_the_summary(sample_csv, new_etfs_csv, config, tmp_path):
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    output = tmp_path / "neighbors.csv"
    search_main(["--csv", str(new_etfs_csv), "--config", config, "--output", str(output)])
    summary = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "Quasi-doublon" in summary and "World Tracker" in summary


def test_without_any_model_a_clear_message_is_shown(new_etfs_csv, config, capsys):
    assert search_main(["--csv", str(new_etfs_csv), "--config", config]) == 2
    assert "Aucun modèle" in capsys.readouterr().err


def test_model_without_index_asks_to_retrain(sample_csv, new_etfs_csv, config, sim_settings, capsys):
    train_main(["--csv", str(sample_csv), "--config", config, "--no-azure"])
    (load_bundle("current", sim_settings).path / "similarity.joblib").unlink()
    assert search_main(["--csv", str(new_etfs_csv), "--config", config]) == 4
    assert "réentraîner" in capsys.readouterr().err
