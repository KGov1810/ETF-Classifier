"""Fixtures for the app tests: isolated configuration, model trained on the fixture, fake Azure client."""

import copy
import importlib.util
from pathlib import Path

import pytest

TESTS = Path(__file__).parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fake_azure = _load("fake_azure_app", TESTS / "features" / "fake_azure.py")


@pytest.fixture
def app_settings(settings, tmp_path, sample_csv):
    """Every file (journal, registry, artifacts, reports, cache, cost log) in a temporary folder."""
    cfg = copy.deepcopy(settings)
    cfg["feedback"].update({
        "reference_csv": str(sample_csv),
        "journal_path": str(tmp_path / "data" / "validations.csv"),
        "categories_path": str(tmp_path / "data" / "categories.csv"),
        "backups_dir": str(tmp_path / "data" / "backups"),
        "lock_timeout_seconds": 2,
    })
    cfg["features"]["blocks"]["embeddings"]["enabled"] = False
    cfg["features"]["text_tfidf"]["min_df"] = 1
    cfg["features"]["embeddings"].update({"dimensions": 8, "cache_path": str(tmp_path / "cache" / "emb.sqlite")})
    cfg["features"]["artifacts_dir"] = str(tmp_path / "artifacts" / "features")
    cfg["model"]["artifacts_dir"] = str(tmp_path / "artifacts" / "models")
    cfg["model"]["reports_dir"] = str(tmp_path / "reports")
    cfg["costs"]["usage_log_path"] = str(tmp_path / "logs" / "usage.csv")
    cfg["retraining"].update({"history_path": str(tmp_path / "artifacts" / "models" / "retraining_history.csv"),
                              "log_path": str(tmp_path / "logs" / "retraining.log")})
    return cfg


@pytest.fixture
def fake_client():
    return fake_azure.FakeAzureClient()


def _train(settings: dict, tmp_path: Path, sample_csv: Path, client=None) -> None:
    import yaml

    from etf_classifier.model.train import main

    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    args = ["--csv", str(sample_csv), "--config", str(config), "--yes"]
    assert main(args, client=client) == 0


@pytest.fixture
def trained_settings(app_settings, tmp_path, sample_csv):
    """Current model trained on the fixture, local blocks only."""
    _train(app_settings, tmp_path, sample_csv)
    return app_settings


@pytest.fixture
def trained_emb_settings(app_settings, tmp_path, sample_csv, fake_client):
    """Current model trained on the fixture with the (fake) Azure embeddings."""
    app_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    _train(app_settings, tmp_path, sample_csv, client=fake_client)
    return app_settings
