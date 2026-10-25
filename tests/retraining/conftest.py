"""Fixtures for the retraining tests: every file in a temporary folder, fake Azure client."""

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


fake_azure = _load("fake_azure_rt", TESTS / "features" / "fake_azure.py")


@pytest.fixture
def rt_settings(settings, tmp_path, sample_csv):
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
    cfg["retraining"].update({
        "history_path": str(tmp_path / "artifacts" / "models" / "retraining_history.csv"),
        "log_path": str(tmp_path / "logs" / "retraining.log"),
    })
    return cfg


@pytest.fixture
def fake_client():
    return fake_azure.FakeAzureClient()
