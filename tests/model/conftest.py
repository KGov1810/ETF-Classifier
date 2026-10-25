"""Fixtures for the model tests: synthetic dataset, isolated configuration, fake Azure client."""

import copy
import importlib.util
from pathlib import Path

import pytest


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synthetic = _load("synthetic", Path(__file__).parent / "synthetic.py")
fake_azure = _load("fake_azure_model", Path(__file__).parent.parent / "features" / "fake_azure.py")


@pytest.fixture
def synthetic_data():
    return synthetic.make_synthetic(400, seed=0)


@pytest.fixture
def make_synthetic():
    return synthetic.make_synthetic


@pytest.fixture
def model_settings(settings, tmp_path):
    """Local blocks; artifacts, reports, cache and log in a temporary folder."""
    cfg = copy.deepcopy(settings)
    cfg["features"]["blocks"]["embeddings"]["enabled"] = False
    cfg["features"]["text_tfidf"]["min_df"] = 1
    cfg["features"]["embeddings"].update({"dimensions": 8, "cache_path": str(tmp_path / "cache" / "emb.sqlite")})
    cfg["features"]["artifacts_dir"] = str(tmp_path / "artifacts" / "features")
    cfg["model"]["artifacts_dir"] = str(tmp_path / "artifacts" / "models")
    cfg["model"]["reports_dir"] = str(tmp_path / "reports")
    cfg["costs"]["usage_log_path"] = str(tmp_path / "logs" / "usage.csv")
    return cfg


@pytest.fixture
def fake_client():
    return fake_azure.FakeAzureClient()
