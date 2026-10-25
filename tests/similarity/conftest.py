"""Fixtures du module similarity : jeu synthétique, configuration isolée, faux client Azure."""

import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from etf_classifier.features.pipeline import FeaturePipeline

TESTS = Path(__file__).parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synthetic = _load("synthetic_sim", TESTS / "model" / "synthetic.py")
fake_azure = _load("fake_azure_sim", TESTS / "features" / "fake_azure.py")


@pytest.fixture
def sim_settings(settings, tmp_path):
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
def make_synthetic():
    return synthetic.make_synthetic


@pytest.fixture
def fake_client():
    return fake_azure.FakeAzureClient()


@pytest.fixture
def small_reference() -> pd.DataFrame:
    """Petit jeu de référence lisible : deux trackers Monde, un ETF or, un ETF sans composition."""
    return pd.DataFrame([
        {"row_id": 0, "name": "World A", "description": "Tracks the MSCI World index",
         "composition": {"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0}, "categories": ["Actions Monde"]},
        {"row_id": 1, "name": "World B", "description": "MSCI World index tracker",
         "composition": {"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0}, "categories": ["Actions Monde"]},
        {"row_id": 2, "name": "Gold Miners", "description": "Gold mining companies",
         "composition": {"NEM": 12.0, "GOLD": 10.0, "AEM": 9.0}, "categories": ["Or"]},
        {"row_id": 3, "name": "Physical Gold", "description": "Physical gold bullion",
         "composition": {}, "categories": ["Or"]},
    ])


@pytest.fixture
def small_pipeline(small_reference, sim_settings):
    return FeaturePipeline(sim_settings).fit(small_reference)
