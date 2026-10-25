"""Fixtures for the features tests: isolated configuration and fake Azure client."""

import copy
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("fake_azure", Path(__file__).parent / "fake_azure.py")
fake_azure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fake_azure)


@pytest.fixture
def fake_client():
    return fake_azure.FakeAzureClient()


@pytest.fixture
def fake_vector():
    """Raw vector returned by the fake client for a given text."""
    return fake_azure.fake_vector


@pytest.fixture
def azure_settings(settings, tmp_path):
    """Test configuration: small dimension, cache, log and artifacts in a temporary folder."""
    cfg = copy.deepcopy(settings)
    emb = cfg["features"]["embeddings"]
    emb.update({"dimensions": 8, "batch_size": 64, "cache_path": str(tmp_path / "cache" / "emb.sqlite")})
    cfg["features"]["text_tfidf"]["min_df"] = 1
    cfg["features"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg["costs"]["usage_log_path"] = str(tmp_path / "logs" / "usage.csv")
    return cfg


@pytest.fixture
def no_azure_env(monkeypatch):
    for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
                 "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]:
        monkeypatch.delenv(name, raising=False)
