"""Fixtures shared by the tests: configuration, isolated paths, fictitious data and fake Azure clients."""

import copy
import importlib.util
from pathlib import Path

import pytest

from etf_classifier.config import load_settings

TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "data" / "fixtures"


def _load_support(name: str):
    """Test doubles of tests/support/ (loaded by path: the tests folder is not a package)."""
    spec = importlib.util.spec_from_file_location(f"support_{name}", TESTS_DIR / "support" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fake_azure_module = _load_support("fake_azure")
fake_chat_module = _load_support("fake_chat")
synthetic_module = _load_support("synthetic")


@pytest.fixture
def settings() -> dict:
    """Default project configuration."""
    return load_settings()


@pytest.fixture
def sample_csv() -> Path:
    """Fictitious CSV covering every quality rule."""
    return FIXTURES_DIR / "sample.csv"


@pytest.fixture
def isolated_settings(settings, tmp_path, sample_csv) -> dict:
    """Default configuration where every file lives in a temporary folder.

    Journal, category registry, backups, artifacts, reports, caches and logs are isolated; the reference CSV is
    the fixture; local feature blocks only (embeddings disabled, small dimension when a test enables them).
    """
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
    cfg["llm"].update({"justifications_path": str(tmp_path / "data" / "llm_justifications.jsonl"),
                       "cache_path": str(tmp_path / "cache" / "llm.sqlite")})
    return cfg


@pytest.fixture
def fake_client():
    """Fake Azure embeddings client: deterministic vectors, no network call."""
    return fake_azure_module.FakeAzureClient()


@pytest.fixture
def fake_vector():
    """Raw vector returned by the fake embeddings client for a given text."""
    return fake_azure_module.fake_vector


@pytest.fixture
def fake_chat():
    """Module of the fake chat client (FakeChatClient(answer, reject_schema=…, finish_reason=…))."""
    return fake_chat_module


@pytest.fixture
def make_synthetic():
    """Synthetic dataset generator whose answer is known (categories set by words and tickers)."""
    return synthetic_module.make_synthetic


@pytest.fixture
def synthetic_data():
    return synthetic_module.make_synthetic(400, seed=0)
