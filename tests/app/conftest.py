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


@pytest.fixture(autouse=True)
def restore_main_module():
    """AppTest leaves the last page as `__main__`: a later `spawn` subprocess (concurrency test of the
    feedback module) would then re-run that page at start-up and fail. Restore the real `__main__` after each test."""
    import sys

    main = sys.modules["__main__"]
    attributes = {name: getattr(main, name) for name in ("__file__", "__spec__") if hasattr(main, name)}
    yield
    sys.modules["__main__"] = main
    for name, value in attributes.items():
        setattr(main, name, value)


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
    cfg["llm"].update({"justifications_path": str(tmp_path / "data" / "llm_justifications.jsonl"),
                       "cache_path": str(tmp_path / "cache" / "llm.sqlite")})
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


fake_chat = _load("fake_chat_app", TESTS / "llm" / "fake_chat.py")


@pytest.fixture
def llm_answer():
    return {
        "summary": "Gold miners: Gold fits; Commodities is doubtful.",
        "categories": [{"category": "Gold", "verdict": "fits", "reason": "Gold mining companies."}],
        "suggested_additions": ["ESG"],
        "suggested_removals": ["Commodities"],
        "new_category": {"name": "Precious Metals Miners", "description": "Miners of gold and silver",
                         "reason": "More precise than Commodities."},
        "inconsistency": None,
        "confidence": "medium",
    }


@pytest.fixture
def chat_client(llm_answer):
    return fake_chat.FakeChatClient(llm_answer)


@pytest.fixture
def azure_configured(monkeypatch, chat_client):
    """Azure 'configured' with the fake chat client: the app builds its client exactly as in production."""
    from etf_classifier.features.embeddings import ENV_VARIABLES

    for variable in ENV_VARIABLES:
        monkeypatch.setenv(variable, "test")
    monkeypatch.setattr("openai.AzureOpenAI", lambda **kwargs: chat_client)
    return chat_client
