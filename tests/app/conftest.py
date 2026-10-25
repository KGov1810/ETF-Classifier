"""Fixtures for the app tests: model trained on the fixture, fake chat client, `__main__` restored."""

import sys
from pathlib import Path

import pytest
import yaml

from etf_classifier.model.train import main as train_main


@pytest.fixture(autouse=True)
def restore_main_module():
    """AppTest leaves the last page as `__main__`: a later `spawn` subprocess (concurrency test of the
    feedback module) would then re-run that page at start-up and fail. Restore the real `__main__` after each test."""
    main = sys.modules["__main__"]
    attributes = {name: getattr(main, name) for name in ("__file__", "__spec__") if hasattr(main, name)}
    yield
    sys.modules["__main__"] = main
    for name, value in attributes.items():
        setattr(main, name, value)


def _train(settings: dict, tmp_path: Path, sample_csv: Path, client=None) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    assert train_main(["--csv", str(sample_csv), "--config", str(config), "--yes"], client=client) == 0


@pytest.fixture
def trained_settings(isolated_settings, tmp_path, sample_csv):
    """Current model trained on the fixture, local blocks only."""
    _train(isolated_settings, tmp_path, sample_csv)
    return isolated_settings


@pytest.fixture
def trained_emb_settings(isolated_settings, tmp_path, sample_csv, fake_client):
    """Current model trained on the fixture with the (fake) Azure embeddings."""
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    _train(isolated_settings, tmp_path, sample_csv, client=fake_client)
    return isolated_settings


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
def chat_client(llm_answer, fake_chat):
    return fake_chat.FakeChatClient(llm_answer)


@pytest.fixture
def azure_configured(monkeypatch, chat_client):
    """Azure 'configured' with the fake chat client: the app builds its client exactly as in production."""
    from etf_classifier.azure import ENV_VARIABLES

    for variable in ENV_VARIABLES:
        monkeypatch.setenv(variable, "test")
    monkeypatch.setattr("openai.AzureOpenAI", lambda **kwargs: chat_client)
    return chat_client
