"""Fixtures for the features tests."""

import pytest


@pytest.fixture
def azure_settings(isolated_settings):
    """Isolated configuration with the embeddings block enabled (used with the fake client)."""
    isolated_settings["features"]["blocks"]["embeddings"]["enabled"] = True
    return isolated_settings


@pytest.fixture
def no_azure_env(monkeypatch):
    for name in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION",
                 "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]:
        monkeypatch.delenv(name, raising=False)
