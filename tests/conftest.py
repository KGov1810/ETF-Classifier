"""Fixtures partagées par les tests."""

from pathlib import Path

import pytest

from etf_classifier.config import load_settings

FIXTURES_DIR = Path(__file__).parent / "data" / "fixtures"


@pytest.fixture
def settings() -> dict:
    """Configuration par défaut du projet."""
    return load_settings()


@pytest.fixture
def sample_csv() -> Path:
    """CSV fictif couvrant chaque règle de qualité."""
    return FIXTURES_DIR / "sample.csv"
