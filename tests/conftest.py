"""Fixtures shared by the tests."""

from pathlib import Path

import pytest

from etf_classifier.config import load_settings

FIXTURES_DIR = Path(__file__).parent / "data" / "fixtures"


@pytest.fixture
def settings() -> dict:
    """Default project configuration."""
    return load_settings()


@pytest.fixture
def sample_csv() -> Path:
    """Fictitious CSV covering every quality rule."""
    return FIXTURES_DIR / "sample.csv"
