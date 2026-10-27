"""Fixtures for the similarity tests: a small readable reference set and its pipeline."""

import pandas as pd
import pytest

from etf_classifier.features.pipeline import FeaturePipeline


@pytest.fixture
def small_reference() -> pd.DataFrame:
    """Small readable reference set: two World trackers, a gold ETF, an ETF without composition."""
    return pd.DataFrame([
        {"row_id": 0, "ticker": "WLDA", "name": "World A", "description": "Tracks the MSCI World index",
         "composition": {"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0}, "categories": ["Global Equity"]},
        {"row_id": 1, "ticker": "WLDB", "name": "World B", "description": "MSCI World index tracker",
         "composition": {"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0}, "categories": ["Global Equity"]},
        {"row_id": 2, "ticker": "GDX", "name": "Gold Miners", "description": "Gold mining companies",
         "composition": {"NEM": 12.0, "GOLD": 10.0, "AEM": 9.0}, "categories": ["Gold"]},
        {"row_id": 3, "ticker": "PHAU", "name": "Physical Gold", "description": "Physical gold bullion",
         "composition": {}, "categories": ["Gold"]},
    ])


@pytest.fixture
def small_pipeline(small_reference, isolated_settings):
    return FeaturePipeline(isolated_settings).fit(small_reference)
