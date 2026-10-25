"""Fixtures for the feedback tests: a store on the isolated journal, a new ETF."""

import pytest

from etf_classifier.feedback.store import FeedbackStore


@pytest.fixture
def store(isolated_settings):
    return FeedbackStore(isolated_settings)


@pytest.fixture
def new_etf():
    return {"name": "New Clean Energy", "description": "Global clean energy producers",
            "composition": {"ENPH": 6.0, "FSLR": 5.0, "VWS": 4.0}}
