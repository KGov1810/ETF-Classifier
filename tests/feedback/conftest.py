"""Fixtures du module feedback : journal, registre et sauvegardes dans un dossier temporaire."""

import copy

import pytest

from etf_classifier.feedback.store import FeedbackStore


@pytest.fixture
def fb_settings(settings, tmp_path, sample_csv):
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
    cfg["features"]["artifacts_dir"] = str(tmp_path / "artifacts" / "features")
    cfg["model"]["artifacts_dir"] = str(tmp_path / "artifacts" / "models")
    cfg["model"]["reports_dir"] = str(tmp_path / "reports")
    return cfg


@pytest.fixture
def store(fb_settings):
    return FeedbackStore(fb_settings)


@pytest.fixture
def new_etf():
    return {"name": "New Clean Energy", "description": "Global clean energy producers",
            "composition": {"ENPH": 6.0, "FSLR": 5.0, "VWS": 4.0}}
