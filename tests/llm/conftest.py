"""Fixtures for the llm tests: a proposal built by hand, known categories, isolated paths, fake chat client."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

fake_chat = None


def _load_fake_chat():
    spec = importlib.util.spec_from_file_location("fake_chat_llm", Path(__file__).parent / "fake_chat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def llm_settings(settings, tmp_path):
    cfg = copy.deepcopy(settings)
    cfg["llm"].update({"justifications_path": str(tmp_path / "data" / "llm_justifications.jsonl"),
                       "cache_path": str(tmp_path / "cache" / "llm.sqlite")})
    cfg["costs"]["usage_log_path"] = str(tmp_path / "logs" / "usage.csv")
    return cfg


def _neighbor(name, categories, overlap):
    return SimpleNamespace(name=name, categories=categories, overlap_pct=overlap, score=0.8,
                           shared_holdings=[("NEM", 10.0)])


@pytest.fixture
def proposal():
    holdings = {f"T{i}": float(20 - i) for i in range(14)}  # 14 holdings: only the 10 largest are sent
    return SimpleNamespace(
        name="Global Gold Miners", description="Companies mining gold worldwide",
        composition=holdings, categories=["Gold", "Commodities"],
        scores=[("Gold", 0.91), ("Commodities", 0.74), ("Global Equity", 0.20)], uncertain=False,
        neighbors=[_neighbor(f"Neighbour {i}", ["Gold"], 50.0 - i) for i in range(7)],
        model_version="20261025-120000",
    )


@pytest.fixture
def known_categories():
    return {"Gold": "Gold and gold miners", "Commodities": "", "Global Equity": "Developed markets equity",
            "ESG": "Environmental, social and governance screening"}


@pytest.fixture
def valid_answer():
    return {
        "summary": "A gold miners ETF; Gold and Commodities fit.",
        "categories": [{"category": "Gold", "verdict": "fits", "reason": "Gold mining companies."},
                       {"category": "Commodities", "verdict": "doubtful", "reason": "Equities, not commodities."}],
        "suggested_additions": [],
        "suggested_removals": ["Commodities"],
        "new_category": None,
        "inconsistency": None,
        "confidence": "high",
    }


@pytest.fixture
def chat_module():
    return _load_fake_chat()
