"""Fixtures for the llm tests: a proposal built by hand, known categories, a valid answer."""

from types import SimpleNamespace

import pytest


def _neighbor(i, category, overlap):
    return SimpleNamespace(ticker=f"NB{i}", name=f"Neighbour {i}", categories=[category], overlap_pct=overlap,
                           score=0.8, shared_holdings=[("NEM", 10.0)])


@pytest.fixture
def proposal():
    holdings = {f"T{i}": float(20 - i) for i in range(14)}  # 14 holdings: only the 10 largest are sent
    return SimpleNamespace(
        ticker="GGMX", name="Global Gold Miners", description="Companies mining gold worldwide",
        composition=holdings, categories=["Gold"],
        scores=[("Gold", 0.71), ("Commodities", 0.19), ("Global Equity", 0.10)], uncertain=False,
        neighbors=[_neighbor(i, "Gold", 50.0 - i) for i in range(7)],
        established="Commodities", established_score=0.19, check="disagreement",
        model_version="20261025-120000",
    )


@pytest.fixture
def known_categories():
    return {"Gold": "Gold and gold miners", "Commodities": "", "Global Equity": "Developed markets equity",
            "ESG": "Environmental, social and governance screening"}


@pytest.fixture
def valid_answer():
    return {
        "summary": "A gold miners ETF: Gold fits better than Commodities.",
        "verdict": "fits",
        "reason": "Its holdings are gold mining companies.",
        "alternative": None,
        "new_category": None,
        "inconsistency": None,
        "confidence": "high",
    }
