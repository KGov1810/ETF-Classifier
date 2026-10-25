"""Synthetic dataset where the answer is known: each category has its own signature words and tickers."""

import numpy as np
import pandas as pd

CATEGORIES = {
    "Gold": (["gold", "bullion", "miners"], ["NEM", "GOLD", "AEM", "KGC"]),
    "Technology": (["technology", "software", "semiconductor"], ["AAPL", "MSFT", "NVDA", "AVGO"]),
    "Healthcare": (["healthcare", "pharma", "biotech"], ["JNJ", "PFE", "MRK", "ABBV"]),
    "Energy": (["energy", "oil", "gas"], ["XOM", "CVX", "TTE", "SHEL"]),
    "Bonds": (["bond", "treasury", "fixed income"], ["UST10", "UST5", "BUND", "OAT"]),
    "ESG": (["esg", "sustainable", "responsible"], ["ESG1", "ESG2", "ESG3", "ESG4"]),
}
NOISE_WORDS = [f"word{i}" for i in range(300)]
NOISE_TICKERS = [f"N{i}" for i in range(500)]


def make_synthetic(n_etfs: int = 400, seed: int = 0) -> pd.DataFrame:
    """ETFs with 1 or 2 categories; description and composition mix signatures and noise."""
    rng = np.random.default_rng(seed)
    names = list(CATEGORIES)
    rows = []
    for i in range(n_etfs):
        n_cats = 1 if rng.random() < 0.7 else 2
        cats = list(rng.choice(names, size=n_cats, replace=False))
        words, tickers = [], {}
        for cat in cats:
            cat_words, cat_tickers = CATEGORIES[cat]
            words += list(rng.choice(cat_words, size=2, replace=False))
            for ticker in rng.choice(cat_tickers, size=3, replace=False):
                tickers[str(ticker)] = float(rng.uniform(3, 10))
        words += list(rng.choice(NOISE_WORDS, size=8))
        for ticker in rng.choice(NOISE_TICKERS, size=10, replace=False):
            tickers[str(ticker)] = float(rng.uniform(0.5, 3))
        rows.append({
            "name": f"ETF {i}",
            "description": " ".join(rng.permutation(words)),
            "composition": tickers,
            "categories": [str(c) for c in cats],
            "group_id": i,
        })
    return pd.DataFrame(rows)
