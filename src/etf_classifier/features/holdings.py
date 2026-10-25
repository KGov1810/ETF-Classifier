"""holdings (weighted tickers) and composition_stats (portfolio shape) blocks."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler, normalize

STAT_NAMES = ["coverage", "log_n_holdings", "top1_weight", "top10_concentration",
              "herfindahl", "is_empty", "unknown_weight_share"]


class HoldingsBlock:
    """Sparse vector ticker → weight (%) × ticker rarity (IDF), L2-normalised.

    The vocabulary and the IDF are learned on the training set; an unknown ticker is ignored.
    """

    name = "holdings"

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])

    def fit(self, data: pd.DataFrame) -> "HoldingsBlock":
        document_frequency: dict[str, int] = {}
        for composition in data["composition"]:
            for ticker in composition:
                document_frequency[ticker] = document_frequency.get(ticker, 0) + 1
        kept = sorted(t for t, df in document_frequency.items() if df >= self.min_df)
        self.vocabulary = {t: i for i, t in enumerate(kept)}
        n_docs = len(data)
        # smoothed IDF (same formula as scikit-learn): very common tickers weigh less
        self.idf = np.array([np.log((1 + n_docs) / (1 + document_frequency[t])) + 1 for t in kept])
        return self

    def transform(self, data: pd.DataFrame) -> sparse.csr_matrix:
        rows, cols, values = [], [], []
        for row, composition in enumerate(data["composition"]):
            for ticker, weight in composition.items():
                col = self.vocabulary.get(ticker)
                if col is not None and weight > 0:
                    rows.append(row)
                    cols.append(col)
                    values.append(weight * self.idf[col])
        matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(data), max(len(self.vocabulary), 1)))
        if not self.vocabulary:
            matrix = matrix[:, :0]
        return normalize(matrix, norm="l2", copy=False).tocsr()

    def feature_names(self) -> list[str]:
        return [f"holding:{t}" for t in self.vocabulary]


def composition_stats(composition: dict[str, float], known_tickers: set[str]) -> dict[str, float]:
    """Statistics of a composition; always finite, including for an empty composition."""
    weights = np.array(sorted(composition.values(), reverse=True), dtype=float)
    coverage = float(weights.sum())
    if coverage <= 0:
        return {"coverage": 0.0, "log_n_holdings": 0.0, "top1_weight": 0.0, "top10_concentration": 0.0,
                "herfindahl": 0.0, "is_empty": 1.0, "unknown_weight_share": 0.0}
    shares = weights / coverage
    unknown = sum(w for t, w in composition.items() if t not in known_tickers)
    return {
        "coverage": coverage,
        "log_n_holdings": float(np.log1p(len(weights))),
        "top1_weight": float(weights[0]),
        "top10_concentration": float(shares[:10].sum()),
        "herfindahl": float((shares ** 2).sum()),
        "is_empty": 0.0,
        "unknown_weight_share": float(unknown / coverage),
    }


class CompositionStatsBlock:
    """Portfolio shape (coverage, concentration…), standardised on the training set."""

    name = "composition_stats"

    def __init__(self):
        self.known_tickers: set[str] = set()
        self._scaler = StandardScaler()

    def raw_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([composition_stats(c, self.known_tickers) for c in data["composition"]],
                            columns=STAT_NAMES)

    def fit(self, data: pd.DataFrame) -> "CompositionStatsBlock":
        self.known_tickers = {t for composition in data["composition"] for t in composition}
        self._scaler.fit(self.raw_stats(data).to_numpy())
        return self

    def transform(self, data: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.csr_matrix(self._scaler.transform(self.raw_stats(data).to_numpy()))

    def feature_names(self) -> list[str]:
        return [f"stats:{name}" for name in STAT_NAMES]
