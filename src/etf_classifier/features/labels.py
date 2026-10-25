"""Encoding of the categories as a multi-label matrix (one column per category)."""

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CategoryEncoder:
    """Learns the list of categories on the training set and applies it unchanged afterwards."""

    def __init__(self):
        self.classes: list[str] = []
        self._index: dict[str, int] = {}
        self.last_unknown: dict[str, int] = {}

    def fit(self, data: pd.DataFrame) -> "CategoryEncoder":
        self.classes = sorted({c for cats in data["categories"] for c in cats})
        self._index = {c: i for i, c in enumerate(self.classes)}
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """0/1 matrix; categories unknown to the training set are reported, never silently ignored."""
        matrix = np.zeros((len(data), len(self.classes)), dtype=np.int8)
        unknown: Counter = Counter()
        for row, cats in enumerate(data["categories"]):
            for category in cats:
                if category in self._index:
                    matrix[row, self._index[category]] = 1
                else:
                    unknown[category] += 1
        self.last_unknown = dict(unknown)
        if unknown:
            logger.warning("Categories missing from the training set (cannot be predicted): %s",
                           ", ".join(f"{c} ({n})" for c, n in unknown.most_common()))
        return matrix
