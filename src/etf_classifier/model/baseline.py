"""Trivial baseline: predict, for every ETF, the most frequent categories of the training set.

It uses no feature. A model that does not clearly beat it learns nothing useful.
"""

import numpy as np


class FrequencyBaseline:
    def __init__(self):
        self.order: np.ndarray = np.array([], dtype=int)
        self.n_predicted = 1
        self.n_classes = 0

    def fit(self, y_train: np.ndarray) -> "FrequencyBaseline":
        y_train = np.asarray(y_train)
        self.n_classes = y_train.shape[1]
        self.order = np.argsort(-y_train.sum(axis=0), kind="stable")
        median = np.median(y_train.sum(axis=1)) if len(y_train) else 1
        self.n_predicted = max(1, int(round(median)))
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        """Same prediction for every ETF: the median number of categories, among the most frequent ones."""
        row = np.zeros(self.n_classes, dtype=np.int8)
        row[self.order[: self.n_predicted]] = 1
        return np.tile(row, (n_rows, 1))
