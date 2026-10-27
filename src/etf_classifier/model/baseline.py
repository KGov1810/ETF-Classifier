"""Trivial baseline: predict, for every ETF, the most frequent category of the training set.

It uses no feature. A model that does not clearly beat it learns nothing useful.
"""

import numpy as np


class FrequencyBaseline:
    def __init__(self):
        self.best = 0
        self.n_classes = 0

    def fit(self, y_train: np.ndarray) -> "FrequencyBaseline":
        y_train = np.asarray(y_train)
        self.n_classes = y_train.shape[1]
        self.best = int(np.argmax(y_train.sum(axis=0)))  # on a tie, the first category
        return self

    def predict(self, n_rows: int) -> np.ndarray:
        """Same prediction for every ETF: the most frequent category."""
        predicted = np.zeros((n_rows, self.n_classes), dtype=np.int8)
        predicted[:, self.best] = 1
        return predicted
