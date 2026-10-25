"""Référence triviale : prédire pour tout ETF les catégories les plus fréquentes de l'entraînement.

Elle n'utilise aucune variable. Un modèle qui ne la bat pas nettement n'apprend rien d'utile.
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
        """Même prédiction pour chaque ETF : le nombre médian de catégories, parmi les plus fréquentes."""
        row = np.zeros(self.n_classes, dtype=np.int8)
        row[self.order[: self.n_predicted]] = 1
        return np.tile(row, (n_rows, 1))
