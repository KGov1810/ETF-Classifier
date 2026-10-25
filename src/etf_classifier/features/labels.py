"""Encodage des catégories en matrice multi-étiquettes (une colonne par catégorie)."""

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CategoryEncoder:
    """Apprend la liste des catégories sur l'entraînement et l'applique telle quelle ensuite."""

    def __init__(self):
        self.classes: list[str] = []
        self._index: dict[str, int] = {}
        self.last_unknown: dict[str, int] = {}

    def fit(self, data: pd.DataFrame) -> "CategoryEncoder":
        self.classes = sorted({c for cats in data["categories"] for c in cats})
        self._index = {c: i for i, c in enumerate(self.classes)}
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Matrice 0/1 ; les catégories inconnues de l'entraînement sont signalées, jamais ignorées en silence."""
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
            logger.warning("Catégories absentes de l'entraînement (non prédictibles) : %s",
                           ", ".join(f"{c} ({n})" for c, n in unknown.most_common()))
        return matrix
