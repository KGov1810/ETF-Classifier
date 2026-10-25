"""Most influential features per category (positive coefficients of the logistic regression).

Expected example: "gold", "NEM" and "miners" push towards the "Gold" category.
"""

import numpy as np


def top_features(model, feature_names: list[str], k: int = 10) -> dict[str, list[tuple[str, float]]]:
    """For each category, the k features with the most positive coefficient (none if the category is constant)."""
    result: dict[str, list[tuple[str, float]]] = {}
    for category, estimator in zip(model.classes, model.estimators):
        coef = getattr(estimator, "coef_", None)
        if coef is None:  # category present (or absent) everywhere in training: nothing to learn
            result[category] = []
            continue
        coef = np.asarray(coef).ravel()
        best = np.argsort(-coef, kind="stable")[:k]
        result[category] = [(feature_names[i], float(coef[i])) for i in best if coef[i] > 0]
    return result
