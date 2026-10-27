"""Most influential features per category (positive coefficients of the logistic regression).

Expected example: "gold", "NEM" and "miners" push towards the "Gold" category.
"""

import numpy as np


def top_features(model, feature_names: list[str], k: int = 10) -> dict[str, list[tuple[str, float]]]:
    """For each category, the k features with the most positive coefficient (none if nothing was learned)."""
    result: dict[str, list[tuple[str, float]]] = {}
    for category, coef in model.coefficients().items():
        if coef is None:  # single training category, or absent: nothing to learn
            result[category] = []
            continue
        coef = np.asarray(coef).ravel()
        best = np.argsort(-coef, kind="stable")[:k]
        result[category] = [(feature_names[i], float(coef[i])) for i in best if coef[i] > 0]
    return result
