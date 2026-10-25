"""Variables les plus influentes par catégorie (coefficients positifs de la régression logistique).

Exemple attendu : « gold », « NEM » et « miners » poussent vers la catégorie « Or ».
"""

import numpy as np


def top_features(model, feature_names: list[str], k: int = 10) -> dict[str, list[tuple[str, float]]]:
    """Pour chaque catégorie, les k variables au coefficient le plus positif (aucune si la catégorie est constante)."""
    result: dict[str, list[tuple[str, float]]] = {}
    for category, estimator in zip(model.classes, model.estimators):
        coef = getattr(estimator, "coef_", None)
        if coef is None:  # catégorie présente (ou absente) partout à l'entraînement : rien à apprendre
            result[category] = []
            continue
        coef = np.asarray(coef).ravel()
        best = np.argsort(-coef, kind="stable")[:k]
        result[category] = [(feature_names[i], float(coef[i])) for i in best if coef[i] > 0]
    return result
