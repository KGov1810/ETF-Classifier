"""Vote des voisins : seconde méthode de catégorisation, comparée au modèle.

Score d'une catégorie = somme des scores des voisins qui l'ont / somme des scores de tous les voisins.
"""

import numpy as np

from etf_classifier.similarity.index import Neighbor, QueryResult


def neighbor_vote(neighbors: list[Neighbor]) -> dict[str, float]:
    """Score de chaque catégorie présente chez les voisins, trié par score décroissant."""
    weights = [max(n.score, 0.0) for n in neighbors]
    total = sum(weights)
    votes: dict[str, float] = {}
    for neighbor, weight in zip(neighbors, weights):
        for category in neighbor.categories:
            votes[category] = votes.get(category, 0.0) + (weight / total if total else 0.0)
    return dict(sorted(votes.items(), key=lambda item: (-item[1], item[0])))


def vote_matrix(results: list[QueryResult], classes: list[str]) -> np.ndarray:
    """Matrice des scores de vote (une ligne par ETF, une colonne par catégorie connue)."""
    position = {c: j for j, c in enumerate(classes)}
    matrix = np.zeros((len(results), len(classes)))
    for row, result in enumerate(results):
        for category, score in neighbor_vote(result.neighbors).items():
            if category in position:
                matrix[row, position[category]] = score
    return matrix
