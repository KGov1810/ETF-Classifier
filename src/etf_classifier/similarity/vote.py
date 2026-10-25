"""Neighbour vote: a second categorisation method, compared with the model.

Score of a category = sum of the scores of the neighbours holding it / sum of the scores of all neighbours.
"""

import numpy as np

from etf_classifier.similarity.index import Neighbor, QueryResult


def neighbor_vote(neighbors: list[Neighbor]) -> dict[str, float]:
    """Score of each category present among the neighbours, sorted by decreasing score."""
    weights = [max(n.score, 0.0) for n in neighbors]
    total = sum(weights)
    votes: dict[str, float] = {}
    for neighbor, weight in zip(neighbors, weights):
        for category in neighbor.categories:
            votes[category] = votes.get(category, 0.0) + (weight / total if total else 0.0)
    return dict(sorted(votes.items(), key=lambda item: (-item[1], item[0])))


def vote_matrix(results: list[QueryResult], classes: list[str]) -> np.ndarray:
    """Matrix of vote scores (one row per ETF, one column per known category)."""
    position = {c: j for j, c in enumerate(classes)}
    matrix = np.zeros((len(results), len(classes)))
    for row, result in enumerate(results):
        for category, score in neighbor_vote(result.neighbors).items():
            if category in position:
                matrix[row, position[category]] = score
    return matrix
