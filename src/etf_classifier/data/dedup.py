"""Detection of near-duplicates (ETFs replicating the same index).

Weighted overlap between two ETFs A and B:
    Σ min(wA, wB) / min(ΣwA, ΣwB)    over the shared tickers.
We divide by the smallest coverage because many compositions are partial (top 10 holdings
only). Two ETFs are linked when the overlap reaches the threshold and they share at least
`min_common` tickers; groups are the connected components of this graph (hence transitive).

Performance: the ETF × ticker matrix is stored in sparse columns (CSC). For each ETF, only the
columns of its own tickers are scanned: only pairs sharing at least one ticker are examined,
never the ~50 million possible pairs.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

_EPSILON = 1e-9


@dataclass
class DedupResult:
    """Near-duplicate groups: one group_id per ETF, in input order."""

    group_ids: np.ndarray
    n_groups: int
    largest_group_size: int
    n_links: int
    warnings: list[str] = field(default_factory=list)


def weighted_overlap(first: dict[str, float], second: dict[str, float]) -> float:
    """Weighted overlap of two compositions (reference version, not vectorised)."""
    smallest_coverage = min(sum(first.values()), sum(second.values()))
    if smallest_coverage <= 0:
        return 0.0
    shared = sum(min(first[t], second[t]) for t in set(first) & set(second))
    return shared / smallest_coverage


def _to_sparse(compositions: list[dict[str, float]]) -> sparse.csr_matrix:
    vocabulary: dict[str, int] = {}
    rows, cols, values = [], [], []
    for row, composition in enumerate(compositions):
        for ticker, weight in composition.items():
            if weight > 0:
                rows.append(row)
                cols.append(vocabulary.setdefault(ticker, len(vocabulary)))
                values.append(weight)
    shape = (len(compositions), max(len(vocabulary), 1))
    return sparse.csr_matrix((values, (rows, cols)), shape=shape, dtype=np.float64)


def find_near_duplicate_pairs(
    compositions: list[dict[str, float]], threshold: float, min_common: int
) -> list[tuple[int, int, float]]:
    """Pairs (i, j, overlap) with i < j that reach the threshold and the number of shared tickers."""
    n_etfs = len(compositions)
    matrix_rows = _to_sparse(compositions)
    matrix_cols = matrix_rows.tocsc()
    coverage = np.asarray(matrix_rows.sum(axis=1)).ravel()
    pairs: list[tuple[int, int, float]] = []

    for i in range(n_etfs):
        start, end = matrix_rows.indptr[i], matrix_rows.indptr[i + 1]
        if end - start < min_common:
            continue
        own_cols, own_weights = matrix_rows.indices[start:end], matrix_rows.data[start:end]
        block = matrix_cols[:, own_cols]  # ETFs holding at least one of i's tickers
        per_column = np.diff(block.indptr)
        shared_min = np.minimum(block.data, np.repeat(own_weights, per_column))
        others = block.indices
        keep = others > i  # each pair is examined only once
        others, shared_min = others[keep], shared_min[keep]
        if others.size == 0:
            continue
        candidates, inverse, common_counts = np.unique(others, return_inverse=True, return_counts=True)
        shared = np.bincount(inverse, weights=shared_min)
        smallest = np.minimum(coverage[i], coverage[candidates])
        overlap = np.divide(shared, smallest, out=np.zeros_like(shared), where=smallest > 0)
        linked = (common_counts >= min_common) & (overlap >= threshold - _EPSILON)
        pairs.extend((i, int(j), float(o)) for j, o in zip(candidates[linked], overlap[linked]))
    return pairs


def assign_groups(
    compositions: list[dict[str, float]],
    threshold: float,
    min_common: int,
    max_group_share: float = 1.0,
) -> DedupResult:
    """Groups the near-duplicates; warns when a group exceeds max_group_share of the dataset."""
    n_etfs = len(compositions)
    if n_etfs == 0:
        return DedupResult(np.array([], dtype=int), 0, 0, 0)
    pairs = find_near_duplicate_pairs(compositions, threshold, min_common)
    first = np.array([p[0] for p in pairs], dtype=int)
    second = np.array([p[1] for p in pairs], dtype=int)
    graph = sparse.coo_matrix((np.ones(len(pairs)), (first, second)), shape=(n_etfs, n_etfs))
    n_groups, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    largest = int(sizes.max())
    warnings = []
    if largest > max_group_share * n_etfs and largest > 1:
        warnings.append(
            f"The largest near-duplicate group contains {largest} ETFs "
            f"({largest / n_etfs:.1%} of the dataset, above {max_group_share:.0%}): "
            "possible chaining effect, check the overlap threshold."
        )
    return DedupResult(labels.astype(int), int(n_groups), largest, len(pairs), warnings)


def add_groups(data, settings: dict) -> DedupResult:
    """Adds the group_id column to the usable dataset (in place) and returns the detailed result."""
    cfg = settings["dedup"]
    result = assign_groups(
        list(data["composition"]),
        threshold=cfg["overlap_threshold"],
        min_common=cfg["min_common_holdings"],
        max_group_share=cfg["max_group_share"],
    )
    data["group_id"] = result.group_ids
    return result
