"""Détection des quasi-doublons (ETF répliquant le même indice).

Recouvrement pondéré entre deux ETF A et B :
    Σ min(wA, wB) / min(ΣwA, ΣwB)    sur les tickers communs.
On divise par la plus petite couverture car beaucoup de compositions sont partielles
(10 premières lignes seulement). Deux ETF sont liés si le recouvrement atteint le seuil
et s'ils ont au moins `min_common` tickers en commun ; les groupes sont les composantes
connexes de ce graphe (donc transitifs).

Performance : la matrice ETF × tickers est stockée en colonnes creuses (CSC). Pour chaque
ETF, on ne parcourt que les colonnes de ses propres tickers : seules les paires qui
partagent au moins un ticker sont examinées, jamais les ~50 millions de paires possibles.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

_EPSILON = 1e-9


@dataclass
class DedupResult:
    """Groupes de quasi-doublons : un group_id par ETF, dans l'ordre d'entrée."""

    group_ids: np.ndarray
    n_groups: int
    largest_group_size: int
    n_links: int
    warnings: list[str] = field(default_factory=list)


def weighted_overlap(first: dict[str, float], second: dict[str, float]) -> float:
    """Recouvrement pondéré de deux compositions (version de référence, non vectorisée)."""
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
    """Paires (i, j, recouvrement) avec i < j qui dépassent le seuil et le nombre de tickers communs."""
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
        block = matrix_cols[:, own_cols]  # ETF qui détiennent au moins un ticker de i
        per_column = np.diff(block.indptr)
        shared_min = np.minimum(block.data, np.repeat(own_weights, per_column))
        others = block.indices
        keep = others > i  # chaque paire n'est examinée qu'une fois
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
    """Regroupe les quasi-doublons ; avertit si un groupe dépasse max_group_share du jeu."""
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
            f"Le plus gros groupe de quasi-doublons contient {largest} ETF "
            f"({largest / n_etfs:.1%} du jeu, au-delà de {max_group_share:.0%}) : "
            "possible effet de chaîne, vérifier le seuil de recouvrement."
        )
    return DedupResult(labels.astype(int), int(n_groups), largest, len(pairs), warnings)


def add_groups(data, settings: dict) -> DedupResult:
    """Ajoute la colonne group_id au jeu exploitable (en place) et renvoie le résultat détaillé."""
    cfg = settings["dedup"]
    result = assign_groups(
        list(data["composition"]),
        threshold=cfg["overlap_threshold"],
        min_common=cfg["min_common_holdings"],
        max_group_share=cfg["max_group_share"],
    )
    data["group_id"] = result.group_ids
    return result
