import time

import numpy as np
import pytest

from etf_classifier.data.dedup import assign_groups, find_near_duplicate_pairs, weighted_overlap


def _index(n_holdings: int, prefix: str = "T") -> dict[str, float]:
    """Fictitious index with decreasing weights summing to 100."""
    raw = np.array([1.0 / (rank + 1) for rank in range(n_holdings)])
    weights = raw / raw.sum() * 100
    return {f"{prefix}{rank}": float(weight) for rank, weight in enumerate(weights)}


def test_overlap_is_relative_to_the_smallest_coverage():
    full = {"A": 10.0, "B": 10.0, "C": 80.0}
    top_two = {"A": 10.0, "B": 10.0}
    assert weighted_overlap(full, top_two) == pytest.approx(1.0)


def test_overlap_of_disjoint_etfs_is_zero():
    assert weighted_overlap({"A": 50.0}, {"B": 50.0}) == 0.0


def test_full_tracker_and_top10_tracker_are_grouped():
    full = _index(200)
    top10 = dict(list(full.items())[:10])
    result = assign_groups([full, top10], threshold=0.8, min_common=3)
    assert result.group_ids[0] == result.group_ids[1]


def test_single_shared_big_line_is_not_enough():
    first = {"X": 50.0, "A1": 1.0, "A2": 1.0}
    second = {"X": 50.0, "B1": 1.0, "B2": 1.0}
    result = assign_groups([first, second], threshold=0.8, min_common=3)
    assert result.group_ids[0] != result.group_ids[1]


def test_unrelated_etfs_stay_in_separate_groups():
    result = assign_groups([_index(30, "U"), _index(30, "E")], threshold=0.8, min_common=3)
    assert result.n_groups == 2


def test_empty_compositions_are_singletons():
    result = assign_groups([{}, {}, _index(10)], threshold=0.8, min_common=3)
    assert len(set(result.group_ids)) == 3


def test_groups_are_transitive():
    base = {f"T{i}": 10.0 for i in range(10)}
    a = dict(base)
    b = {**{f"T{i}": 10.0 for i in range(8)}, "N1": 10.0, "N2": 10.0}
    c = {**{f"T{i}": 10.0 for i in range(6)}, "N1": 10.0, "N2": 10.0, "N3": 10.0, "N4": 10.0}
    assert weighted_overlap(a, c) < 0.8
    result = assign_groups([a, b, c], threshold=0.8, min_common=3)
    assert len(set(result.group_ids)) == 1


def test_group_larger_than_allowed_share_raises_a_warning():
    clones = [_index(20) for _ in range(5)] + [_index(20, f"Z{k}_") for k in range(45)]
    result = assign_groups(clones, threshold=0.8, min_common=3, max_group_share=0.02)
    assert result.largest_group_size == 5
    assert result.warnings and "5" in result.warnings[0]


def test_group_ids_are_consecutive_integers():
    result = assign_groups([_index(10), _index(10), _index(10, "Q")], threshold=0.8, min_common=3)
    assert sorted(set(result.group_ids)) == list(range(result.n_groups))


def test_vectorised_pairs_match_a_naive_computation():
    rng = np.random.default_rng(0)
    universe = [f"S{i}" for i in range(40)]
    compositions = []
    for _ in range(60):
        size = int(rng.integers(3, 15))
        tickers = rng.choice(universe, size=size, replace=False)
        compositions.append({t: float(w) for t, w in zip(tickers, rng.uniform(0.5, 10, size))})
    expected = set()
    for i in range(len(compositions)):
        for j in range(i + 1, len(compositions)):
            common = set(compositions[i]) & set(compositions[j])
            if len(common) >= 3 and weighted_overlap(compositions[i], compositions[j]) >= 0.5:
                expected.add((i, j))
    found = {(i, j) for i, j, _ in find_near_duplicate_pairs(compositions, threshold=0.5, min_common=3)}
    assert found == expected


@pytest.mark.slow
def test_10000_etfs_are_processed_in_less_than_60_seconds():
    rng = np.random.default_rng(42)
    n_tickers = 8000
    popularity = 1.0 / np.arange(1, n_tickers + 1) ** 0.8
    popularity /= popularity.sum()
    compositions = []
    for _ in range(10_000):
        size = int(rng.integers(10, 120))
        tickers = rng.choice(n_tickers, size=size, replace=False, p=popularity)
        weights = rng.dirichlet(np.ones(size)) * rng.uniform(20, 100)
        compositions.append({f"T{t}": float(w) for t, w in zip(tickers, weights)})
    start = time.perf_counter()
    assign_groups(compositions, threshold=0.8, min_common=3)
    assert time.perf_counter() - start < 60
