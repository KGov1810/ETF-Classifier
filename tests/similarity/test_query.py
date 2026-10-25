import time

import pandas as pd
import pytest

from etf_classifier.features.pipeline import FeaturePipeline
from etf_classifier.similarity.index import SimilarityIndex


def _query_df(**fields):
    base = {"name": "Query", "description": "", "composition": {}, "categories": []}
    return pd.DataFrame([{**base, **fields}])


@pytest.fixture
def index(small_reference, small_pipeline, isolated_settings):
    return SimilarityIndex.build(small_reference, small_pipeline, isolated_settings)


def test_identical_composition_gives_similarity_1_and_a_near_duplicate_alert(index, small_pipeline):
    query = _query_df(name="World C", description="World index",
                      composition={"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0})
    result = index.query(query, small_pipeline, k=2)[0]
    assert result.neighbors[0].composition_similarity == pytest.approx(1.0)
    assert result.near_duplicate is not None
    assert result.near_duplicate.name in {"World A", "World B"}


def test_small_etf_contained_in_a_large_one_is_not_identical(small_pipeline, isolated_settings):
    large = {f"T{i}": 10.0 for i in range(10)}
    reference = pd.DataFrame([{"row_id": 0, "name": "Large", "description": "broad",
                               "composition": large, "categories": ["A"]}])
    index = SimilarityIndex.build(reference, FeaturePipeline(isolated_settings).fit(reference), isolated_settings)
    query = _query_df(composition={"T0": 10.0, "T1": 10.0, "T2": 10.0})
    neighbor = index.query(query, index_pipeline(reference, isolated_settings), k=1)[0].neighbors[0]
    assert neighbor.overlap_pct == pytest.approx(100.0)
    assert neighbor.composition_similarity < 0.6


def index_pipeline(reference, settings):
    return FeaturePipeline(settings).fit(reference)


def test_an_etf_of_the_reference_is_never_its_own_neighbor(index, small_reference, small_pipeline):
    results = index.query(small_reference, small_pipeline, k=3, exclude_row_ids=small_reference["row_id"])
    for row_id, result in zip(small_reference["row_id"], results):
        assert row_id not in [n.row_id for n in result.neighbors]


def test_neighbors_are_sorted_by_decreasing_score(index, small_pipeline):
    query = _query_df(description="gold miners", composition={"NEM": 10.0, "GOLD": 8.0})
    scores = [n.score for n in index.query(query, small_pipeline, k=4)[0].neighbors]
    assert scores == sorted(scores, reverse=True)
    assert index.query(query, small_pipeline, k=1)[0].neighbors[0].name == "Gold Miners"


def test_empty_composition_uses_text_only(index, small_pipeline):
    result = index.query(_query_df(description="Physical gold bullion"), small_pipeline, k=1)[0]
    assert result.text_only is True
    assert result.neighbors[0].name == "Physical Gold"
    assert result.neighbors[0].score == pytest.approx(result.neighbors[0].text_similarity)


def test_explanation_lists_shared_holdings_by_weight(index, small_pipeline):
    query = _query_df(composition={"NVDA": 6.0, "AAPL": 2.0, "TSLA": 9.0})
    neighbor = index.query(query, small_pipeline, k=1)[0].neighbors[0]
    assert [ticker for ticker, _ in neighbor.shared_holdings] == ["NVDA", "AAPL"]


def test_unknown_tickers_lower_the_composition_similarity(index, small_pipeline):
    known = _query_df(composition={"NEM": 12.0, "GOLD": 10.0, "AEM": 9.0})
    half_unknown = _query_df(composition={"NEM": 12.0, "GOLD": 10.0, "AEM": 9.0, "X1": 12.0, "X2": 10.0, "X3": 9.0})
    best_known = index.query(known, small_pipeline, k=1)[0].neighbors[0].composition_similarity
    best_half = index.query(half_unknown, small_pipeline, k=1)[0].neighbors[0].composition_similarity
    assert best_half < best_known


@pytest.mark.slow
def test_100_queries_on_10000_etfs_in_less_than_10_seconds(make_synthetic, isolated_settings):
    reference = make_synthetic(10_000, seed=4).assign(row_id=lambda d: range(len(d)))
    pipeline = FeaturePipeline(isolated_settings).fit(reference)
    index = SimilarityIndex.build(reference, pipeline, isolated_settings)
    queries = make_synthetic(100, seed=5)
    start = time.perf_counter()
    index.query(queries, pipeline, k=5)
    assert time.perf_counter() - start < 10
