import numpy as np
import pandas as pd
import pytest

from etf_classifier.features.holdings import CompositionStatsBlock, HoldingsBlock, composition_stats


def _df(compositions):
    return pd.DataFrame({"composition": compositions})


TRAIN = _df([{"AAPL": 5.0, "MSFT": 4.0, "NVDA": 3.0}, {"AAPL": 6.0, "NEM": 10.0}, {"NEM": 12.0, "GOLD": 9.0}])


def test_ticker_seen_only_in_test_is_ignored():
    block = HoldingsBlock(min_df=1).fit(TRAIN)
    assert "holding:TSLA" not in block.feature_names()
    assert block.transform(_df([{"TSLA": 50.0}])).nnz == 0


def test_rare_tickers_weigh_more_than_common_ones():
    block = HoldingsBlock(min_df=1).fit(TRAIN)
    row = block.transform(_df([{"AAPL": 5.0, "NVDA": 5.0}])).toarray()[0]
    names = block.feature_names()
    assert row[names.index("holding:NVDA")] > row[names.index("holding:AAPL")]


def test_holdings_rows_are_l2_normalised():
    block = HoldingsBlock(min_df=1).fit(TRAIN)
    matrix = block.transform(TRAIN).toarray()
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_min_df_filters_rare_tickers():
    block = HoldingsBlock(min_df=2).fit(TRAIN)
    assert block.feature_names() == ["holding:AAPL", "holding:NEM"]


def test_stats_computed_by_hand():
    stats = composition_stats({"A": 50.0, "B": 30.0, "C": 20.0}, known_tickers={"A", "B"})
    assert stats["coverage"] == pytest.approx(100.0)
    assert stats["top1_weight"] == pytest.approx(50.0)
    assert stats["top10_concentration"] == pytest.approx(1.0)
    assert stats["herfindahl"] == pytest.approx(0.25 + 0.09 + 0.04)
    assert stats["unknown_weight_share"] == pytest.approx(0.2)
    assert stats["is_empty"] == 0.0


def test_top10_concentration_on_a_long_composition():
    composition = {f"T{i}": 1.0 for i in range(20)}
    assert composition_stats(composition, known_tickers=set())["top10_concentration"] == pytest.approx(0.5)


def test_empty_composition_has_finite_stats():
    stats = composition_stats({}, known_tickers=set())
    assert stats["is_empty"] == 1.0
    assert all(np.isfinite(value) for value in stats.values())


def test_stats_block_is_scaled_and_named():
    block = CompositionStatsBlock().fit(TRAIN)
    matrix = block.transform(TRAIN)
    assert "stats:herfindahl" in block.feature_names()
    assert matrix.shape == (3, len(block.feature_names()))
    assert np.allclose(matrix.toarray().mean(axis=0), 0.0, atol=1e-9)  # centré sur l'entraînement


def test_stats_block_measures_unknown_tickers_against_the_training_vocabulary():
    block = CompositionStatsBlock().fit(TRAIN)
    raw = block.raw_stats(_df([{"TSLA": 10.0, "AAPL": 10.0}]))
    assert raw.loc[0, "unknown_weight_share"] == pytest.approx(0.5)
