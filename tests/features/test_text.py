import numpy as np
import pandas as pd

from etf_classifier.features.text import TextTfidfBlock, build_text


def _df(rows):
    return pd.DataFrame(rows, columns=["name", "description"])


TRAIN = _df([
    ("Clean Energy ETF", "Invests in clean energy companies worldwide."),
    ("Global Clean Energy", "Tracks clean energy producers."),
    ("World Equity", "Tracks the MSCI World index."),
    ("World ESG Equity", "Tracks the MSCI World ESG index."),
])


def test_text_is_name_then_description():
    assert build_text("World Equity", "Tracks MSCI World.") == "World Equity. Tracks MSCI World."


def test_word_seen_only_in_test_is_not_in_the_vocabulary():
    block = TextTfidfBlock(ngram_max=2, min_df=1, max_features=1000).fit(TRAIN)
    assert not any("uranium" in name for name in block.feature_names())
    row = block.transform(_df([("Uranium Miners", "uranium uranium")]))
    assert row.nnz == 0


def test_empty_description_gives_a_null_row_without_error():
    block = TextTfidfBlock(ngram_max=2, min_df=1, max_features=1000).fit(TRAIN)
    assert block.transform(_df([("", "")])).nnz == 0


def test_feature_names_are_readable():
    block = TextTfidfBlock(ngram_max=2, min_df=2, max_features=1000).fit(TRAIN)
    assert "text:clean energy" in block.feature_names()


def test_english_stop_words_are_removed():
    block = TextTfidfBlock(ngram_max=1, min_df=1, max_features=1000).fit(TRAIN)
    assert "text:the" not in block.feature_names()


def test_too_strict_min_df_falls_back_instead_of_failing():
    tiny = _df([("Alpha", "one"), ("Beta", "two")])
    block = TextTfidfBlock(ngram_max=1, min_df=2, max_features=1000).fit(tiny)
    assert block.transform(tiny).shape[0] == 2


def test_rows_are_l2_normalised():
    block = TextTfidfBlock(ngram_max=2, min_df=1, max_features=1000).fit(TRAIN)
    norms = np.sqrt(np.asarray(block.transform(TRAIN).multiply(block.transform(TRAIN)).sum(axis=1))).ravel()
    assert np.allclose(norms, 1.0)
