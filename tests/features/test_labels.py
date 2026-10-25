import logging

import pandas as pd

from etf_classifier.features.labels import CategoryEncoder


def _df(categories):
    return pd.DataFrame({"categories": categories})


def test_train_and_test_matrices_share_the_same_columns():
    encoder = CategoryEncoder().fit(_df([["Global Equity", "ESG"], ["Gold"]]))
    matrix = encoder.transform(_df([["Gold"], ["ESG", "Global Equity"]]))
    assert encoder.classes == ["ESG", "Global Equity", "Gold"]
    assert matrix.tolist() == [[0, 0, 1], [1, 1, 0]]


def test_unknown_category_is_reported(caplog):
    encoder = CategoryEncoder().fit(_df([["Gold"]]))
    with caplog.at_level(logging.WARNING):
        matrix = encoder.transform(_df([["Gold", "New category"]]))
    assert "New category" in caplog.text
    assert encoder.last_unknown == {"New category": 1}
    assert matrix.tolist() == [[1]]
