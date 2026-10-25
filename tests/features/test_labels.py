import logging

import pandas as pd

from etf_classifier.features.labels import CategoryEncoder


def _df(categories):
    return pd.DataFrame({"categories": categories})


def test_train_and_test_matrices_share_the_same_columns():
    encoder = CategoryEncoder().fit(_df([["Actions Monde", "ESG"], ["Or"]]))
    matrix = encoder.transform(_df([["Or"], ["ESG", "Actions Monde"]]))
    assert encoder.classes == ["Actions Monde", "ESG", "Or"]
    assert matrix.tolist() == [[0, 0, 1], [1, 1, 0]]


def test_unknown_category_is_reported(caplog):
    encoder = CategoryEncoder().fit(_df([["Or"]]))
    with caplog.at_level(logging.WARNING):
        matrix = encoder.transform(_df([["Or", "Nouvelle catégorie"]]))
    assert "Nouvelle catégorie" in caplog.text
    assert encoder.last_unknown == {"Nouvelle catégorie": 1}
    assert matrix.tolist() == [[1]]
