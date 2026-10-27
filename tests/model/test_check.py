import pytest

from etf_classifier.model.check import AGREEMENT, DISAGREEMENT, UNKNOWN, check_established
from etf_classifier.model.classifier import Prediction

CLASSES = ["Gold", "Technology", "Bonds"]
PREDICTION = Prediction(categories=["Gold"], scores=[("Gold", 0.7), ("Technology", 0.2), ("Bonds", 0.1)],
                        uncertain=False)


@pytest.mark.parametrize("established, verdict, score", [
    ("Gold", AGREEMENT, 0.7),
    ("Technology", DISAGREEMENT, 0.2),
    ("Crypto", UNKNOWN, None),
])
def test_established_classification_is_checked_against_the_proposal(established, verdict, score):
    check = check_established(PREDICTION, established, CLASSES)
    assert (check.verdict, check.established, check.established_score) == (verdict, established, score)


def test_no_established_classification_gives_no_verdict():
    check = check_established(PREDICTION, "", CLASSES)
    assert check.verdict is None and check.established_score is None
