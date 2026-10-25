import pandas as pd
import pytest

from etf_classifier.data.split import make_split


def _dataset(n_rows: int = 100) -> pd.DataFrame:
    """Fictitious dataset: groups of 1 to 3 ETFs, 4 frequent categories and a rare one."""
    rows = []
    group = 0
    while len(rows) < n_rows:
        size = 1 + group % 3
        category = ["Global Equity", "European Equity", "Bonds", "Gold"][group % 4]
        for _ in range(size):
            rows.append({"name": f"ETF{len(rows)}", "categories": [category], "group_id": group})
        group += 1
    rows.append({"name": "Rare", "categories": ["Rare"], "group_id": group})
    return pd.DataFrame(rows[: n_rows] + rows[-1:])


def test_no_group_is_split_between_train_and_test():
    result = make_split(_dataset(), test_size=0.2, seed=1)
    assert set(result.train["group_id"]).isdisjoint(result.test["group_id"])


def test_every_row_ends_up_on_exactly_one_side():
    data = _dataset()
    result = make_split(data, test_size=0.2, seed=1)
    assert len(result.train) + len(result.test) == len(data)


def test_same_seed_gives_the_same_split():
    first = make_split(_dataset(), test_size=0.2, seed=7)
    second = make_split(_dataset(), test_size=0.2, seed=7)
    assert list(first.test["name"]) == list(second.test["name"])


def test_different_seeds_give_different_splits():
    first = make_split(_dataset(), test_size=0.2, seed=1)
    second = make_split(_dataset(), test_size=0.2, seed=2)
    assert list(first.test["name"]) != list(second.test["name"])


def test_test_share_is_close_to_the_target():
    result = make_split(_dataset(200), test_size=0.2, seed=3)
    assert result.summary.test_share == pytest.approx(0.2, abs=0.05)


def test_frequent_categories_appear_on_both_sides():
    result = make_split(_dataset(), test_size=0.2, seed=4)
    for category in ["Global Equity", "European Equity", "Bonds", "Gold"]:
        assert any(category in cats for cats in result.test["categories"])
        assert any(category in cats for cats in result.train["categories"])


def test_single_group_category_is_reported_as_missing_on_one_side():
    result = make_split(_dataset(), test_size=0.2, seed=5)
    missing = set(result.summary.missing_in_test) | set(result.summary.missing_in_train)
    assert "Rare" in missing


def test_rare_categories_are_listed_with_their_count():
    result = make_split(_dataset(), test_size=0.2, seed=5, rare_min=5)
    assert result.summary.rare_categories == {"Rare": 1}


def test_split_requires_group_ids():
    with pytest.raises(ValueError, match="group_id"):
        make_split(pd.DataFrame({"categories": [["A"]]}))
