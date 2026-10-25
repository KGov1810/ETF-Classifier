import pytest

from etf_classifier.data.parsing import (
    ParsingError,
    WeightError,
    parse_categories,
    parse_composition,
    parse_composition_items,
)


def test_composition_list_of_single_key_dicts_becomes_one_dict():
    assert parse_composition("[{'AAPL': 6.5}, {'MSFT': 6.1}]") == {"AAPL": 6.5, "MSFT": 6.1}


def test_composition_duplicate_ticker_weights_are_summed():
    assert parse_composition("[{'AAPL': 3.0}, {'AAPL': 2.0}, {'MSFT': 4}]") == {"AAPL": 5.0, "MSFT": 4.0}


def test_composition_items_keep_duplicates_for_quality_checks():
    items = parse_composition_items("[{'AAPL': 3.0}, {'AAPL': 2.0}]")
    assert items == [("AAPL", 3.0), ("AAPL", 2.0)]


def test_composition_accepts_a_plain_dict():
    assert parse_composition("{'AAPL': 6.5, 'MSFT': 6.1}") == {"AAPL": 6.5, "MSFT": 6.1}


@pytest.mark.parametrize("raw", ["", "   ", "[]"])
def test_empty_composition_gives_empty_dict(raw):
    assert parse_composition(raw) == {}


def test_unreadable_composition_raises_with_an_extract():
    with pytest.raises(ParsingError, match=r"\[\{'AAPL'"):
        parse_composition("[{'AAPL': 5.0}, {'MSFT'")


def test_composition_that_is_not_a_list_of_dicts_raises():
    with pytest.raises(ParsingError):
        parse_composition("['AAPL', 'MSFT']")


def test_text_weight_raises_weight_error():
    with pytest.raises(WeightError):
        parse_composition("[{'AAPL': 'cinq'}]")


def test_boolean_weight_is_not_numeric():
    with pytest.raises(WeightError):
        parse_composition("[{'AAPL': True}]")


def test_composition_never_evaluates_code():
    with pytest.raises(ParsingError):
        parse_composition("__import__('os').system('echo pwned')")


def test_categories_are_stripped_and_deduplicated_in_order():
    assert parse_categories("[' Actions Monde ', 'ESG', 'Actions Monde']") == ["Actions Monde", "ESG"]


def test_empty_category_list_gives_empty_list():
    assert parse_categories("[]") == []


def test_unreadable_categories_raise():
    with pytest.raises(ParsingError):
        parse_categories("['Actions US'")


def test_categories_must_be_strings():
    with pytest.raises(ParsingError):
        parse_categories("[1, 2]")
