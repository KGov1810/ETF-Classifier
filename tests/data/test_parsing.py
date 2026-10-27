import pytest

from etf_classifier.data.parsing import (
    ParsingError,
    WeightError,
    normalize_etf_ticker,
    parse_category,
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


def test_category_is_one_plain_text_value_stripped():
    assert parse_category("  Global Equity ") == "Global Equity"


def test_empty_category_gives_empty_text():
    assert parse_category("") == "" and parse_category(None) == ""


@pytest.mark.parametrize("raw, expected", [(" iwda ", "IWDA"), ("IWDA  LN", "IWDA LN"), ("", ""), (None, "")])
def test_etf_ticker_is_trimmed_and_upper_cased(raw, expected):
    assert normalize_etf_ticker(raw) == expected


def test_format_composition_is_the_inverse_of_parsing():
    import numpy as np

    from etf_classifier.data.parsing import format_composition, parse_composition

    text = format_composition({"AAPL": np.float64(6.5), "MSFT": 6})
    assert text == "[{'AAPL': 6.5}, {'MSFT': 6.0}]"
    assert parse_composition(text) == {"AAPL": 6.5, "MSFT": 6.0}


def test_format_composition_keeps_text_weights_for_the_quality_rules():
    from etf_classifier.data.parsing import format_composition

    assert format_composition([("AAPL", "five")]) == "[{'AAPL': 'five'}]"
    assert format_composition("[{'AAPL': 1.0}] ") == "[{'AAPL': 1.0}]"
