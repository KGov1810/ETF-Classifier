import pandas as pd
import pytest

from etf_classifier.data.normalize import (
    find_similar_tickers,
    load_aliases,
    normalize_composition,
    normalize_dataset,
    normalize_ticker,
)


def test_ticker_is_uppercased_and_trimmed():
    assert normalize_ticker(" aapl ") == "AAPL"


def test_inner_spaces_are_collapsed():
    assert normalize_ticker("AAPL   us") == "AAPL US"


def test_exchange_suffix_is_kept_by_default():
    assert normalize_ticker("AAPL US") == "AAPL US"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("AAPL US", "AAPL"),
        ("MC.PA", "MC"),
        ("AAPL UW Equity", "AAPL"),
        ("005930 KS", "005930"),
        ("BRK.B", "BRK.B"),  # classe d'action, pas une place de cotation
    ],
)
def test_exchange_suffix_is_removed_when_enabled(raw, expected):
    assert normalize_ticker(raw, strip_suffix=True) == expected


def test_aliases_are_loaded_and_normalized(tmp_path, settings):
    aliases_file = tmp_path / "aliases.csv"
    aliases_file.write_text("raw;canonical\ngoog ;GOOGL\n", encoding="utf-8")
    settings["normalize"]["aliases_path"] = str(aliases_file)
    assert load_aliases(settings) == {"GOOG": "GOOGL"}


def test_missing_aliases_file_gives_no_alias(tmp_path, settings):
    settings["normalize"]["aliases_path"] = str(tmp_path / "absent.csv")
    assert load_aliases(settings) == {}


def test_alias_merges_weights_on_the_canonical_ticker():
    assert normalize_composition({"GOOG": 1.0, "googl": 2.0}, {"GOOG": "GOOGL"}) == {"GOOGL": 3.0}


def test_normalize_dataset_keeps_original_tickers():
    data = pd.DataFrame({"composition": [{" aapl": 5.0}]})
    result = normalize_dataset(data, {}, strip_suffix=False)
    assert result.loc[0, "composition"] == {"AAPL": 5.0}
    assert result.loc[0, "composition_raw"] == {" aapl": 5.0}


def test_similar_tickers_share_a_root_with_different_suffixes():
    similar = find_similar_tickers(["AAPL", "AAPL US", "MSFT", "MC.PA", "MC FP"])
    assert similar == {"AAPL": ["AAPL", "AAPL US"], "MC": ["MC FP", "MC.PA"]}
