import pandas as pd

from etf_classifier.app import services
from etf_classifier.data.loader import load_dataset, load_records, read_raw_rows


def _table(rows):
    return pd.DataFrame(rows, columns=["Ticker", "Weight (%)"])


def test_load_records_gives_the_same_dataset_as_load_dataset(sample_csv, settings):
    settings = dict(settings)
    settings.pop("feedback")  # no journal merge: compare the CSV path only
    rows, bad_lines = read_raw_rows(sample_csv, settings)
    from_records = load_records(rows, settings, bad_lines=bad_lines)
    from_csv = load_dataset(sample_csv, settings)
    pd.testing.assert_frame_equal(from_records.data, from_csv.data)
    assert len(from_records.log.issues) == len(from_csv.log.issues)


def test_typed_etf_is_normalised_like_a_csv_row(app_settings):
    checked = services.etf_from_inputs("My World", "Tracks the MSCI World index",
                                       _table([[" aapl ", 5.0], ["MSFT", "4.5"]]), app_settings)
    assert checked.errors == []
    assert checked.data.loc[0, "composition"] == {"AAPL": 5.0, "MSFT": 4.5}
    assert checked.data.loc[0, "coverage"] == 9.5


def test_negative_weight_is_refused_with_its_reason(app_settings):
    checked = services.etf_from_inputs("Bad", "desc", _table([["AAPL", -5.0]]), app_settings)
    assert checked.data.empty
    assert any("Negative weight" in error for error in checked.errors)


def test_non_numeric_weight_is_refused(app_settings):
    checked = services.etf_from_inputs("Bad", "desc", _table([["AAPL", "five"]]), app_settings)
    assert checked.data.empty
    assert any("Non-numeric weight" in error for error in checked.errors)


def test_empty_table_rows_are_ignored(app_settings):
    checked = services.etf_from_inputs("Gold", "Gold miners",
                                       _table([["NEM", 12.0], [None, None], ["", ""]]), app_settings)
    assert checked.errors == []
    assert checked.data.loc[0, "composition"] == {"NEM": 12.0}


def test_weight_without_ticker_is_an_error(app_settings):
    checked = services.etf_from_inputs("Gold", "Gold miners", _table([["NEM", 12.0], ["", 3.0]]), app_settings)
    assert checked.data.empty
    assert any("ticker" in error.lower() for error in checked.errors)


def test_duplicate_ticker_is_summed_with_a_warning(app_settings):
    checked = services.etf_from_inputs("Dup", "desc", _table([["AAPL", 3.0], ["aapl", 2.0]]), app_settings)
    assert checked.data.loc[0, "composition"] == {"AAPL": 5.0}
    assert any("several times" in warning for warning in checked.warnings)


def test_empty_name_is_an_error(app_settings):
    checked = services.etf_from_inputs("  ", "desc", _table([["AAPL", 3.0]]), app_settings)
    assert checked.data.empty
    assert any("name" in error.lower() for error in checked.errors)


def test_empty_composition_is_accepted_with_a_warning(app_settings):
    checked = services.etf_from_inputs("Physical Gold", "Physical gold bullion", _table([]), app_settings)
    assert checked.errors == []
    assert checked.data.loc[0, "composition"] == {}
    assert checked.warnings


def test_validators_are_read_from_the_configuration(app_settings):
    assert services.validators(app_settings) == ["Alice Martin", "Bruno Leroy", "Chloe Dubois"]


def test_weights_pasted_as_text_with_a_comma_are_converted(app_settings):
    checked = services.etf_from_inputs("Pasted", "desc", _table([["AAPL", "4,5"]]), app_settings)
    assert checked.data.loc[0, "composition"] == {"AAPL": 4.5}
