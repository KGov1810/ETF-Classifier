import pytest
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


def test_typed_etf_is_normalised_like_a_csv_row(isolated_settings):
    checked = services.etf_from_inputs("My World", "Tracks the MSCI World index",
                                       _table([[" aapl ", 5.0], ["MSFT", "4.5"]]), isolated_settings)
    assert checked.errors == []
    assert checked.data.loc[0, "composition"] == {"AAPL": 5.0, "MSFT": 4.5}
    assert checked.data.loc[0, "coverage"] == 9.5


def test_negative_weight_is_refused_with_its_reason(isolated_settings):
    checked = services.etf_from_inputs("Bad", "desc", _table([["AAPL", -5.0]]), isolated_settings)
    assert checked.data.empty
    assert any("Negative weight" in error for error in checked.errors)


def test_non_numeric_weight_is_refused(isolated_settings):
    checked = services.etf_from_inputs("Bad", "desc", _table([["AAPL", "five"]]), isolated_settings)
    assert checked.data.empty
    assert any("Non-numeric weight" in error for error in checked.errors)


def test_empty_table_rows_are_ignored(isolated_settings):
    checked = services.etf_from_inputs("Gold", "Gold miners",
                                       _table([["NEM", 12.0], [None, None], ["", ""]]), isolated_settings)
    assert checked.errors == []
    assert checked.data.loc[0, "composition"] == {"NEM": 12.0}


def test_weight_without_ticker_is_an_error(isolated_settings):
    checked = services.etf_from_inputs("Gold", "Gold miners", _table([["NEM", 12.0], ["", 3.0]]), isolated_settings)
    assert checked.data.empty
    assert any("ticker" in error.lower() for error in checked.errors)


def test_duplicate_ticker_is_summed_with_a_warning(isolated_settings):
    checked = services.etf_from_inputs("Dup", "desc", _table([["AAPL", 3.0], ["aapl", 2.0]]), isolated_settings)
    assert checked.data.loc[0, "composition"] == {"AAPL": 5.0}
    assert any("several times" in warning for warning in checked.warnings)


def test_empty_name_is_an_error(isolated_settings):
    checked = services.etf_from_inputs("  ", "desc", _table([["AAPL", 3.0]]), isolated_settings)
    assert checked.data.empty
    assert any("name" in error.lower() for error in checked.errors)


def test_empty_composition_is_accepted_with_a_warning(isolated_settings):
    checked = services.etf_from_inputs("Physical Gold", "Physical gold bullion", _table([]), isolated_settings)
    assert checked.errors == []
    assert checked.data.loc[0, "composition"] == {}
    assert checked.warnings


def test_validators_are_read_from_the_configuration(isolated_settings):
    assert services.validators(isolated_settings) == ["Alice Martin", "Bruno Leroy", "Chloe Dubois"]


def test_weights_pasted_as_text_with_a_comma_are_converted(isolated_settings):
    checked = services.etf_from_inputs("Pasted", "desc", _table([["AAPL", "4,5"]]), isolated_settings)
    assert checked.data.loc[0, "composition"] == {"AAPL": 4.5}


def _batch_csv(sample_csv) -> bytes:
    """The fixture without its Category column, as a team member would upload it."""
    frame = pd.read_csv(sample_csv, sep=";", dtype=str, keep_default_na=False)
    return frame.drop(columns=["Category"]).to_csv(sep=";", index=False).encode("utf-8")


def test_batch_upload_lists_usable_etfs_and_excluded_rows(isolated_settings, sample_csv):
    batch = services.load_batch(_batch_csv(sample_csv), isolated_settings)
    assert "Gold Miners" in set(batch.data["name"])
    reasons = {item["name"]: item["reason"] for item in batch.exclusions}
    assert "Negative weight" in reasons["Negative Weight"]
    assert "Unreadable composition" in reasons["Broken Composition"]
    assert "Text Weight" not in set(batch.data["name"])


def test_batch_upload_with_missing_columns_gives_a_clear_error(isolated_settings):
    with pytest.raises(services.AppError, match="Composition"):
        services.load_batch(b"Name;Description\nA;B\n", isolated_settings)


def test_batch_upload_accepts_a_category_column(isolated_settings, sample_csv):
    batch = services.load_batch(sample_csv.read_bytes(), isolated_settings)
    assert "Gold Miners" in set(batch.data["name"])


def test_queue_order_puts_uncertain_etfs_first():
    class P:  # minimal stand-in for a Proposal
        def __init__(self, uncertain):
            self.uncertain = uncertain
    proposals = [P(False), P(True), P(False), P(True)]
    assert services.queue_order(proposals, uncertain_first=True) == [1, 3, 0, 2]
    assert services.queue_order(proposals, uncertain_first=False) == [0, 1, 2, 3]
