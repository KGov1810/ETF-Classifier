import pytest

from etf_classifier.data.loader import MissingColumnsError, load_dataset, read_raw_rows


def test_raw_rows_are_read_with_semicolon_and_utf8(sample_csv, settings):
    rows, bad_lines = read_raw_rows(sample_csv, settings)
    assert len(rows) == 20
    assert bad_lines == []
    assert rows[18]["Category"] == "['Actions Émergents']"


def test_raw_rows_keep_the_file_line_number(sample_csv, settings):
    rows, _ = read_raw_rows(sample_csv, settings)
    assert rows[0]["__line__"] == 2  # ligne 1 = en-tête


def test_missing_column_raises_a_clear_error(tmp_path, settings):
    path = tmp_path / "bad.csv"
    path.write_text("Name;Description;Composition\nA;B;[]\n", encoding="utf-8")
    with pytest.raises(MissingColumnsError, match="Category"):
        read_raw_rows(path, settings)


def test_line_with_wrong_number_of_fields_is_reported(tmp_path, settings):
    path = tmp_path / "bad.csv"
    path.write_text(
        "Name;Description;Composition;Category\n"
        "Ok;Desc;[];['Actions US']\n"
        "Broken;Desc avec ; non protégé;[];['Actions US']\n",
        encoding="utf-8",
    )
    rows, bad_lines = read_raw_rows(path, settings)
    assert len(rows) == 1
    assert bad_lines[0]["line"] == 3


def test_load_dataset_returns_parsed_columns(sample_csv, settings):
    result = load_dataset(sample_csv, settings)
    first = result.data.iloc[0]
    assert first["name"] == "World Tracker A"
    assert first["composition"] == {"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0, "GOOGL": 2.5}
    assert first["categories"] == ["Actions Monde"]
    assert first["row_id"] == 0
