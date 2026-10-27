import pytest

from etf_classifier.data.loader import MissingColumnsError, load_dataset, read_raw_rows


def test_raw_rows_are_read_with_semicolon_and_utf8(sample_csv, settings):
    rows, bad_lines = read_raw_rows(sample_csv, settings)
    assert len(rows) == 24
    assert bad_lines == []
    assert rows[17]["Classification"] == "Emerging Markets Equity"


def test_raw_rows_keep_the_file_line_number(sample_csv, settings):
    rows, _ = read_raw_rows(sample_csv, settings)
    assert rows[0]["__line__"] == 2  # line 1 = header


def test_missing_column_raises_a_clear_error(tmp_path, settings):
    path = tmp_path / "bad.csv"
    path.write_text("Ticker;Nom;Description;Composition\nA;B;C;[]\n", encoding="utf-8")
    with pytest.raises(MissingColumnsError, match="Classification"):
        read_raw_rows(path, settings)


def test_line_with_wrong_number_of_fields_is_reported(tmp_path, settings):
    path = tmp_path / "bad.csv"
    path.write_text(
        "Ticker;Nom;Description;Classification;Composition\n"
        "OK;Ok;Desc;US Equity;[]\n"
        "BRK;Broken;Desc with ; unquoted;US Equity;[]\n",
        encoding="utf-8",
    )
    rows, bad_lines = read_raw_rows(path, settings)
    assert len(rows) == 1
    assert bad_lines[0]["line"] == 3


def test_load_dataset_returns_parsed_columns(sample_csv, settings):
    result = load_dataset(sample_csv, settings)
    first = result.data.iloc[0]
    assert first["ticker"] == "IWDA"
    assert first["name"] == "World Tracker A"
    assert first["composition"] == {"AAPL": 5.0, "MSFT": 4.5, "NVDA": 4.0, "AMZN": 3.0, "GOOGL": 2.5}
    assert first["categories"] == ["Global Equity"]
    assert first["row_id"] == 0


def test_csv_without_category_column_loads_when_categories_are_not_required(tmp_path, settings):
    path = tmp_path / "new.csv"
    path.write_text("Ticker;Nom;Description;Composition\nNEW;New ETF;Clean energy;\"[{'ENPH': 5.0}]\"\n",
                    encoding="utf-8")
    result = load_dataset(path, settings, require_categories=False)
    assert len(result.data) == 1
    assert result.data.iloc[0]["categories"] == []


def test_empty_category_is_kept_when_categories_are_not_required(tmp_path, settings):
    path = tmp_path / "new.csv"
    path.write_text("Ticker;Nom;Description;Classification;Composition\nNEW;New ETF;Desc;;[]\n", encoding="utf-8")
    result = load_dataset(path, settings, require_categories=False)
    assert len(result.data) == 1 and not result.log.exclusions()


def test_category_column_is_still_required_by_default(tmp_path, settings):
    path = tmp_path / "new.csv"
    path.write_text("Ticker;Nom;Description;Composition\nNEW;New ETF;Desc;[]\n", encoding="utf-8")
    with pytest.raises(MissingColumnsError):
        load_dataset(path, settings)


def test_column_order_does_not_matter(tmp_path, settings):
    path = tmp_path / "reordered.csv"
    path.write_text("Composition;Classification;Nom;Ticker;Description\n"
                    "\"[{'NEM': 5.0}]\";Gold;Gold ETF;gld;Gold\n", encoding="utf-8")
    row = load_dataset(path, settings).data.iloc[0]
    assert (row["ticker"], row["name"], row["categories"]) == ("GLD", "Gold ETF", ["Gold"])
