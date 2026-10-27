import pytest
import yaml

from etf_classifier.feedback.importer import main
from etf_classifier.feedback.store import FeedbackStore

HEADER = "Ticker;Nom;Description;Composition;PredictedCategory;Scores;Uncertain;ModelVersion;Classification\n"


@pytest.fixture
def config(isolated_settings, tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(isolated_settings, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture
def validated_csv(tmp_path):
    path = tmp_path / "validated.csv"
    path.write_text(
        HEADER
        + "NCLE;New Clean Energy;Clean energy;\"[{'ENPH': 6.0}]\";Clean Energy;[];False;v1;Clean Energy\n"
        + "NHYD;New Hydrogen;Hydrogen;\"[{'PLUG': 5.0}]\";Clean Energy;[];True;v1;Hydrogen\n"
        + " ;No Ticker;Gold;\"[{'NEM': 5.0}]\";Gold;[];False;v1;Gold\n"
        + "NCAT;No Category;Desc;\"[{'AAPL': 5.0}]\";US Equity;[];False;v1;\n"
        + "CORR;Corrected;World;\"[{'AAPL': 5.0}]\";US Equity;[];False;v1;Global Equity\n",
        encoding="utf-8",
    )
    return path


def test_valid_lines_are_recorded_with_the_deduced_decision(validated_csv, config, isolated_settings):
    assert main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config]) == 0
    journal = FeedbackStore(isolated_settings).read_journal().set_index("Ticker")
    assert set(journal.index) == {"NCLE", "CORR"}
    assert journal.loc["NCLE", "Decision"] == "accepted"
    assert journal.loc["CORR", "Decision"] == "corrected"
    assert journal.loc["CORR", "ModelVersion"] == "v1"


def test_refused_lines_are_listed_with_their_reason(validated_csv, config, capsys):
    main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config])
    output = capsys.readouterr().out
    assert "2 validation(s) recorded" in output and "3 line(s) refused" in output
    for reason in ["Hydrogen", "ticker", "Classification"]:
        assert reason in output


def test_create_categories_flag_creates_and_counts_them(validated_csv, config, capsys):
    main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config, "--create-categories"])
    assert "1 category(ies) created" in capsys.readouterr().out


def test_missing_columns_give_an_error_code(tmp_path, config, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("Ticker;Nom;Classification\nX;Y;Gold\n", encoding="utf-8")
    assert main(["--csv", str(path), "--by", "Kevin", "--config", config]) == 2
    assert "Composition" in capsys.readouterr().err


def test_import_rows_can_be_used_without_the_command_line(store, isolated_settings):
    from etf_classifier.feedback.importer import import_rows

    rows = [(2, {"Ticker": "GLD1", "Nom": "Gold One", "Description": "d", "Composition": "[{'NEM': 5.0}]",
                 "Classification": "Gold", "PredictedCategory": "Gold"}),
            (3, {"Ticker": "NCAT", "Nom": "No Category", "Description": "d", "Composition": "[]",
                 "Classification": ""})]
    result = import_rows(rows, store, "Kevin", isolated_settings["csv"]["columns"])
    assert result.accepted == [(2, "GLD1", "accepted")]
    assert result.refused[0][:2] == (3, "NCAT")
    assert result.created == [] and result.error == ""


def test_import_stops_cleanly_when_the_journal_is_unavailable(store, monkeypatch, isolated_settings):
    from etf_classifier.feedback.importer import import_rows
    from etf_classifier.feedback.store import FeedbackError

    def locked(*_, **__):
        raise FeedbackError("Another record is being saved: try again in a few seconds.")

    monkeypatch.setattr(store, "record_validation", locked)
    rows = [(2, {"Ticker": "GLD1", "Nom": "Gold One", "Description": "d", "Composition": "[{'NEM': 5.0}]",
                 "Classification": "Gold"})]
    result = import_rows(rows, store, "Kevin", isolated_settings["csv"]["columns"])
    assert result.accepted == [] and "try again" in result.error
