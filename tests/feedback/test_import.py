import pytest
import yaml

from etf_classifier.feedback.importer import main
from etf_classifier.feedback.store import FeedbackStore

HEADER = "Name;Description;Composition;PredictedCategories;Scores;Uncertain;ModelVersion;Category\n"


@pytest.fixture
def config(fb_settings, tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(fb_settings, allow_unicode=True), encoding="utf-8")
    return str(path)


@pytest.fixture
def validated_csv(tmp_path):
    path = tmp_path / "validated.csv"
    path.write_text(
        HEADER
        + "New Clean Energy;Clean energy;\"[{'ENPH': 6.0}]\";['Clean Energy'];[];False;v1;['Clean Energy']\n"
        + "New Hydrogen;Hydrogen;\"[{'PLUG': 5.0}]\";['Clean Energy'];[];True;v1;['Hydrogen']\n"
        + "Gold Miners;Gold;\"[{'NEM': 5.0}]\";['Gold'];[];False;v1;['Gold']\n"
        + "No Category;Desc;\"[{'AAPL': 5.0}]\";['US Equity'];[];False;v1;\n"
        + "Corrected;World;\"[{'AAPL': 5.0}]\";['US Equity'];[];False;v1;['Global Equity', 'ESG']\n",
        encoding="utf-8",
    )
    return path


def test_valid_lines_are_recorded_with_the_deduced_decision(validated_csv, config, fb_settings):
    assert main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config]) == 0
    journal = FeedbackStore(fb_settings).read_journal().set_index("Name")
    assert set(journal.index) == {"New Clean Energy", "Corrected"}
    assert journal.loc["New Clean Energy", "Decision"] == "accepted"
    assert journal.loc["Corrected", "Decision"] == "corrected"
    assert journal.loc["Corrected", "ModelVersion"] == "v1"


def test_refused_lines_are_listed_with_their_reason(validated_csv, config, capsys):
    main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config])
    output = capsys.readouterr().out
    assert "2 validation(s) recorded" in output and "3 line(s) refused" in output
    for reason in ["Hydrogen", "already present", "validated category"]:
        assert reason in output


def test_create_categories_flag_creates_and_counts_them(validated_csv, config, capsys):
    main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config, "--create-categories"])
    assert "1 category(ies) created" in capsys.readouterr().out


def test_missing_columns_give_an_error_code(tmp_path, config, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("Name;Category\nX;['Gold']\n", encoding="utf-8")
    assert main(["--csv", str(path), "--by", "Kevin", "--config", config]) == 2
    assert "Composition" in capsys.readouterr().err
