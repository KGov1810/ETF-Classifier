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
    path = tmp_path / "valides.csv"
    path.write_text(
        HEADER
        + "New Clean Energy;Clean energy;\"[{'ENPH': 6.0}]\";['Énergies propres'];[];False;v1;['Énergies propres']\n"
        + "New Hydrogen;Hydrogen;\"[{'PLUG': 5.0}]\";['Énergies propres'];[];True;v1;['Hydrogène']\n"
        + "Gold Miners;Gold;\"[{'NEM': 5.0}]\";['Or'];[];False;v1;['Or']\n"
        + "No Category;Desc;\"[{'AAPL': 5.0}]\";['Actions US'];[];False;v1;\n"
        + "Corrected;World;\"[{'AAPL': 5.0}]\";['Actions US'];[];False;v1;['Actions Monde', 'ESG']\n",
        encoding="utf-8",
    )
    return path


def test_valid_lines_are_recorded_with_the_deduced_decision(validated_csv, config, fb_settings):
    assert main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config]) == 0
    journal = FeedbackStore(fb_settings).read_journal().set_index("Name")
    assert set(journal.index) == {"New Clean Energy", "Corrected"}
    assert journal.loc["New Clean Energy", "Decision"] == "accepté"
    assert journal.loc["Corrected", "Decision"] == "corrigé"
    assert journal.loc["Corrected", "ModelVersion"] == "v1"


def test_refused_lines_are_listed_with_their_reason(validated_csv, config, capsys):
    main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config])
    output = capsys.readouterr().out
    assert "2 validation(s) enregistrée(s)" in output and "3 ligne(s) refusée(s)" in output
    for reason in ["Hydrogène", "déjà présent", "catégorie validée"]:
        assert reason in output


def test_create_categories_flag_creates_and_counts_them(validated_csv, config, capsys):
    main(["--csv", str(validated_csv), "--by", "Kevin", "--config", config, "--create-categories"])
    assert "1 catégorie(s) créée(s)" in capsys.readouterr().out


def test_missing_columns_give_an_error_code(tmp_path, config, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("Name;Category\nX;['Or']\n", encoding="utf-8")
    assert main(["--csv", str(path), "--by", "Kevin", "--config", config]) == 2
    assert "Composition" in capsys.readouterr().err
