import pytest

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.data.report import build_report, main


@pytest.fixture
def report(sample_csv, settings):
    return build_report(sample_csv, settings)


def test_report_counts_rows(report):
    assert report.stats["total_rows"] == 20
    assert report.stats["valid_rows"] == 15
    assert report.stats["excluded_rows"] == 5
    assert report.stats["rows_with_warnings"] == 5


def test_report_describes_categories(report):
    assert report.stats["n_categories"] == 11
    assert report.stats["mean_categories_per_etf"] == pytest.approx(1.4)


def test_report_describes_near_duplicate_groups(report):
    # World Tracker A/B sont regroupés, ainsi que World ESG/US Tech (ESG contenu dans US Tech)
    assert report.stats["n_groups"] == 13
    assert report.stats["largest_group_size"] == 2


@pytest.mark.parametrize(
    "section",
    [
        "## Résumé",
        "## Exclusions",
        "## Avertissements",
        "## Catégories",
        "## Couverture des compositions",
        "## Tickers les plus fréquents",
        "## Tickers ressemblants",
        "## Quasi-doublons",
        "## Découpage entraînement/test",
    ],
)
def test_report_contains_every_section(report, section):
    assert section in report.markdown


def test_report_shows_the_exclusion_reason_and_line(report):
    assert "Composition impossible à lire" in report.markdown
    assert "| 15 |" in report.markdown


def test_cli_writes_the_markdown_report(sample_csv, tmp_path, capsys):
    output = tmp_path / "rapport.md"
    exit_code = main(["--csv", str(sample_csv), "--output", str(output)])
    assert exit_code == 0
    assert output.read_text(encoding="utf-8").startswith("# Rapport de qualité")
    assert "Lignes exploitables" in capsys.readouterr().out


def test_cli_returns_an_error_code_for_a_missing_file(tmp_path, capsys):
    assert main(["--csv", str(tmp_path / "absent.csv"), "--output", str(tmp_path / "r.md")]) == 2
    assert "introuvable" in capsys.readouterr().err


def test_cli_returns_an_error_code_for_missing_columns(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("Name;Description\nA;B\n", encoding="utf-8")
    assert main(["--csv", str(path), "--output", str(tmp_path / "r.md")]) == 3
    assert "Composition" in capsys.readouterr().err


def test_prepare_dataset_returns_groups_and_split(sample_csv, settings):
    prepared = prepare_dataset(sample_csv, settings)
    assert "group_id" in prepared.data.columns
    assert len(prepared.split.train) + len(prepared.split.test) == 15
