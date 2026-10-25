import pytest

from etf_classifier.data.loader import read_raw_rows
from etf_classifier.data.validation import EXCLUSION, WARNING, validate_rows


@pytest.fixture
def validated(sample_csv, settings):
    rows, bad_lines = read_raw_rows(sample_csv, settings)
    return validate_rows(rows, bad_lines, settings)


def _rules_for(log, name_row_id):
    return {issue.rule for issue in log.issues if issue.row_id == name_row_id}


def test_fixture_has_15_usable_rows_and_5_exclusions(validated):
    data, log = validated
    assert len(data) == 15
    assert len(log.exclusions()) == 5


@pytest.mark.parametrize(
    "row_id, rule",
    [
        (13, "composition_illisible"),
        (14, "categories_illisibles"),
        (15, "aucune_categorie"),
        (16, "poids_negatif"),
        (17, "poids_non_numerique"),
    ],
)
def test_each_exclusion_rule_is_logged_with_its_row(validated, row_id, rule):
    data, log = validated
    assert rule in _rules_for(log, row_id)
    assert row_id not in set(data["row_id"])


@pytest.mark.parametrize(
    "row_id, rule",
    [
        (8, "composition_vide"),
        (9, "description_vide"),
        (10, "ticker_en_double"),
        (11, "somme_poids_elevee"),
        (12, "nom_en_double"),
    ],
)
def test_each_warning_rule_keeps_the_row(validated, row_id, rule):
    data, log = validated
    assert rule in _rules_for(log, row_id)
    assert row_id in set(data["row_id"])


def test_exclusions_and_warnings_have_the_right_severity(validated):
    _, log = validated
    assert all(issue.severity == EXCLUSION for issue in log.exclusions())
    assert all(issue.severity == WARNING for issue in log.warnings())


def test_logged_issue_carries_file_line_and_extract(validated):
    _, log = validated
    issue = next(issue for issue in log.issues if issue.rule == "composition_illisible")
    assert issue.line == 15
    assert "AAPL" in issue.extract


def test_coverage_is_the_sum_of_weights(validated):
    data, _ = validated
    row = data.set_index("row_id").loc[0]
    assert row["coverage"] == pytest.approx(19.0)
    assert row["n_holdings"] == 5


def test_duplicate_ticker_weights_are_summed(validated):
    data, _ = validated
    assert data.set_index("row_id").loc[10, "composition"]["AAPL"] == pytest.approx(5.0)


def test_empty_composition_has_zero_coverage(validated):
    data, _ = validated
    assert data.set_index("row_id").loc[8, "coverage"] == 0.0


def test_malformed_line_is_an_exclusion(settings):
    data, log = validate_rows([], [{"line": 7, "extract": "Broken;x;y;z;w"}], settings)
    assert data.empty
    assert log.exclusions()[0].rule == "ligne_mal_formee"
    assert log.exclusions()[0].line == 7


def test_log_counts_by_rule(validated):
    _, log = validated
    counts = log.counts()
    assert counts[("exclusion", "poids_negatif")] == 1
    assert counts[("avertissement", "nom_en_double")] == 1
