import pytest

from etf_classifier.data.loader import read_raw_rows
from etf_classifier.data.validation import EXCLUSION, WARNING, validate_rows


@pytest.fixture
def validated(sample_csv, settings):
    rows, bad_lines = read_raw_rows(sample_csv, settings)
    return validate_rows(rows, bad_lines, settings)


def _rules_for(log, row_id):
    return {issue.rule for issue in log.issues if issue.row_id == row_id}


def _row(settings, ticker="ABC", name="An ETF", description="Some description", category="Gold",
         composition="[{'NEM': 5.0}]", record=0) -> dict:
    cols = settings["csv"]["columns"]
    return {cols["ticker"]: ticker, cols["name"]: name, cols["description"]: description,
            cols["category"]: category, cols["composition"]: composition, "__line__": record + 2,
            "__record__": record}


def test_fixture_has_15_usable_rows_and_9_exclusions(validated):
    data, log = validated
    assert len(data) == 15
    assert len(log.exclusions()) == 9


@pytest.mark.parametrize(
    "row_id, rule",
    [
        (13, "unreadable_composition"),
        (14, "no_category"),
        (15, "negative_weight"),
        (16, "non_numeric_weight"),
        (19, "missing_ticker"),
        (20, "missing_name"),
        (21, "no_description_no_composition"),
        (22, "duplicate_etf_ticker"),
        (23, "duplicate_etf_ticker"),
    ],
)
def test_each_exclusion_rule_is_logged_with_its_row(validated, row_id, rule):
    data, log = validated
    assert rule in _rules_for(log, row_id)
    assert row_id not in set(data["row_id"])


@pytest.mark.parametrize(
    "row_id, rule",
    [
        (8, "empty_composition"),
        (9, "empty_description"),
        (10, "duplicate_ticker"),
        (11, "weight_sum_too_high"),
    ],
)
def test_each_warning_rule_keeps_the_row(validated, row_id, rule):
    data, log = validated
    assert rule in _rules_for(log, row_id)
    assert row_id in set(data["row_id"])


def test_two_etfs_may_share_a_name(validated):
    data, log = validated
    assert set(data.loc[data["name"] == "World Tracker A", "ticker"]) == {"IWDA", "IWDD"}
    assert not _rules_for(log, 12)


def test_ticker_is_normalised(validated):
    data, _ = validated
    assert data.set_index("row_id").loc[0, "ticker"] == "IWDA"


def test_classification_is_one_category(validated):
    data, _ = validated
    assert data.set_index("row_id").loc[6, "categories"] == ["Gold"]


def test_duplicate_ticker_ignores_spaces_and_case(settings):
    rows = [_row(settings, ticker=" abc ", record=0), _row(settings, ticker="ABC", record=1)]
    data, log = validate_rows(rows, [], settings)
    assert data.empty
    assert [issue.rule for issue in log.exclusions()] == ["duplicate_etf_ticker"] * 2


def test_a_row_excluded_for_another_reason_does_not_make_its_ticker_duplicate(settings):
    rows = [_row(settings, record=0), _row(settings, composition="[{'NEM'", record=1)]
    data, log = validate_rows(rows, [], settings)
    assert list(data["ticker"]) == ["ABC"]
    assert {issue.rule for issue in log.exclusions()} == {"unreadable_composition"}


def test_description_alone_is_enough(settings):
    data, log = validate_rows([_row(settings, composition="")], [], settings)
    assert len(data) == 1
    assert {issue.rule for issue in log.warnings()} == {"empty_composition"}


def test_composition_alone_is_enough(settings):
    data, log = validate_rows([_row(settings, description="  ")], [], settings)
    assert len(data) == 1
    assert {issue.rule for issue in log.warnings()} == {"empty_description"}


def test_category_is_optional_for_new_etfs(settings):
    data, log = validate_rows([_row(settings, category="")], [], settings, require_categories=False)
    assert data.iloc[0]["categories"] == [] and not log.exclusions()


def test_exclusions_and_warnings_have_the_right_severity(validated):
    _, log = validated
    assert all(issue.severity == EXCLUSION for issue in log.exclusions())
    assert all(issue.severity == WARNING for issue in log.warnings())


def test_logged_issue_carries_file_line_and_extract(validated):
    _, log = validated
    issue = next(issue for issue in log.issues if issue.rule == "unreadable_composition")
    assert issue.line == 15
    assert "AAPL" in issue.extract


def test_coverage_is_the_sum_of_weights(validated):
    data, _ = validated
    row = data.set_index("row_id").loc[0]
    assert row["coverage"] == pytest.approx(19.0)
    assert row["n_holdings"] == 5


def test_duplicate_holding_weights_are_summed(validated):
    data, _ = validated
    assert data.set_index("row_id").loc[10, "composition"]["AAPL"] == pytest.approx(5.0)


def test_empty_composition_has_zero_coverage(validated):
    data, _ = validated
    assert data.set_index("row_id").loc[8, "coverage"] == 0.0


def test_malformed_line_is_an_exclusion(settings):
    data, log = validate_rows([], [{"line": 7, "extract": "Broken;x;y;z;w"}], settings)
    assert data.empty
    assert log.exclusions()[0].rule == "malformed_line"
    assert log.exclusions()[0].line == 7


def test_log_counts_by_rule(validated):
    _, log = validated
    counts = log.counts()
    assert counts[("exclusion", "negative_weight")] == 1
    assert counts[("exclusion", "duplicate_etf_ticker")] == 2
