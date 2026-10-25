from datetime import date, datetime, timezone

import pytest

from etf_classifier.costs.report import build_cost_report, main
from etf_classifier.costs.tracker import CostTracker, Usage


@pytest.fixture
def usage_log(pricing, tmp_path):
    """Fictitious log: 3 calls over 2 days, 2 modules."""
    path = tmp_path / "usage.csv"
    moments = iter([
        datetime(2026, 10, 24, 9, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 25, 9, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc),
    ])
    tracker = CostTracker(pricing, path, now=lambda: next(moments))
    tracker.record("features", "embeddings_batch", "my-embeddings", Usage(input_tokens=1_000_000),
                   estimated_cost_usd=0.12)
    tracker.record("features", "embeddings_batch", "my-embeddings", Usage(input_tokens=2_000_000))
    tracker.record("llm-justify", "justify", "my-chat", Usage(input_tokens=500, output_tokens=100))
    return path


def test_total_and_breakdown_by_module(usage_log, pricing):
    report = build_cost_report(usage_log, pricing, today=date(2026, 10, 25))
    chat_cost = (500 * 2.40 + 100 * 12.00) / 1_000_000
    assert report.stats["total_usd"] == pytest.approx(0.39 + chat_cost)
    assert report.stats["by_module"]["features"] == pytest.approx(0.39)
    assert report.stats["n_calls"] == 3


def test_breakdown_by_day(usage_log, pricing):
    report = build_cost_report(usage_log, pricing, today=date(2026, 10, 25))
    assert report.stats["by_day"]["2026-10-24"] == pytest.approx(0.13)


def test_period_filter(usage_log, pricing):
    report = build_cost_report(usage_log, pricing, since=date(2026, 10, 25), today=date(2026, 10, 25))
    assert report.stats["n_calls"] == 2


def test_estimate_versus_actual_gap(usage_log, pricing):
    report = build_cost_report(usage_log, pricing, today=date(2026, 10, 25))
    assert report.stats["estimate_gap_pct"] == pytest.approx((0.13 - 0.12) / 0.12 * 100)


def test_stale_grid_warning(usage_log, pricing):
    report = build_cost_report(usage_log, pricing, today=date(2027, 3, 1))
    assert "price list" in report.markdown.lower() and "⚠️" in report.markdown


def test_missing_log_gives_a_clear_message(tmp_path, pricing):
    report = build_cost_report(tmp_path / "absent.csv", pricing, today=date(2026, 10, 25))
    assert report.stats["n_calls"] == 0
    assert "No Azure call" in report.markdown


def test_cli_writes_the_report(usage_log, pricing_path, tmp_path, capsys):
    output = tmp_path / "costs.md"
    code = main(["--log", str(usage_log), "--pricing", str(pricing_path), "--output", str(output)])
    assert code == 0
    assert output.read_text(encoding="utf-8").startswith("# Azure OpenAI costs")
