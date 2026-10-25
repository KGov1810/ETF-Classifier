import csv
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from etf_classifier.costs.tracker import LOG_COLUMNS, CostTracker, Usage

FIXED_NOW = datetime(2026, 10, 25, 14, 3, 11, tzinfo=timezone.utc)


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


@pytest.fixture
def tracker(pricing, tmp_path):
    return CostTracker(pricing, tmp_path / "logs" / "usage.csv", now=lambda: FIXED_NOW)


def test_record_appends_one_line_with_all_columns(tracker):
    tracker.record("features", "embeddings_batch", "my-embeddings", Usage(input_tokens=1_000_000), note="lot 1")
    rows = _rows(tracker.log_path)
    assert len(rows) == 1
    assert list(rows[0]) == LOG_COLUMNS
    assert rows[0]["timestamp_utc"] == "2026-10-25T14:03:11Z"
    assert float(rows[0]["cost_usd"]) == pytest.approx(0.13)
    assert rows[0]["pricing_as_of"] == "2026-10-01"


def test_successive_records_keep_a_single_header(tracker):
    for _ in range(3):
        tracker.record("features", "embeddings_batch", "my-embeddings", Usage(input_tokens=10))
    assert len(_rows(tracker.log_path)) == 3


def test_unknown_deployment_is_logged_without_cost(tracker, caplog):
    record = tracker.record("features", "embeddings_batch", "inconnu", Usage(input_tokens=500))
    row = _rows(tracker.log_path)[0]
    assert record["cost_usd"] is None
    assert row["cost_usd"] == ""
    assert row["input_tokens"] == "500"
    assert "inconnu" in caplog.text


def test_estimate_is_stored_next_to_the_actual_cost(tracker):
    tracker.record("features", "embeddings_batch", "my-embeddings", Usage(input_tokens=100), estimated_cost_usd=0.00002)
    assert float(_rows(tracker.log_path)[0]["estimated_cost_usd"]) == pytest.approx(0.00002)


def test_usage_from_an_azure_embeddings_response():
    usage = Usage.from_openai(SimpleNamespace(prompt_tokens=120, total_tokens=120))
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) == (120, 0, 0)


def test_usage_from_an_azure_chat_response_with_cache():
    raw = SimpleNamespace(prompt_tokens=900, completion_tokens=50,
                          prompt_tokens_details=SimpleNamespace(cached_tokens=400))
    usage = Usage.from_openai(raw)
    assert (usage.input_tokens, usage.output_tokens, usage.cached_input_tokens) == (900, 50, 400)


def test_usage_from_a_plain_dict():
    usage = Usage.from_openai({"prompt_tokens": 10, "completion_tokens": 5})
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)


def test_text_content_is_never_logged(tracker):
    tracker.record("features", "embeddings_batch", "my-embeddings", Usage(input_tokens=10), note="3 textes")
    content = tracker.log_path.read_text(encoding="utf-8")
    assert "3 textes" in content and "Description" not in content
