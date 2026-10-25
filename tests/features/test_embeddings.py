import csv
import logging

import numpy as np
import pytest

from etf_classifier.features.embeddings import create_embedder, embedding_text


def _log_rows(settings):
    from pathlib import Path

    path = Path(settings["costs"]["usage_log_path"])
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_150_texts_are_sent_in_three_ordered_batches(azure_settings, fake_client, fake_vector):
    embedder = create_embedder(azure_settings, client=fake_client)
    texts = [f"ETF number {i}" for i in range(150)]
    vectors = embedder.embed(texts)
    assert [len(call["input"]) for call in fake_client.calls] == [64, 64, 22]
    assert vectors.shape == (150, 8)
    for row in (0, 149):
        expected = np.array(fake_vector(texts[row], 8))
        assert np.allclose(vectors[row], expected / np.linalg.norm(expected), atol=1e-6)


def test_vectors_are_l2_normalised(azure_settings, fake_client):
    vectors = create_embedder(azure_settings, client=fake_client).embed(["a", "b", "c"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_duplicate_texts_are_computed_once(azure_settings, fake_client):
    create_embedder(azure_settings, client=fake_client).embed(["same", "same", "other"])
    assert sorted(fake_client.calls[0]["input"]) == ["other", "same"]


def test_second_pass_makes_no_api_call(azure_settings, fake_client):
    texts = [f"ETF {i}" for i in range(10)]
    create_embedder(azure_settings, client=fake_client).embed(texts)
    calls_after_first = len(fake_client.calls)
    create_embedder(azure_settings, client=fake_client).embed(texts)
    assert len(fake_client.calls) == calls_after_first


def test_changing_dimensions_does_not_reuse_cached_vectors(azure_settings, fake_client):
    create_embedder(azure_settings, client=fake_client).embed(["text"])
    azure_settings["features"]["embeddings"]["dimensions"] = 4
    vectors = create_embedder(azure_settings, client=fake_client).embed(["text"])
    assert len(fake_client.calls) == 2
    assert vectors.shape == (1, 4)


def test_one_cost_line_per_batch_sent_and_none_for_cached_texts(azure_settings, fake_client):
    embedder = create_embedder(azure_settings, client=fake_client)
    embedder.embed([f"ETF {i}" for i in range(100)])
    embedder.embed([f"ETF {i}" for i in range(100)])
    rows = _log_rows(azure_settings)
    assert len(rows) == 2
    assert {r["module"] for r in rows} == {"features"}
    assert all(float(r["cost_usd"]) > 0 for r in rows)
    assert embedder.n_api_calls == 2


def test_estimate_counts_only_uncached_texts_without_calling_azure(azure_settings, fake_client):
    embedder = create_embedder(azure_settings, client=fake_client)
    embedder.embed(["already computed"])
    plan = embedder.plan(["already computed", "new 1", "new 2"])
    assert (plan.n_total, plan.n_cached, plan.n_to_compute) == (3, 1, 2)
    assert plan.estimate.cost_usd > 0
    assert len(fake_client.calls) == 1


def test_long_text_is_truncated_with_a_warning(azure_settings, fake_client, caplog):
    azure_settings["features"]["embeddings"]["max_tokens"] = 10
    with caplog.at_level(logging.WARNING):
        create_embedder(azure_settings, client=fake_client).embed(["word " * 500])
    assert "truncated" in caplog.text
    assert len(fake_client.calls[0]["input"][0]) < len("word " * 500)


def test_without_azure_variables_no_embedder_is_created(azure_settings, no_azure_env, caplog):
    with caplog.at_level(logging.INFO):
        assert create_embedder(azure_settings) is None
    assert "AZURE_OPENAI" in caplog.text


def test_category_is_never_part_of_the_embedded_text():
    text = embedding_text("World ESG", "Tracks MSCI World ESG.", {"AAPL": 5.0})
    assert text == "World ESG. Tracks MSCI World ESG."


def test_top_tickers_option_adds_the_largest_holdings():
    text = embedding_text("X", "Y.", {"A": 1.0, "B": 9.0}, include_top_tickers=True)
    assert text.endswith("Top holdings: B, A.")
