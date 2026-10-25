import pytest

from etf_classifier.costs.estimate import estimate_chat, estimate_embeddings
from etf_classifier.costs.pricing import UnknownDeploymentError
from etf_classifier.costs.tokens import TokenCounter, get_counter


def test_fallback_counter_approximates_four_characters_per_token():
    counter = TokenCounter("encodage-inexistant")
    assert counter.exact is False
    assert counter.count("x" * 40) == 10


def test_fallback_truncation_is_conservative():
    counter = TokenCounter("encodage-inexistant")
    text, truncated = counter.truncate("y" * 1000, max_tokens=100)
    assert truncated is True
    assert counter.count(text) <= 100


def test_short_text_is_not_truncated():
    text, truncated = TokenCounter("encodage-inexistant").truncate("court", max_tokens=100)
    assert (text, truncated) == ("court", False)


def test_exact_counter_when_tiktoken_tables_are_available():
    counter = get_counter("cl100k_base")
    if not counter.exact:
        pytest.skip("Tables tiktoken non téléchargeables ici (réseau restreint)")
    assert counter.count("hello world") == 2


def test_embedding_estimate_sums_tokens_and_cost(pricing):
    estimate = estimate_embeddings(["a" * 400] * 10, "my-embeddings", pricing)
    assert estimate.n_items == 10
    assert estimate.input_tokens > 0
    assert estimate.cost_usd == pytest.approx(estimate.input_tokens * 0.13 / 1_000_000)
    assert estimate.pricing_as_of == "2026-10-01"


def test_empty_batch_costs_nothing(pricing):
    estimate = estimate_embeddings([], "my-embeddings", pricing)
    assert (estimate.input_tokens, estimate.cost_usd) == (0, 0.0)


def test_chat_estimate_is_flagged_approximate(pricing):
    estimate = estimate_chat([{"role": "user", "content": "Classify this ETF."}], "my-chat", pricing,
                             expected_output_tokens=200)
    assert estimate.approximate is True
    assert estimate.output_tokens == 200
    assert estimate.cost_usd > 200 * 12.00 / 1_000_000


def test_estimate_for_unknown_deployment_raises(pricing):
    with pytest.raises(UnknownDeploymentError):
        estimate_embeddings(["texte"], "inconnu", pricing)
