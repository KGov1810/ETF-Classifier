from datetime import date

import pytest

from etf_classifier.costs.pricing import PricingError, UnknownDeploymentError, cost_usd, load_pricing


def _write(tmp_path, text):
    path = tmp_path / "p.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_grid_is_loaded(pricing):
    assert pricing.currency == "USD"
    assert pricing.as_of == date(2026, 10, 1)
    assert pricing.model_for("my-embeddings").name == "embed-model"


def test_deployment_can_be_the_model_name_itself(pricing):
    assert pricing.model_for("chat-model").name == "chat-model"


def test_unknown_deployment_is_detected(pricing):
    assert pricing.find_model("inconnu") is None
    with pytest.raises(UnknownDeploymentError, match="inconnu"):
        pricing.model_for("inconnu")


def test_missing_input_price_raises(tmp_path):
    path = _write(tmp_path, 'currency: USD\nas_of: "2026-10-01"\nmodels:\n  m:\n    kind: embedding\n    tokenizer: cl100k_base\n')
    with pytest.raises(PricingError, match="input_per_million"):
        load_pricing(path)


def test_negative_price_raises(tmp_path):
    path = _write(tmp_path, 'currency: USD\nas_of: "2026-10-01"\nmodels:\n  m:\n    kind: embedding\n    tokenizer: cl100k_base\n    input_per_million: -1\n')
    with pytest.raises(PricingError, match="négatif"):
        load_pricing(path)


def test_chat_model_requires_an_output_price(tmp_path):
    path = _write(tmp_path, 'currency: USD\nas_of: "2026-10-01"\nmodels:\n  m:\n    kind: chat\n    tokenizer: o200k_base\n    input_per_million: 1\n')
    with pytest.raises(PricingError, match="output_per_million"):
        load_pricing(path)


def test_grid_age_in_days(pricing):
    assert pricing.age_days(today=date(2026, 10, 25)) == 24


def test_one_million_embedding_tokens_cost_0_13_dollar(pricing):
    assert cost_usd(pricing.model_for("my-embeddings"), input_tokens=1_000_000) == pytest.approx(0.13)


def test_chat_cost_separates_cached_and_output_tokens(pricing):
    model = pricing.model_for("my-chat")
    # 900 tokens d'entrée dont 400 en cache, 100 en sortie
    expected = (500 * 2.40 + 400 * 0.12 + 100 * 12.00) / 1_000_000
    assert cost_usd(model, input_tokens=900, cached_input_tokens=400, output_tokens=100) == pytest.approx(expected)


def test_long_context_rates_apply_above_the_threshold(pricing):
    model = pricing.model_for("my-chat")
    expected = (2000 * 4.80 + 10 * 18.00) / 1_000_000
    assert cost_usd(model, input_tokens=2000, output_tokens=10) == pytest.approx(expected)


def test_default_project_grid_is_valid():
    pricing = load_pricing()
    assert pricing.model_for("text-embedding-3-large").kind == "embedding"
    assert pricing.model_for("gpt-6.1-sol").kind == "chat"
