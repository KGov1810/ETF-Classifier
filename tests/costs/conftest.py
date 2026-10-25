"""Test price list, independent of config/pricing.yaml."""

import textwrap

import pytest

from etf_classifier.costs.pricing import load_pricing

TEST_PRICING = textwrap.dedent(
    """
    currency: USD
    as_of: "2026-10-01"
    source: "test price list"
    models:
      embed-model:
        kind: embedding
        tokenizer: cl100k_base
        input_per_million: 0.13
      chat-model:
        kind: chat
        tokenizer: o200k_base
        tokenizer_exact: false
        input_per_million: 2.40
        cached_input_per_million: 0.12
        output_per_million: 12.00
        long_context_threshold: 1000
        long_context:
          input_per_million: 4.80
          cached_input_per_million: 0.24
          output_per_million: 18.00
    deployments:
      my-embeddings: embed-model
      my-chat: chat-model
    """
)


@pytest.fixture
def pricing_path(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(TEST_PRICING, encoding="utf-8")
    return path


@pytest.fixture
def pricing(pricing_path):
    return load_pricing(pricing_path)
