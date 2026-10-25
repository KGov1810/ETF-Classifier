"""Estimation du coût avant un appel Azure."""

from dataclasses import dataclass

from etf_classifier.costs.pricing import PricingTable, cost_usd
from etf_classifier.costs.tokens import get_counter

TOKENS_PER_MESSAGE = 4  # enveloppe approximative d'un message de chat (rôle, séparateurs)


@dataclass(frozen=True)
class CostEstimate:
    deployment: str
    n_items: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    approximate: bool
    pricing_as_of: str


def estimate_embeddings(texts: list[str], deployment: str, pricing: PricingTable) -> CostEstimate:
    """Tokens et coût pour calculer les embeddings d'un lot de textes."""
    model = pricing.model_for(deployment)
    counter = get_counter(model.tokenizer)
    tokens = sum(counter.count(text) for text in texts)
    return CostEstimate(
        deployment=deployment,
        n_items=len(texts),
        input_tokens=tokens,
        output_tokens=0,
        cost_usd=cost_usd(model, input_tokens=tokens),
        approximate=not (counter.exact and model.tokenizer_exact),
        pricing_as_of=pricing.as_of.isoformat(),
    )


def estimate_chat(
    messages: list[dict] | str, deployment: str, pricing: PricingTable, expected_output_tokens: int
) -> CostEstimate:
    """Coût d'un appel de chat : entrée comptée, sortie selon une hypothèse explicite."""
    model = pricing.model_for(deployment)
    counter = get_counter(model.tokenizer)
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    input_tokens = sum(counter.count(str(m.get("content", ""))) + TOKENS_PER_MESSAGE for m in messages)
    return CostEstimate(
        deployment=deployment,
        n_items=1,
        input_tokens=input_tokens,
        output_tokens=expected_output_tokens,
        cost_usd=cost_usd(model, input_tokens=input_tokens, output_tokens=expected_output_tokens),
        approximate=True,  # la sortie est une hypothèse, et l'enveloppe des messages est approximative
        pricing_as_of=pricing.as_of.isoformat(),
    )
