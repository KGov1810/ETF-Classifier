"""Reading and checking the price list (config/pricing.yaml), cost computation.

No price is written in the code: everything comes from the dated and sourced price list.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from etf_classifier.config import load_settings, resolve_path

KINDS = {"embedding", "chat"}
PER_MILLION = 1_000_000


class PricingError(ValueError):
    """Invalid price list."""


class UnknownDeploymentError(PricingError):
    """Deployment missing from the price list."""


@dataclass(frozen=True)
class Rates:
    """Prices in USD per million tokens."""

    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class ModelPricing:
    name: str
    kind: str
    tokenizer: str
    tokenizer_exact: bool
    rates: Rates
    long_context_threshold: int | None = None
    long_context: Rates | None = None


@dataclass(frozen=True)
class PricingTable:
    currency: str
    as_of: date
    source: str
    models: dict[str, ModelPricing]
    deployments: dict[str, str]

    def find_model(self, deployment: str) -> ModelPricing | None:
        """Model associated with a deployment (or the model name itself), None if unknown."""
        return self.models.get(self.deployments.get(deployment, deployment))

    def model_for(self, deployment: str) -> ModelPricing:
        model = self.find_model(deployment)
        if model is None:
            raise UnknownDeploymentError(
                f"Deployment '{deployment}' missing from the price list: add it to config/pricing.yaml"
            )
        return model

    def age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.as_of).days


def _price(block: dict, key: str, label: str, default: float | None = None) -> float:
    value = block.get(key, default)
    if value is None:
        raise PricingError(f"{label}: price '{key}' missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PricingError(f"{label}: price '{key}' is not numeric ({value!r})")
    if value < 0:
        raise PricingError(f"{label}: price '{key}' is negative ({value})")
    return float(value)


def _rates(block: dict, kind: str, label: str) -> Rates:
    input_price = _price(block, "input_per_million", label)
    output_price = _price(block, "output_per_million", label) if kind == "chat" else 0.0
    cached = _price(block, "cached_input_per_million", label, default=input_price)
    return Rates(input=input_price, cached_input=cached, output=output_price)


def _model(name: str, block: dict) -> ModelPricing:
    kind = block.get("kind")
    if kind not in KINDS:
        raise PricingError(f"Model {name}: 'kind' must be one of {sorted(KINDS)}")
    if not block.get("tokenizer"):
        raise PricingError(f"Model {name}: 'tokenizer' missing")
    threshold, long_rates = block.get("long_context_threshold"), block.get("long_context")
    if (threshold is None) != (long_rates is None):
        raise PricingError(f"Model {name}: 'long_context_threshold' and 'long_context' go together")
    return ModelPricing(
        name=name,
        kind=kind,
        tokenizer=block["tokenizer"],
        tokenizer_exact=bool(block.get("tokenizer_exact", True)),
        rates=_rates(block, kind, f"Model {name}"),
        long_context_threshold=int(threshold) if threshold is not None else None,
        long_context=_rates(long_rates, kind, f"Model {name} (long context)") if long_rates else None,
    )


def load_pricing(path: str | Path | None = None) -> PricingTable:
    """Loads and checks the price list (by default the one set in settings.yaml)."""
    pricing_path = Path(path) if path else resolve_path(load_settings()["costs"]["pricing_path"])
    with pricing_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        as_of = date.fromisoformat(str(raw["as_of"]))
    except (KeyError, ValueError) as exc:
        raise PricingError("Price list: 'as_of' date missing or invalid (YYYY-MM-DD)") from exc
    models = {name: _model(name, block or {}) for name, block in (raw.get("models") or {}).items()}
    if not models:
        raise PricingError("Price list: no model declared")
    deployments = {str(k): str(v) for k, v in (raw.get("deployments") or {}).items()}
    for deployment, model in deployments.items():
        if model not in models:
            raise PricingError(f"Deployment {deployment}: model '{model}' missing from the price list")
    return PricingTable(raw.get("currency", "USD"), as_of, raw.get("source", ""), models, deployments)


def cost_usd(model: ModelPricing, input_tokens: int, cached_input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Cost of a call. Cached tokens are part of the input tokens (Azure convention)."""
    rates = model.rates
    if model.long_context_threshold is not None and input_tokens > model.long_context_threshold:
        rates = model.long_context
    cached = min(cached_input_tokens, input_tokens)
    billed = (input_tokens - cached) * rates.input + cached * rates.cached_input + output_tokens * rates.output
    return billed / PER_MILLION
