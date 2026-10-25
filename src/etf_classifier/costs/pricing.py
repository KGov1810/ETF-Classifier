"""Lecture et contrôle de la grille tarifaire (config/pricing.yaml), calcul des coûts.

Aucun prix n'est écrit dans le code : tout vient de la grille, datée et sourcée.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from etf_classifier.config import load_settings, resolve_path

KINDS = {"embedding", "chat"}
PER_MILLION = 1_000_000


class PricingError(ValueError):
    """Grille tarifaire invalide."""


class UnknownDeploymentError(PricingError):
    """Déploiement absent de la grille tarifaire."""


@dataclass(frozen=True)
class Rates:
    """Prix en USD par million de tokens."""

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
        """Modèle associé à un déploiement (ou au nom de modèle lui-même), None s'il est inconnu."""
        return self.models.get(self.deployments.get(deployment, deployment))

    def model_for(self, deployment: str) -> ModelPricing:
        model = self.find_model(deployment)
        if model is None:
            raise UnknownDeploymentError(
                f"Déploiement « {deployment} » absent de la grille tarifaire : l'ajouter dans config/pricing.yaml"
            )
        return model

    def age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.as_of).days


def _price(block: dict, key: str, label: str, default: float | None = None) -> float:
    value = block.get(key, default)
    if value is None:
        raise PricingError(f"{label} : prix « {key} » manquant")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PricingError(f"{label} : prix « {key} » non numérique ({value!r})")
    if value < 0:
        raise PricingError(f"{label} : prix « {key} » négatif ({value})")
    return float(value)


def _rates(block: dict, kind: str, label: str) -> Rates:
    input_price = _price(block, "input_per_million", label)
    output_price = _price(block, "output_per_million", label) if kind == "chat" else 0.0
    cached = _price(block, "cached_input_per_million", label, default=input_price)
    return Rates(input=input_price, cached_input=cached, output=output_price)


def _model(name: str, block: dict) -> ModelPricing:
    kind = block.get("kind")
    if kind not in KINDS:
        raise PricingError(f"Modèle {name} : « kind » doit valoir {sorted(KINDS)}")
    if not block.get("tokenizer"):
        raise PricingError(f"Modèle {name} : « tokenizer » manquant")
    threshold, long_rates = block.get("long_context_threshold"), block.get("long_context")
    if (threshold is None) != (long_rates is None):
        raise PricingError(f"Modèle {name} : « long_context_threshold » et « long_context » vont ensemble")
    return ModelPricing(
        name=name,
        kind=kind,
        tokenizer=block["tokenizer"],
        tokenizer_exact=bool(block.get("tokenizer_exact", True)),
        rates=_rates(block, kind, f"Modèle {name}"),
        long_context_threshold=int(threshold) if threshold is not None else None,
        long_context=_rates(long_rates, kind, f"Modèle {name} (long contexte)") if long_rates else None,
    )


def load_pricing(path: str | Path | None = None) -> PricingTable:
    """Charge et contrôle la grille (par défaut celle indiquée dans settings.yaml)."""
    pricing_path = Path(path) if path else resolve_path(load_settings()["costs"]["pricing_path"])
    with pricing_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        as_of = date.fromisoformat(str(raw["as_of"]))
    except (KeyError, ValueError) as exc:
        raise PricingError("Grille tarifaire : date « as_of » manquante ou invalide (AAAA-MM-JJ)") from exc
    models = {name: _model(name, block or {}) for name, block in (raw.get("models") or {}).items()}
    if not models:
        raise PricingError("Grille tarifaire : aucun modèle déclaré")
    deployments = {str(k): str(v) for k, v in (raw.get("deployments") or {}).items()}
    for deployment, model in deployments.items():
        if model not in models:
            raise PricingError(f"Déploiement {deployment} : modèle « {model} » absent de la grille")
    return PricingTable(raw.get("currency", "USD"), as_of, raw.get("source", ""), models, deployments)


def cost_usd(model: ModelPricing, input_tokens: int, cached_input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Coût d'un appel. Les tokens en cache font partie des tokens d'entrée (convention Azure)."""
    rates = model.rates
    if model.long_context_threshold is not None and input_tokens > model.long_context_threshold:
        rates = model.long_context
    cached = min(cached_input_tokens, input_tokens)
    billed = (input_tokens - cached) * rates.input + cached * rates.cached_input + output_tokens * rates.output
    return billed / PER_MILLION
