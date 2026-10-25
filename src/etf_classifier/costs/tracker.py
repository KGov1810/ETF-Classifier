"""Log of the actual cost of every Azure call (logs/azure_usage.csv).

Only counters are recorded, never the content of the texts sent.
"""

import csv
import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from etf_classifier.costs.pricing import PricingTable, cost_usd

logger = logging.getLogger(__name__)

LOG_COLUMNS = [
    "timestamp_utc", "module", "operation", "deployment", "input_tokens", "cached_input_tokens",
    "output_tokens", "cost_usd", "estimated_cost_usd", "pricing_as_of", "note",
]
LOG_SEPARATOR = ";"


def _field(obj: Any, name: str, default: Any = 0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass(frozen=True)
class Usage:
    """Tokens consumed by a call (cached tokens are included in input_tokens)."""

    input_tokens: int
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: Any) -> "Usage":
        """Builds the usage from the Azure response (openai library object or dict)."""
        details = _field(usage, "prompt_tokens_details", None)
        return cls(
            input_tokens=int(_field(usage, "prompt_tokens") or 0),
            output_tokens=int(_field(usage, "completion_tokens") or 0),
            cached_input_tokens=int(_field(details, "cached_tokens") or 0),
        )


class CostTracker:
    """Computes and records the cost of every Azure call."""

    def __init__(self, pricing: PricingTable, log_path: str | Path,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.pricing = pricing
        self.log_path = Path(log_path)
        self._now = now

    def record(self, module: str, operation: str, deployment: str, usage: Usage,
               estimated_cost_usd: float | None = None, note: str = "") -> dict:
        """Appends a line to the log and returns it. Unknown deployment: empty cost + warning."""
        model = self.pricing.find_model(deployment)
        cost = None
        if model is None:
            logger.warning("Deployment '%s' missing from the price list: call recorded without a cost", deployment)
        else:
            cost = round(cost_usd(model, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens), 6)
        record = {
            "timestamp_utc": self._now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "module": module,
            "operation": operation,
            "deployment": deployment,
            "input_tokens": usage.input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_usd": cost,
            "estimated_cost_usd": round(estimated_cost_usd, 6) if estimated_cost_usd is not None else None,
            "pricing_as_of": self.pricing.as_of.isoformat(),
            "note": note,
        }
        self._append(record)
        return record

    def _append(self, record: dict) -> None:
        """Writes the line in a single operation, in append mode (header if the file is new)."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=LOG_COLUMNS, delimiter=LOG_SEPARATOR, lineterminator="\n")
        is_new = not self.log_path.exists() or self.log_path.stat().st_size == 0
        if is_new:
            writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in record.items()})
        with self.log_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(buffer.getvalue())
