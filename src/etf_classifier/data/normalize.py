"""Ticker normalisation.

Conservative rules: upper case and whitespace. Exchange suffixes (" US", ".PA", " UW Equity")
are kept by default, because the same ticker can refer to different companies on different
exchanges: a wrong merge would be worse than no merge.
"""

import csv
import re
from collections import defaultdict

import pandas as pd

from etf_classifier.config import resolve_path

_EQUITY_SUFFIX = re.compile(r" EQUITY$")
# Exchange: space + 2 letters (" US") or dot + 2-3 letters (".PA").
# A dot + 1 letter ("BRK.B") is a share class and is never removed.
_EXCHANGE_SUFFIX = re.compile(r"( [A-Z]{2}|\.[A-Z]{2,3})$")


def normalize_ticker(raw: str, strip_suffix: bool = False) -> str:
    """Normalised key of a ticker: upper case, cleaned whitespace, optional exchange suffix removal."""
    ticker = " ".join(str(raw).split()).upper()
    if strip_suffix:
        ticker = _EXCHANGE_SUFFIX.sub("", _EQUITY_SUFFIX.sub("", ticker))
    return ticker


def load_aliases(settings: dict) -> dict[str, str]:
    """Reads config/holding_aliases.csv (columns raw;canonical). Missing file: no alias."""
    path = resolve_path(settings["normalize"]["aliases_path"])
    if not path.exists():
        return {}
    strip_suffix = settings["normalize"]["strip_exchange_suffix"]
    aliases: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            raw, canonical = (row.get("raw") or "").strip(), (row.get("canonical") or "").strip()
            if raw and canonical:
                aliases[normalize_ticker(raw, strip_suffix)] = normalize_ticker(canonical, strip_suffix)
    return aliases


def normalize_composition(
    composition: dict[str, float], aliases: dict[str, str], strip_suffix: bool = False
) -> dict[str, float]:
    """Applies normalisation and aliases; the weights of merged tickers are summed."""
    normalized: dict[str, float] = {}
    for ticker, weight in composition.items():
        key = normalize_ticker(ticker, strip_suffix)
        key = aliases.get(key, key)
        normalized[key] = normalized.get(key, 0.0) + weight
    return normalized


def normalize_dataset(data: pd.DataFrame, aliases: dict[str, str], strip_suffix: bool) -> pd.DataFrame:
    """Adds composition_raw (original tickers) and replaces composition with its normalised version."""
    result = data.copy()
    result["composition_raw"] = result["composition"]
    result["composition"] = [normalize_composition(comp, aliases, strip_suffix) for comp in result["composition"]]
    return result


def find_similar_tickers(tickers) -> dict[str, list[str]]:
    """Groups tickers that share a root but have different suffixes.

    Helps to fill in holding_aliases.csv; no merge is applied automatically.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for ticker in tickers:
        normalized = normalize_ticker(ticker)
        groups[normalize_ticker(normalized, strip_suffix=True)].add(normalized)
    return {root: sorted(variants) for root, variants in sorted(groups.items()) if len(variants) > 1}
