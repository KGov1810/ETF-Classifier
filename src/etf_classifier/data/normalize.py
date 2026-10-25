"""Normalisation des tickers.

Règles prudentes : majuscules et espaces. Les suffixes de place de cotation (" US", ".PA",
" UW Equity") sont conservés par défaut, car un même ticker peut désigner des sociétés
différentes selon la place : fusionner à tort serait plus grave que ne pas fusionner.
"""

import csv
import re
from collections import defaultdict

import pandas as pd

from etf_classifier.config import resolve_path

_EQUITY_SUFFIX = re.compile(r" EQUITY$")
# Place de cotation : espace + 2 lettres (" US") ou point + 2-3 lettres (".PA").
# Un point + 1 lettre ("BRK.B") est une classe d'action et n'est jamais retiré.
_EXCHANGE_SUFFIX = re.compile(r"( [A-Z]{2}|\.[A-Z]{2,3})$")


def normalize_ticker(raw: str, strip_suffix: bool = False) -> str:
    """Clé normalisée d'un ticker : majuscules, espaces nettoyés, suffixe de place optionnel."""
    ticker = " ".join(str(raw).split()).upper()
    if strip_suffix:
        ticker = _EXCHANGE_SUFFIX.sub("", _EQUITY_SUFFIX.sub("", ticker))
    return ticker


def load_aliases(settings: dict) -> dict[str, str]:
    """Lit config/holding_aliases.csv (colonnes raw;canonical). Fichier absent : aucun alias."""
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
    """Applique la normalisation et les alias ; les poids des tickers fusionnés sont additionnés."""
    normalized: dict[str, float] = {}
    for ticker, weight in composition.items():
        key = normalize_ticker(ticker, strip_suffix)
        key = aliases.get(key, key)
        normalized[key] = normalized.get(key, 0.0) + weight
    return normalized


def normalize_dataset(data: pd.DataFrame, aliases: dict[str, str], strip_suffix: bool) -> pd.DataFrame:
    """Ajoute composition_raw (tickers d'origine) et remplace composition par sa version normalisée."""
    result = data.copy()
    result["composition_raw"] = result["composition"]
    result["composition"] = [normalize_composition(comp, aliases, strip_suffix) for comp in result["composition"]]
    return result


def find_similar_tickers(tickers) -> dict[str, list[str]]:
    """Regroupe les tickers qui partagent une racine mais ont des suffixes différents.

    Aide à remplir holding_aliases.csv ; aucune fusion n'est appliquée automatiquement.
    """
    groups: dict[str, set[str]] = defaultdict(set)
    for ticker in tickers:
        normalized = normalize_ticker(ticker)
        groups[normalize_ticker(normalized, strip_suffix=True)].add(normalized)
    return {root: sorted(variants) for root, variants in sorted(groups.items()) if len(variants) > 1}
