"""Conversion of the Composition and Category cells into Python structures.

The cells are not strict JSON (single quotes): they are read with ast.literal_eval,
which only accepts literals and never executes code.
"""

import ast
from typing import Any

EXTRACT_LENGTH = 60


class ParsingError(ValueError):
    """Unreadable cell or unexpected structure."""


class WeightError(ParsingError):
    """Non-numeric weight in a composition."""


def _extract(raw: str) -> str:
    """Short extract of a cell, for error messages."""
    return raw if len(raw) <= EXTRACT_LENGTH else raw[:EXTRACT_LENGTH] + "…"


def _literal(raw: str, label: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ParsingError(f"Unreadable {label}: {_extract(raw)}") from exc


def parse_composition_items(raw: str) -> list[tuple[str, Any]]:
    """Converts "[{'AAPL': 6.5}, ...]" into [('AAPL', 6.5), ...], keeping duplicates.

    Also accepts a plain dictionary "{'AAPL': 6.5, ...}". An empty cell gives [].
    Weights are not checked here (see parse_composition).
    """
    if raw is None or not str(raw).strip():
        return []
    value = _literal(str(raw).strip(), "Composition")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ParsingError(f"Composition: list of dictionaries expected: {_extract(str(raw))}")
    items: list[tuple[str, Any]] = []
    for item in value:
        for ticker, weight in item.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise ParsingError(f"Composition: invalid ticker {ticker!r}: {_extract(str(raw))}")
            items.append((ticker, weight))
    return items


def to_weight(value: Any) -> float:
    """Converts a weight to float; raises WeightError if it is not numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeightError(f"Non-numeric weight: {value!r}")
    return float(value)


def parse_composition(raw: str) -> dict[str, float]:
    """Converts "[{'AAPL': 6.5}, {'MSFT': 6.1}]" into {'AAPL': 6.5, 'MSFT': 6.1}.

    The weights of a ticker listed several times are summed.
    Raises ParsingError if the cell is unreadable, WeightError if a weight is not numeric.
    """
    composition: dict[str, float] = {}
    for ticker, weight in parse_composition_items(raw):
        composition[ticker] = composition.get(ticker, 0.0) + to_weight(weight)
    return composition


def parse_categories(raw: str) -> list[str]:
    """Converts "['Global Equity', 'ESG']" into ['Global Equity', 'ESG'].

    Whitespace stripped, duplicates removed (order kept), empty strings ignored.
    """
    if raw is None or not str(raw).strip():
        return []
    value = _literal(str(raw).strip(), "Category")
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ParsingError(f"Category: list of strings expected: {_extract(str(raw))}")
    categories: list[str] = []
    for item in value:
        label = item.strip()
        if label and label not in categories:
            categories.append(label)
    return categories
