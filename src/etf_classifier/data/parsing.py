"""Conversion des cellules Composition et Category en structures Python.

Les cellules ne sont pas du JSON strict (apostrophes simples) : elles sont lues avec
ast.literal_eval, qui n'accepte que des littéraux et n'exécute jamais de code.
"""

import ast
from typing import Any

EXTRACT_LENGTH = 60


class ParsingError(ValueError):
    """Cellule illisible ou de structure inattendue."""


class WeightError(ParsingError):
    """Poids non numérique dans une composition."""


def _extract(raw: str) -> str:
    """Extrait court d'une cellule, pour les messages d'erreur."""
    return raw if len(raw) <= EXTRACT_LENGTH else raw[:EXTRACT_LENGTH] + "…"


def _literal(raw: str, label: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ParsingError(f"{label} illisible : {_extract(raw)}") from exc


def parse_composition_items(raw: str) -> list[tuple[str, Any]]:
    """Convertit "[{'AAPL': 6.5}, ...]" en [('AAPL', 6.5), ...], doublons conservés.

    Accepte aussi un dictionnaire simple "{'AAPL': 6.5, ...}". Une cellule vide donne [].
    Les poids ne sont pas contrôlés ici (voir parse_composition).
    """
    if raw is None or not str(raw).strip():
        return []
    value = _literal(str(raw).strip(), "Composition")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ParsingError(f"Composition : liste de dictionnaires attendue : {_extract(str(raw))}")
    items: list[tuple[str, Any]] = []
    for item in value:
        for ticker, weight in item.items():
            if not isinstance(ticker, str) or not ticker.strip():
                raise ParsingError(f"Composition : ticker invalide {ticker!r} : {_extract(str(raw))}")
            items.append((ticker, weight))
    return items


def to_weight(value: Any) -> float:
    """Convertit un poids en float ; lève WeightError s'il n'est pas numérique."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeightError(f"Poids non numérique : {value!r}")
    return float(value)


def parse_composition(raw: str) -> dict[str, float]:
    """Convertit "[{'AAPL': 6.5}, {'MSFT': 6.1}]" en {'AAPL': 6.5, 'MSFT': 6.1}.

    Les poids d'un ticker présent plusieurs fois sont additionnés.
    Lève ParsingError si la cellule est illisible, WeightError si un poids n'est pas numérique.
    """
    composition: dict[str, float] = {}
    for ticker, weight in parse_composition_items(raw):
        composition[ticker] = composition.get(ticker, 0.0) + to_weight(weight)
    return composition


def parse_categories(raw: str) -> list[str]:
    """Convertit "['Actions Monde', 'ESG']" en ['Actions Monde', 'ESG'].

    Espaces retirés, doublons supprimés (ordre conservé), chaînes vides ignorées.
    """
    if raw is None or not str(raw).strip():
        return []
    value = _literal(str(raw).strip(), "Category")
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ParsingError(f"Category : liste de textes attendue : {_extract(str(raw))}")
    categories: list[str] = []
    for item in value:
        label = item.strip()
        if label and label not in categories:
            categories.append(label)
    return categories
