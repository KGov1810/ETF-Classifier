"""Règles de qualité appliquées à chaque ligne du CSV.

Une ligne invalide est exclue du jeu exploitable et consignée avec sa raison ;
elle n'est jamais supprimée du CSV source.
"""

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from etf_classifier.data.parsing import (
    ParsingError,
    WeightError,
    parse_categories,
    parse_composition_items,
    to_weight,
)

EXCLUSION = "exclusion"
WARNING = "avertissement"

DATA_COLUMNS = ["row_id", "line", "name", "description", "composition", "categories", "coverage", "n_holdings"]

RULE_LABELS = {
    "ligne_mal_formee": "Ligne mal formée (nombre de champs incorrect, ';' non protégé ?)",
    "composition_illisible": "Composition impossible à lire",
    "categories_illisibles": "Catégories impossibles à lire",
    "aucune_categorie": "Aucune catégorie",
    "poids_non_numerique": "Poids non numérique",
    "poids_negatif": "Poids négatif",
    "composition_vide": "Composition vide (la description seule servira)",
    "description_vide": "Description vide",
    "ticker_en_double": "Ticker présent plusieurs fois (poids additionnés)",
    "somme_poids_elevee": "Somme des poids supérieure au maximum autorisé",
    "nom_en_double": "Nom d'ETF en double",
}


@dataclass
class QualityIssue:
    """Un problème de qualité rattaché à une ligne du CSV."""

    row_id: int | None
    line: int | None
    rule: str
    severity: str
    message: str
    extract: str = ""


@dataclass
class QualityLog:
    """Journal des exclusions et avertissements produits au chargement."""

    issues: list[QualityIssue] = field(default_factory=list)

    def add(self, row_id, line, rule, severity, message="", extract="") -> None:
        self.issues.append(QualityIssue(row_id, line, rule, severity, message or RULE_LABELS[rule], extract[:60]))

    def exclusions(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == EXCLUSION]

    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    def counts(self) -> Counter:
        """Nombre de problèmes par (gravité, règle)."""
        return Counter((issue.severity, issue.rule) for issue in self.issues)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(issue) for issue in self.issues],
                            columns=["row_id", "line", "rule", "severity", "message", "extract"])


def _validate_row(row: dict, row_id: int, line: int, settings: dict, log: QualityLog) -> dict | None:
    """Contrôle une ligne ; renvoie l'enregistrement exploitable ou None si elle est exclue."""
    cols = settings["csv"]["columns"]
    raw_composition = row[cols["composition"]]
    raw_categories = row[cols["category"]]

    try:
        items = parse_composition_items(raw_composition)
    except ParsingError as exc:
        log.add(row_id, line, "composition_illisible", EXCLUSION, str(exc), raw_composition)
        return None

    try:
        categories = parse_categories(raw_categories)
    except ParsingError as exc:
        log.add(row_id, line, "categories_illisibles", EXCLUSION, str(exc), raw_categories)
        return None
    if not categories:
        log.add(row_id, line, "aucune_categorie", EXCLUSION, extract=raw_categories)
        return None

    composition: dict[str, float] = {}
    for ticker, raw_weight in items:
        try:
            weight = to_weight(raw_weight)
        except WeightError as exc:
            log.add(row_id, line, "poids_non_numerique", EXCLUSION, f"{ticker} : {exc}", raw_composition)
            return None
        if weight < 0:
            log.add(row_id, line, "poids_negatif", EXCLUSION, f"{ticker} : {weight}", raw_composition)
            return None
        if ticker in composition:
            log.add(row_id, line, "ticker_en_double", WARNING, f"Ticker en double : {ticker}", raw_composition)
        composition[ticker] = composition.get(ticker, 0.0) + weight

    coverage = sum(composition.values())
    if not composition:
        log.add(row_id, line, "composition_vide", WARNING, extract=raw_composition)
    if coverage > settings["quality"]["max_weight_sum"]:
        log.add(row_id, line, "somme_poids_elevee", WARNING, f"Somme des poids : {coverage:.2f}", raw_composition)

    description = row[cols["description"]].strip()
    if not description:
        log.add(row_id, line, "description_vide", WARNING, extract=row[cols["name"]])

    return {
        "row_id": row_id,
        "line": line,
        "name": row[cols["name"]].strip(),
        "description": description,
        "composition": composition,
        "categories": categories,
        "coverage": coverage,
        "n_holdings": len(composition),
    }


def validate_rows(rows: list[dict], bad_lines: list[dict], settings: dict) -> tuple[pd.DataFrame, QualityLog]:
    """Applique toutes les règles ; renvoie (jeu exploitable, journal de qualité)."""
    log = QualityLog()
    for bad in bad_lines:
        log.add(bad.get("record"), bad.get("line"), "ligne_mal_formee", EXCLUSION, extract=bad.get("extract", ""))

    records = []
    for position, row in enumerate(rows):
        row_id = row.get("__record__", position)
        record = _validate_row(row, row_id, row.get("__line__"), settings, log)
        if record is not None:
            records.append(record)

    seen_names: set[str] = set()
    for record in records:
        if record["name"] in seen_names:
            log.add(record["row_id"], record["line"], "nom_en_double", WARNING, extract=record["name"])
        seen_names.add(record["name"])

    return pd.DataFrame(records, columns=DATA_COLUMNS), log
