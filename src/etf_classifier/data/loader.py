"""Lecture du CSV d'ETF et orchestration du chargement.

Le CSV source est ouvert en lecture seule et n'est jamais modifié.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from etf_classifier.config import load_settings
from etf_classifier.data.dedup import DedupResult, add_groups
from etf_classifier.data.normalize import load_aliases, normalize_dataset
from etf_classifier.data.split import SplitResult, make_split
from etf_classifier.data.validation import QualityLog, validate_rows


class MissingColumnsError(ValueError):
    """Une ou plusieurs colonnes attendues sont absentes du CSV."""


@dataclass
class LoadResult:
    """Jeu exploitable et journal de qualité produits par load_dataset()."""

    data: pd.DataFrame
    log: QualityLog
    total_rows: int


def read_raw_rows(path: str | Path, settings: dict) -> tuple[list[dict], list[dict]]:
    """Lit le CSV ligne à ligne en conservant le numéro de ligne du fichier.

    Renvoie (lignes valides structurellement, lignes mal formées). Une ligne est mal formée
    si son nombre de champs diffère de celui de l'en-tête (par exemple un ';' non protégé).
    """
    csv_cfg = settings["csv"]
    expected = list(csv_cfg["columns"].values())
    rows: list[dict] = []
    bad_lines: list[dict] = []
    with Path(path).open(encoding=csv_cfg["encoding"], newline="") as handle:
        reader = csv.reader(handle, delimiter=csv_cfg["separator"], quotechar='"')
        header = [column.strip().lstrip("﻿") for column in next(reader, [])]
        missing = [column for column in expected if column not in header]
        if missing:
            raise MissingColumnsError(f"Colonnes absentes du CSV : {', '.join(missing)}")
        record = 0
        previous_line = reader.line_num
        for fields in reader:
            line = previous_line + 1  # première ligne physique de l'enregistrement
            previous_line = reader.line_num
            if not any(field.strip() for field in fields):
                continue
            if len(fields) != len(header):
                extract = csv_cfg["separator"].join(fields)[:60]
                bad_lines.append({"record": record, "line": line, "extract": extract})
            else:
                row = dict(zip(header, fields))
                row["__line__"] = line
                row["__record__"] = record
                rows.append(row)
            record += 1
    return rows, bad_lines


def load_dataset(path: str | Path, settings: dict | None = None) -> LoadResult:
    """Charge, contrôle et normalise le CSV d'ETF.

    Colonnes du jeu exploitable : row_id, line, name, description, composition_raw,
    composition (tickers normalisés), categories, coverage, n_holdings.
    Les lignes exclues et les avertissements sont dans LoadResult.log.
    """
    settings = settings or load_settings()
    rows, bad_lines = read_raw_rows(path, settings)
    data, log = validate_rows(rows, bad_lines, settings)
    aliases = load_aliases(settings)
    data = normalize_dataset(data, aliases, settings["normalize"]["strip_exchange_suffix"])
    return LoadResult(data=data, log=log, total_rows=len(rows) + len(bad_lines))


@dataclass
class PreparedDataset:
    """Jeu prêt pour l'entraînement : données avec group_id, journal, groupes et découpage."""

    data: pd.DataFrame
    log: QualityLog
    total_rows: int
    dedup: DedupResult
    split: SplitResult


def prepare_dataset(path: str | Path, settings: dict | None = None) -> PreparedDataset:
    """Point d'entrée des modules suivants : chargement, quasi-doublons puis découpage."""
    settings = settings or load_settings()
    loaded = load_dataset(path, settings)
    data = loaded.data.reset_index(drop=True)
    dedup = add_groups(data, settings)
    split_cfg = settings["split"]
    split = make_split(data, split_cfg["test_size"], split_cfg["seed"], split_cfg["rare_category_min"])
    return PreparedDataset(data, loaded.log, loaded.total_rows, dedup, split)
