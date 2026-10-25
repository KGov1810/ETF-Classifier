"""Registre des catégories (data/categories.csv) : historiques et créées, avec leur auteur.

Les écritures sont faites par FeedbackStore, sous le verrou du journal.
"""

import csv
import io
from datetime import datetime, timezone

import pandas as pd

from etf_classifier.config import resolve_path

REGISTRY_COLUMNS = ["category", "source", "created_by", "created_at", "description"]


class CategoryRegistry:
    def __init__(self, settings: dict):
        self.path = resolve_path(settings["feedback"]["categories_path"])

    def read(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=REGISTRY_COLUMNS)
        return pd.read_csv(self.path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8")

    def known(self) -> set[str]:
        return set(self.read()["category"])

    def _append(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=REGISTRY_COLUMNS, delimiter=";", lineterminator="\n")
        if not self.path.exists() or self.path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(buffer.getvalue())

    def ensure_initialized(self, reference_categories: set[str]) -> None:
        """Ajoute les catégories historiques absentes du registre (au premier usage, puis si le CSV grandit)."""
        missing = sorted(reference_categories - self.known())
        if missing:
            self._append([{"category": c, "source": "historique", "created_by": "", "created_at": "",
                           "description": ""} for c in missing])

    def add(self, category: str, created_by: str, description: str = "") -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._append([{"category": category, "source": "créée", "created_by": created_by,
                       "created_at": now, "description": description}])
