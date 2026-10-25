"""Category registry (data/categories.csv): historical and created categories, with their author.

Writes are done by FeedbackStore, under the journal lock.
"""

import pandas as pd

from etf_classifier.common import append_csv_rows, read_csv_table, utc_timestamp
from etf_classifier.config import resolve_path

REGISTRY_COLUMNS = ["category", "source", "created_by", "created_at", "description"]
HISTORICAL, CREATED = "historical", "created"


class CategoryRegistry:
    def __init__(self, settings: dict):
        self.path = resolve_path(settings["feedback"]["categories_path"])

    def read(self) -> pd.DataFrame:
        return read_csv_table(self.path, REGISTRY_COLUMNS)

    def known(self) -> set[str]:
        return set(self.read()["category"])

    def created(self) -> pd.DataFrame:
        """Categories created by the validators (excluding historical ones)."""
        registry = self.read()
        return registry[registry["source"] == CREATED]

    def _append(self, rows: list[dict]) -> None:
        append_csv_rows(self.path, REGISTRY_COLUMNS, rows)

    def ensure_initialized(self, reference_categories: set[str]) -> None:
        """Adds the historical categories missing from the registry (on first use, then when the CSV grows)."""
        missing = sorted(reference_categories - self.known())
        if missing:
            self._append([{"category": c, "source": HISTORICAL, "created_by": "", "created_at": "",
                           "description": ""} for c in missing])

    def add(self, category: str, created_by: str, description: str = "") -> None:
        self._append([{"category": category, "source": CREATED, "created_by": created_by,
                       "created_at": utc_timestamp(), "description": description}])
