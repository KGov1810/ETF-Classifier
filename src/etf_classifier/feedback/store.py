"""Validation journal (data/validations.csv): the module's single writing point.

Rules:
- every read and every write of the journal happens under a (portable) file lock;
- a complete line is written in a single operation, in append mode;
- nothing is ever rewritten or deleted: a mistake is fixed with a "cancellation" line;
- the original CSV is never touched.
"""

import csv
import io
import logging
import secrets
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from filelock import FileLock, Timeout

from etf_classifier.config import resolve_path
from etf_classifier.data.parsing import ParsingError, parse_categories
from etf_classifier.feedback.categories import CategoryRegistry
from etf_classifier.feedback.checks import ValidationError, check_validation

logger = logging.getLogger(__name__)

JOURNAL_COLUMNS = ["ValidationId", "Name", "Description", "Composition", "Category", "ValidatedBy",
                   "ValidatedAt", "ModelVersion", "PredictedCategories", "Decision", "CancelsId", "Comment"]
ACCEPTED, CORRECTED, NO_PROPOSAL, CANCELLATION = "accepted", "corrected", "no_proposal", "cancellation"


class FeedbackError(RuntimeError):
    """Write impossible (lock unavailable, file locked by another application)."""


def serialize_composition(composition) -> str:
    """Composition in the reference CSV format: "[{'AAPL': 6.5}, {'MSFT': 6.1}]"."""
    if isinstance(composition, str):
        return composition.strip()
    return str([{str(ticker): float(weight)} for ticker, weight in composition.items()])


def _clean(categories) -> list[str]:
    cleaned: list[str] = []
    for category in categories or []:
        label = str(category).strip()
        if label and label not in cleaned:
            cleaned.append(label)
    return cleaned


class FeedbackStore:
    def __init__(self, settings: dict):
        cfg = settings["feedback"]
        self.settings = settings
        self.journal_path = resolve_path(cfg["journal_path"])
        self.backups_dir = resolve_path(cfg["backups_dir"])
        self.backups_kept = int(cfg["backups_kept"])
        self.lock_timeout = float(cfg["lock_timeout_seconds"])
        self.reference_csv = resolve_path(cfg["reference_csv"])
        self.registry = CategoryRegistry(settings)
        self._reference: tuple[set[str], set[str]] | None = None

    # ---------- lock ----------
    @contextmanager
    def locked(self):
        """Exclusive lock on the journal (and on the category registry)."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.journal_path) + ".lock")
        try:
            lock.acquire(timeout=self.lock_timeout)
        except Timeout as exc:
            raise FeedbackError("Another record is being saved: try again in a few seconds.") from exc
        try:
            yield
        finally:
            lock.release()

    # ---------- reading ----------
    def _read_unlocked(self) -> pd.DataFrame:
        if not self.journal_path.exists() or self.journal_path.stat().st_size == 0:
            return pd.DataFrame(columns=JOURNAL_COLUMNS)
        return pd.read_csv(self.journal_path, sep=";", dtype=str, keep_default_na=False, encoding="utf-8")

    def read_journal(self) -> pd.DataFrame:
        with self.locked():
            return self._read_unlocked()

    @staticmethod
    def _active(journal: pd.DataFrame) -> pd.DataFrame:
        cancelled = set(journal.loc[journal["Decision"] == CANCELLATION, "CancelsId"])
        keep = (journal["Decision"] != CANCELLATION) & ~journal["ValidationId"].isin(cancelled)
        return journal[keep]  # the index stays the position in the journal (for line numbers)

    def known_categories(self) -> set[str]:
        """Historical and created categories (the registry is initialised from the original CSV if needed)."""
        with self.locked():
            self.registry.ensure_initialized(self._reference_names_and_categories()[1])
            return self.registry.known()

    def active_validations(self) -> pd.DataFrame:
        """Validations that have not been cancelled: they are the ones enriching the reference dataset."""
        return self._active(self.read_journal())

    def _reference_names_and_categories(self) -> tuple[set[str], set[str]]:
        """Names and categories of the original CSV, read once (without going through the data module)."""
        if self._reference is None:
            names, categories = set(), set()
            cols = self.settings["csv"]["columns"]
            if self.reference_csv.exists():
                with self.reference_csv.open(encoding=self.settings["csv"]["encoding"], newline="") as handle:
                    for row in csv.DictReader(handle, delimiter=self.settings["csv"]["separator"]):
                        names.add((row.get(cols["name"]) or "").strip())
                        try:
                            categories.update(parse_categories(row.get(cols["category"]) or ""))
                        except ParsingError:
                            continue  # line already reported by the data module's quality report
            else:
                logger.warning("Reference CSV not found (%s): registry initialised empty", self.reference_csv)
            self._reference = (names, categories)
        return self._reference

    # ---------- writing ----------
    def _backup_if_needed(self) -> None:
        """One dated backup per day (before the first write of the day), the last N kept."""
        if not self.journal_path.exists():
            return
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        target = self.backups_dir / f"validations_{datetime.now(timezone.utc):%Y%m%d}.csv"
        if not target.exists():
            shutil.copy2(self.journal_path, target)
            backups = sorted(self.backups_dir.glob("validations_*.csv"))
            for old in backups[: max(0, len(backups) - self.backups_kept)]:
                old.unlink()

    def _append(self, record: dict) -> None:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=JOURNAL_COLUMNS, delimiter=";", lineterminator="\n")
        if not self.journal_path.exists() or self.journal_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({k: ("" if record.get(k) is None else record[k]) for k in JOURNAL_COLUMNS})
        try:
            with self.journal_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write(buffer.getvalue())
        except PermissionError as exc:
            raise FeedbackError("The journal is in use by another application (Excel?): "
                                "close it and try again.") from exc

    @staticmethod
    def _new_id() -> str:
        return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

    def record_validation(self, etf: dict, categories, validated_by: str, predicted_categories=None,
                          model_version: str | None = None, comment: str = "", create_categories: bool = False,
                          allow_duplicate_name: bool = False, category_descriptions: dict | None = None) -> dict:
        """Checks then records a validation; raises ValidationError or FeedbackError otherwise.

        category_descriptions: optional description of each category created by this validation.
        """
        categories = _clean(categories)
        name = str(etf.get("name") or "").strip()
        composition = serialize_composition(etf.get("composition") or {})
        with self.locked():
            reference_names, reference_categories = self._reference_names_and_categories()
            self.registry.ensure_initialized(reference_categories)
            active = self._active(self._read_unlocked())
            errors = check_validation(name, composition, categories, validated_by or "", self.registry.known(),
                                      reference_names | set(active["Name"]), create_categories,
                                      allow_duplicate_name)
            if errors:
                raise ValidationError(errors)
            for category in categories:
                if category not in self.registry.known():
                    self.registry.add(category, validated_by.strip(),
                                      (category_descriptions or {}).get(category, ""))
            if predicted_categories is None:
                decision = NO_PROPOSAL
            else:
                decision = ACCEPTED if set(_clean(predicted_categories)) == set(categories) else CORRECTED
            record = {
                "ValidationId": self._new_id(),
                "Name": name,
                "Description": str(etf.get("description") or "").strip(),
                "Composition": composition,
                "Category": str(categories),
                "ValidatedBy": validated_by.strip(),
                "ValidatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ModelVersion": model_version or "",
                "PredictedCategories": str(_clean(predicted_categories)) if predicted_categories is not None else "",
                "Decision": decision,
                "CancelsId": "",
                "Comment": comment.strip(),
            }
            self._backup_if_needed()
            self._append(record)
        return record

    def cancel_validation(self, validation_id: str, cancelled_by: str, comment: str = "") -> dict:
        """Cancels a validation with a new line; nothing is deleted."""
        if not cancelled_by.strip():
            raise ValidationError(["The identity of the person cancelling is required."])
        with self.locked():
            journal = self._read_unlocked()
            validations = journal[journal["Decision"] != CANCELLATION]
            if validation_id not in set(validations["ValidationId"]):
                raise ValidationError([f"Unknown validation: {validation_id}"])
            if validation_id in set(journal.loc[journal["Decision"] == CANCELLATION, "CancelsId"]):
                raise ValidationError([f"Validation already cancelled: {validation_id}"])
            original = validations.set_index("ValidationId").loc[validation_id]
            record = {
                "ValidationId": self._new_id(), "Name": original["Name"], "Description": "", "Composition": "",
                "Category": "", "ValidatedBy": cancelled_by.strip(),
                "ValidatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ModelVersion": "", "PredictedCategories": "", "Decision": CANCELLATION,
                "CancelsId": validation_id, "Comment": comment.strip(),
            }
            self._backup_if_needed()
            self._append(record)
        return record
