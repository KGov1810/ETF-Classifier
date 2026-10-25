"""Journal des validations (data/validations.csv) : le seul point d'écriture du module.

Règles :
- toute lecture et toute écriture du journal se font sous verrou de fichier (portable) ;
- une ligne complète est écrite en une seule opération, en mode ajout ;
- on ne réécrit ni ne supprime jamais rien : une erreur se corrige par une ligne « annulation » ;
- le CSV d'origine n'est jamais touché.
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
ACCEPTED, CORRECTED, NO_PROPOSAL, CANCELLATION = "accepté", "corrigé", "sans_proposition", "annulation"


class FeedbackError(RuntimeError):
    """Écriture impossible (verrou indisponible, fichier bloqué par une autre application)."""


def serialize_composition(composition) -> str:
    """Composition au format du CSV de référence : "[{'AAPL': 6.5}, {'MSFT': 6.1}]"."""
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

    # ---------- verrou ----------
    @contextmanager
    def locked(self):
        """Verrou exclusif sur le journal (et le registre des catégories)."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.journal_path) + ".lock")
        try:
            lock.acquire(timeout=self.lock_timeout)
        except Timeout as exc:
            raise FeedbackError("Un autre enregistrement est en cours : réessayez dans quelques secondes.") from exc
        try:
            yield
        finally:
            lock.release()

    # ---------- lecture ----------
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
        return journal[keep]  # l'index reste la position dans le journal (pour les numéros de ligne)

    def active_validations(self) -> pd.DataFrame:
        """Validations non annulées : ce sont elles qui enrichissent le jeu de référence."""
        return self._active(self.read_journal())

    def _reference_names_and_categories(self) -> tuple[set[str], set[str]]:
        """Noms et catégories du CSV d'origine, lus une fois (sans passer par le module data)."""
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
                            continue  # ligne déjà signalée par le rapport de qualité du module data
            else:
                logger.warning("CSV de référence introuvable (%s) : registre initialisé vide", self.reference_csv)
            self._reference = (names, categories)
        return self._reference

    # ---------- écriture ----------
    def _backup_if_needed(self) -> None:
        """Une sauvegarde datée par jour (avant la première écriture du jour), N dernières conservées."""
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
            raise FeedbackError("Le journal est utilisé par une autre application (Excel ?) : "
                                "le fermer puis réessayer.") from exc

    @staticmethod
    def _new_id() -> str:
        return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

    def record_validation(self, etf: dict, categories, validated_by: str, predicted_categories=None,
                          model_version: str | None = None, comment: str = "", create_categories: bool = False,
                          allow_duplicate_name: bool = False) -> dict:
        """Contrôle puis enregistre une validation ; lève ValidationError ou FeedbackError sinon."""
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
                    self.registry.add(category, validated_by.strip())
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
        """Annule une validation par une nouvelle ligne ; rien n'est supprimé."""
        if not cancelled_by.strip():
            raise ValidationError(["L'identité de la personne qui annule est obligatoire."])
        with self.locked():
            journal = self._read_unlocked()
            validations = journal[journal["Decision"] != CANCELLATION]
            if validation_id not in set(validations["ValidationId"]):
                raise ValidationError([f"Validation inconnue : {validation_id}"])
            if validation_id in set(journal.loc[journal["Decision"] == CANCELLATION, "CancelsId"]):
                raise ValidationError([f"Validation déjà annulée : {validation_id}"])
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
