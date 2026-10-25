"""Run history (one CSV line per daily run) and the job lock that prevents two runs at the same time."""

from contextlib import contextmanager

import pandas as pd
from filelock import FileLock, Timeout

from etf_classifier.common import append_csv_rows, read_csv_table
from etf_classifier.config import resolve_path

HISTORY_COLUMNS = ["started_at", "outcome", "current_version", "candidate_version", "comparison_etfs",
                   "current_f1_macro", "candidate_f1_macro", "azure_cost_usd", "duration_seconds",
                   "dataset_fingerprint", "reasons"]
PROMOTED, NOT_PROMOTED, SKIPPED, FAILED = "promoted", "not_promoted", "skipped", "failed"


class RetrainingBusy(RuntimeError):
    """Another retraining run holds the lock."""


def _history_path(settings: dict):
    return resolve_path(settings["retraining"]["history_path"])


@contextmanager
def job_lock(settings: dict):
    """Exclusive lock for the whole run; fails immediately if another run is in progress."""
    lock_path = resolve_path(settings["model"]["artifacts_dir"]) / "retraining.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RetrainingBusy("A retraining is already running: this run stops without doing anything.") from exc
    try:
        yield
    finally:
        lock.release()


def append_run(settings: dict, record: dict) -> None:
    """Appends one run to the history (missing columns are left empty)."""
    append_csv_rows(_history_path(settings), HISTORY_COLUMNS, [record])


def last_trained_fingerprint(settings: dict) -> str | None:
    """Dataset fingerprint of the last run that trained a candidate (promoted or not)."""
    history = read_history(settings)
    trained = history[history["outcome"].isin([PROMOTED, NOT_PROMOTED]) & (history["dataset_fingerprint"] != "")]
    return trained.iloc[-1]["dataset_fingerprint"] if not trained.empty else None


def read_history(settings: dict) -> pd.DataFrame:
    """Every run, in chronological order."""
    return read_csv_table(_history_path(settings), HISTORY_COLUMNS)
