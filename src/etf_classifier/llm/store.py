"""Justifications kept in `data/llm_justifications.jsonl` (append-only, one JSON line per event).

Two kinds of lines: a justification (one per paid call) and a link to the validation that used it.
Nothing is ever rewritten; the validation journal format stays unchanged.
"""

import json

from filelock import FileLock

from etf_classifier.common import new_record_id, utc_timestamp
from etf_classifier.config import resolve_path
from etf_classifier.llm.prompts import PROMPT_VERSION

JUSTIFICATION, LINK = "justification", "link"


class JustificationStore:
    def __init__(self, settings: dict):
        self.path = resolve_path(settings["llm"]["justifications_path"])
        self.lock_timeout = float(settings.get("feedback", {}).get("lock_timeout_seconds", 10))

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=self.lock_timeout):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _records(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append(self, ticker: str, etf_name: str, model_version: str, deployment: str, answer: dict, input_tokens: int,
               output_tokens: int, cost_usd: float | None) -> str:
        """Records a justification; returns its identifier."""
        record_id = new_record_id()
        self._append({
            "type": JUSTIFICATION, "id": record_id,
            "created_at": utc_timestamp(),
            "prompt_version": PROMPT_VERSION, "deployment": deployment, "model_version": model_version,
            "ticker": ticker, "etf_name": etf_name, "answer": answer, "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })
        return record_id

    def link(self, justification_id: str, validation_id: str) -> None:
        """Ties a justification to the validation that followed it."""
        self._append({"type": LINK, "justification_id": justification_id, "validation_id": validation_id,
                      "created_at": utc_timestamp()})

    def get(self, justification_id: str) -> dict | None:
        return next((r for r in self._records() if r["type"] == JUSTIFICATION and r["id"] == justification_id), None)

    def for_validation(self, validation_id: str) -> dict | None:
        """Justification linked to a validation (the last link wins), or None."""
        records = self._records()
        links = [r for r in records if r["type"] == LINK and r["validation_id"] == validation_id]
        if not links:
            return None
        target = links[-1]["justification_id"]
        return next((r for r in records if r["type"] == JUSTIFICATION and r["id"] == target), None)
