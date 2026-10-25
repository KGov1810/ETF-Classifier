"""Justification of a proposal: context, estimate, cache, confirmed call, answer checks, cost and storage."""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from etf_classifier.config import resolve_path
from etf_classifier.costs.estimate import CostEstimate, estimate_chat
from etf_classifier.costs.pricing import load_pricing
from etf_classifier.costs.tracker import CostTracker, Usage
from etf_classifier.llm.client import chat_deployment, complete, create_chat_client
from etf_classifier.llm.prompts import (
    PROMPT_VERSION, AnswerError, Justification, build_context, build_messages, check_answer,
)
from etf_classifier.llm.store import JustificationStore


class LLMUnavailable(RuntimeError):
    """Azure OpenAI or the chat deployment is not configured."""


class ConfirmationRequired(RuntimeError):
    """The call costs money: the user must confirm the estimate first."""

    def __init__(self, estimate: CostEstimate):
        self.estimate = estimate
        super().__init__(f"Estimated cost ${estimate.cost_usd:.4f}: confirmation required.")


@dataclass
class Estimate:
    cost: CostEstimate
    cached: bool  # True: the same request was already answered, nothing to pay


@dataclass
class JustificationResult:
    justification: Justification
    justification_id: str
    cost_usd: float
    cached: bool


class _AnswerCache:
    """Checked answers by prompt fingerprint (SQLite, same approach as the embeddings cache)."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, justification_id TEXT, "
                               "answer TEXT NOT NULL, created_at TEXT NOT NULL)")

    def get(self, key: str) -> tuple[str, dict] | None:
        with sqlite3.connect(self.path, timeout=30) as connection:
            row = connection.execute("SELECT justification_id, answer FROM answers WHERE key = ?", (key,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def put(self, key: str, justification_id: str, answer: dict) -> None:
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)",
                               (key, justification_id, json.dumps(answer, ensure_ascii=False),
                                datetime.now(timezone.utc).isoformat(timespec="seconds")))


class Justifier:
    """Entry point used by the interface. The Azure client can be injected (tests)."""

    def __init__(self, settings: dict, client=None):
        self.settings = settings
        self.cfg = settings["llm"]
        self.client = client if client is not None else create_chat_client(settings)
        self.deployment = chat_deployment(settings)
        self.pricing = load_pricing(resolve_path(settings["costs"]["pricing_path"]))
        self.tracker = CostTracker(self.pricing, resolve_path(settings["costs"]["usage_log_path"]))
        self.store = JustificationStore(settings)
        self.cache = _AnswerCache(resolve_path(self.cfg["cache_path"]))
        self.unavailable_reason = ""
        if self.client is None:
            self.unavailable_reason = "Azure OpenAI is not configured (see .env.example)."
        elif self.pricing.find_model(self.deployment) is None:
            self.unavailable_reason = (f"The chat deployment '{self.deployment}' is missing from config/pricing.yaml "
                                       "(deployments section): its cost could not be estimated.")

    @property
    def available(self) -> bool:
        return not self.unavailable_reason

    def _messages(self, proposal, known_categories: dict[str, str]) -> list[dict]:
        return build_messages(build_context(proposal, known_categories, self.settings))

    def _key(self, messages: list[dict]) -> str:
        content = json.dumps([PROMPT_VERSION, self.deployment, messages], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def estimate(self, proposal, known_categories: dict[str, str]) -> Estimate:
        if not self.available:
            raise LLMUnavailable(self.unavailable_reason)
        messages = self._messages(proposal, known_categories)
        cost = estimate_chat(messages, self.deployment, self.pricing, self.cfg["expected_output_tokens"])
        return Estimate(cost, self.cache.get(self._key(messages)) is not None)

    def justify(self, proposal, known_categories: dict[str, str], confirmed: bool = False) -> JustificationResult:
        """Cached answer if any (free); otherwise a paid call, only with confirmed=True."""
        if not self.available:
            raise LLMUnavailable(self.unavailable_reason)
        messages = self._messages(proposal, known_categories)
        key = self._key(messages)
        cached = self.cache.get(key)
        if cached is not None:
            justification_id, answer = cached
            return JustificationResult(check_answer(answer, known_categories), justification_id, 0.0, True)
        estimate = estimate_chat(messages, self.deployment, self.pricing, self.cfg["expected_output_tokens"])
        if not confirmed:
            raise ConfirmationRequired(estimate)

        response = complete(self.client, self.deployment, messages, self.cfg["max_completion_tokens"])
        usage = Usage.from_openai(response.usage)
        record = self.tracker.record("llm-justify", "justify", self.deployment, usage,
                                     estimated_cost_usd=estimate.cost_usd, note=f"prompt {PROMPT_VERSION}")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", "stop") == "length":
            raise AnswerError("The answer was cut off by the token limit: increase llm.max_completion_tokens "
                              "in config/settings.yaml.")
        justification = check_answer(choice.message.content or "", known_categories)
        answer = justification.to_dict()
        justification_id = self.store.append(
            etf_name=proposal.name, model_version=getattr(proposal, "model_version", ""), deployment=self.deployment,
            answer=answer, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cost_usd=record["cost_usd"])
        self.cache.put(key, justification_id, answer)
        return JustificationResult(justification, justification_id, record["cost_usd"] or 0.0, False)
