"""embeddings block: Azure OpenAI vectors (text-embedding-3-large) of the text "Name. Description".

Every batch sent to Azure goes through the cache (so a text is never paid for twice) and through
the costs module (estimate before, actual cost after). The client is injectable: the tests use a
fake client and make no network call.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from etf_classifier.azure import azure_client, deployment_name
from etf_classifier.config import resolve_path
from etf_classifier.costs.estimate import CostEstimate, estimate_embeddings
from etf_classifier.costs.pricing import PricingTable, UnknownDeploymentError, load_pricing
from etf_classifier.costs.tokens import get_counter
from etf_classifier.costs.tracker import CostTracker, Usage
from etf_classifier.features.cache import EmbeddingCache, cache_key
from etf_classifier.features.text import build_text

logger = logging.getLogger(__name__)

TOP_TICKERS = 10
EMPTY_TEXT = "[empty]"  # Azure rejects an empty text
DEFAULT_TOKENIZER = "cl100k_base"


@dataclass(frozen=True)
class EmbeddingPlan:
    """What computing a list of texts would cost, cache deducted."""

    n_total: int
    n_cached: int
    n_to_compute: int
    estimate: CostEstimate | None


class AzureEmbedder:
    """Computes normalised embeddings in batches, with caching and cost tracking."""

    def __init__(self, client, deployment: str, dimensions: int, batch_size: int, max_tokens: int,
                 pricing: PricingTable, tracker: CostTracker, cache: EmbeddingCache):
        self.client = client
        self.deployment = deployment
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.pricing = pricing
        self.tracker = tracker
        self.cache = cache
        model = pricing.find_model(deployment)
        self.model_name = model.name if model else deployment
        self.counter = get_counter(model.tokenizer if model else DEFAULT_TOKENIZER)
        self.n_api_calls = 0
        self.run_cost_usd = 0.0

    def _prepare(self, texts: list[str]) -> list[str]:
        prepared = []
        for text in texts:
            text = text.strip() or EMPTY_TEXT
            truncated_text, truncated = self.counter.truncate(text, self.max_tokens)
            if truncated:
                logger.warning("Text truncated to %d tokens for the embeddings: %s…", self.max_tokens, text[:40])
            prepared.append(truncated_text)
        return prepared

    def _key(self, text: str) -> str:
        return cache_key(self.model_name, self.dimensions, text)

    def _estimate(self, texts: list[str]) -> CostEstimate | None:
        try:
            return estimate_embeddings(texts, self.deployment, self.pricing)
        except UnknownDeploymentError:
            return None

    def _missing(self, texts: list[str]) -> tuple[list[str], dict[str, np.ndarray]]:
        found = self.cache.get_many(self._key(t) for t in texts)
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in found]
        return missing, found

    def plan(self, texts: list[str]) -> EmbeddingPlan:
        """Estimate without any Azure call: only texts missing from the cache are counted."""
        prepared = self._prepare(texts)
        missing, _ = self._missing(prepared)
        unique = len(set(prepared))
        return EmbeddingPlan(unique, unique - len(missing), len(missing), self._estimate(missing))

    def embed(self, texts: list[str]) -> np.ndarray:
        """(n, dimensions) matrix of L2-normalised vectors, in the order of the texts."""
        prepared = self._prepare(texts)
        missing, found = self._missing(prepared)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            estimate = self._estimate(batch)
            response = self.client.embeddings.create(model=self.deployment, input=batch, dimensions=self.dimensions)
            self.n_api_calls += 1
            vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.shape != (len(batch), self.dimensions):
                expected = (len(batch), self.dimensions)
                raise ValueError(f"Unexpected Azure response: {matrix.shape} instead of {expected}")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms > 0, norms, 1.0)
            computed = {self._key(t): v for t, v in zip(batch, matrix)}
            self.cache.put_many(computed, model=self.model_name, dimensions=self.dimensions)
            found.update(computed)
            record = self.tracker.record("features", "embeddings_batch", self.deployment,
                                         Usage.from_openai(response.usage),
                                         estimated_cost_usd=estimate.cost_usd if estimate else None,
                                         note=f"{len(batch)} texts")
            self.run_cost_usd += record["cost_usd"] or 0.0
        if not prepared:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([found[self._key(t)] for t in prepared])


def create_embedder(settings: dict, client=None) -> AzureEmbedder | None:
    """Builds the embedder from the configuration and .env; None if Azure is not configured."""
    emb = settings["features"]["embeddings"]
    client = client if client is not None else azure_client(emb["max_retries"])
    if client is None:
        return None
    costs = settings["costs"]
    pricing = load_pricing(resolve_path(costs["pricing_path"]))
    return AzureEmbedder(
        client=client,
        deployment=deployment_name("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", emb["deployment"]),
        dimensions=emb["dimensions"],
        batch_size=emb["batch_size"],
        max_tokens=emb["max_tokens"],
        pricing=pricing,
        tracker=CostTracker(pricing, resolve_path(costs["usage_log_path"])),
        cache=EmbeddingCache(resolve_path(emb["cache_path"])),
    )


def embedding_text(name: str, description: str, composition: dict | None = None,
                   include_top_tickers: bool = False) -> str:
    """Text sent to the embeddings. The category is never included (that would be a leak)."""
    text = build_text(name, description)
    if include_top_tickers and composition:
        top = sorted(composition, key=composition.get, reverse=True)[:TOP_TICKERS]
        text = f"{text} Top holdings: {', '.join(top)}."
    return text


class EmbeddingsBlock:
    """L2-normalised embeddings. Without an embedder (Azure not available), the block is inactive."""

    name = "embeddings"

    def __init__(self, embedder, dimensions: int, include_top_tickers: bool = False):
        self.embedder = embedder
        self.dimensions = dimensions
        self.include_top_tickers = include_top_tickers

    def __getstate__(self):
        state = self.__dict__.copy()
        state["embedder"] = None  # the Azure client is never saved with the pipeline
        return state

    def texts(self, data: pd.DataFrame) -> list[str]:
        compositions = data["composition"] if "composition" in data else [None] * len(data)
        return [embedding_text(n, d, c, self.include_top_tickers)
                for n, d, c in zip(data["name"], data["description"], compositions)]

    def fit(self, data: pd.DataFrame) -> "EmbeddingsBlock":
        return self  # nothing to learn: the embedding model is already trained

    def transform(self, data: pd.DataFrame) -> sparse.csr_matrix:
        if self.embedder is None:
            raise RuntimeError("Embeddings block without Azure access: reload the pipeline with an embedder")
        vectors = self.embedder.embed(self.texts(data))
        return sparse.csr_matrix(np.asarray(vectors, dtype=np.float32))

    def feature_names(self) -> list[str]:
        return [f"emb:{i}" for i in range(self.dimensions)]


# --- Command-line helpers: cost announcement and confirmation before spending ---------------------

EXIT_CANCELLED, EXIT_NO_AZURE = 1, 3  # exit codes shared by every command that may call Azure


class CommandStopped(Exception):
    """A command stops before any Azure call (exit_code: 1 = cancelled, 3 = Azure not configured)."""

    def __init__(self, exit_code: int):
        super().__init__(exit_code)
        self.exit_code = exit_code


def plan_embeddings(settings: dict, data, embedder) -> EmbeddingPlan:
    """What computing the embeddings of these ETFs would cost (cache deducted), without calling Azure."""
    texts = EmbeddingsBlock(None, embedder.dimensions,
                            settings["features"]["embeddings"]["include_top_tickers"]).texts(data)
    return embedder.plan(texts)


def announce_embedding_plan(settings: dict, data, embedder) -> tuple[EmbeddingPlan, str]:
    """Shows what computing the embeddings would cost (cache deducted), without calling Azure."""
    plan = plan_embeddings(settings, data, embedder)
    cost = f"${plan.estimate.cost_usd:.4f}" if plan.estimate else "unknown (deployment missing from the price list)"
    approx = " (approximate)" if plan.estimate and plan.estimate.approximate else ""
    tokens = plan.estimate.input_tokens if plan.estimate else "?"
    print(f"Embeddings: {plan.n_to_compute} text(s) to compute, {plan.n_cached} already cached · "
          f"{tokens} tokens · estimated cost {cost}{approx}")
    return plan, cost


def confirm_spending(plan: EmbeddingPlan, cost: str, assume_yes: bool, input_fn: Callable[[str], str]) -> bool:
    """Asks for confirmation before spending on Azure (unless there is nothing to compute or assume_yes)."""
    if not plan.n_to_compute or assume_yes:
        return True
    answer = input_fn(f"Confirm the estimated spending of {cost}? [y/N] ").strip().lower()
    if answer in {"y", "yes"}:
        return True
    print("Cancelled: no Azure call made.")
    return False


def embedder_for_command(settings: dict, data, assume_yes: bool, input_fn: Callable[[str], str], client=None,
                         enabled: bool = True, required: bool = False) -> AzureEmbedder | None:
    """Embedder for a command: announces the cost of the texts not cached yet and asks for confirmation.

    None if the embeddings are disabled, or if Azure is not configured and not required.
    Raises CommandStopped when the user cancels, or when a required Azure configuration is missing.
    """
    if not enabled:
        return None
    embedder = create_embedder(settings, client=client)
    if embedder is None:
        if required:
            print("Error: this model uses Azure embeddings, but Azure OpenAI is not configured "
                  "(see .env.example).", file=sys.stderr)
            raise CommandStopped(EXIT_NO_AZURE)
        return None
    plan, cost = announce_embedding_plan(settings, data, embedder)
    if not confirm_spending(plan, cost, assume_yes, input_fn):
        raise CommandStopped(EXIT_CANCELLED)
    return embedder
