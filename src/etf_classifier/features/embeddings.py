"""Bloc embeddings : vecteurs Azure OpenAI (text-embedding-3-large) du texte « Name. Description ».

Chaque lot envoyé à Azure passe par le cache (pour ne jamais repayer un texte) et par le module
costs (estimation avant, coût réel après). Le client est injectable : les tests utilisent un faux
client et ne font aucun appel réseau.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy import sparse

from etf_classifier.config import PROJECT_ROOT, resolve_path
from etf_classifier.costs.estimate import CostEstimate, estimate_embeddings
from etf_classifier.costs.pricing import PricingTable, UnknownDeploymentError, load_pricing
from etf_classifier.costs.tokens import get_counter
from etf_classifier.costs.tracker import CostTracker, Usage
from etf_classifier.features.cache import EmbeddingCache, cache_key
from etf_classifier.features.text import build_text

logger = logging.getLogger(__name__)

TOP_TICKERS = 10
EMPTY_TEXT = "[empty]"  # Azure refuse un texte vide
ENV_VARIABLES = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION")
DEFAULT_TOKENIZER = "cl100k_base"


@dataclass(frozen=True)
class EmbeddingPlan:
    """Ce que coûterait le calcul d'une liste de textes, cache déduit."""

    n_total: int
    n_cached: int
    n_to_compute: int
    estimate: CostEstimate | None


class AzureEmbedder:
    """Calcule des embeddings normalisés par lots, avec cache et suivi des coûts."""

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
                logger.warning("Texte tronqué à %d tokens pour les embeddings : %s…", self.max_tokens, text[:40])
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
        """Estimation sans aucun appel à Azure : seuls les textes absents du cache sont comptés."""
        prepared = self._prepare(texts)
        missing, _ = self._missing(prepared)
        unique = len(set(prepared))
        return EmbeddingPlan(unique, unique - len(missing), len(missing), self._estimate(missing))

    def embed(self, texts: list[str]) -> np.ndarray:
        """Matrice (n, dimensions) de vecteurs normalisés L2, dans l'ordre des textes."""
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
                raise ValueError(f"Réponse Azure inattendue : {matrix.shape} au lieu de {(len(batch), self.dimensions)}")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms > 0, norms, 1.0)
            computed = {self._key(t): v for t, v in zip(batch, matrix)}
            self.cache.put_many(computed, model=self.model_name, dimensions=self.dimensions)
            found.update(computed)
            record = self.tracker.record("features", "embeddings_batch", self.deployment,
                                         Usage.from_openai(response.usage),
                                         estimated_cost_usd=estimate.cost_usd if estimate else None,
                                         note=f"{len(batch)} textes")
            self.run_cost_usd += record["cost_usd"] or 0.0
        if not prepared:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([found[self._key(t)] for t in prepared])


def create_embedder(settings: dict, client=None) -> AzureEmbedder | None:
    """Construit l'embedder à partir de la configuration et du .env ; None si Azure n'est pas configuré."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    emb = settings["features"]["embeddings"]
    if client is None:
        missing = [name for name in ENV_VARIABLES if not os.environ.get(name)]
        if missing:
            logger.info("Azure OpenAI non configuré (variables absentes : %s) : voir .env.example", ", ".join(missing))
            return None
        from openai import AzureOpenAI

        client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                             api_key=os.environ["AZURE_OPENAI_API_KEY"],
                             api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                             max_retries=emb["max_retries"])
    costs = settings["costs"]
    pricing = load_pricing(resolve_path(costs["pricing_path"]))
    return AzureEmbedder(
        client=client,
        deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or emb["deployment"],
        dimensions=emb["dimensions"],
        batch_size=emb["batch_size"],
        max_tokens=emb["max_tokens"],
        pricing=pricing,
        tracker=CostTracker(pricing, resolve_path(costs["usage_log_path"])),
        cache=EmbeddingCache(resolve_path(emb["cache_path"])),
    )


def embedding_text(name: str, description: str, composition: dict | None = None,
                   include_top_tickers: bool = False) -> str:
    """Texte envoyé aux embeddings. La catégorie n'y figure jamais (ce serait une fuite)."""
    text = build_text(name, description)
    if include_top_tickers and composition:
        top = sorted(composition, key=composition.get, reverse=True)[:TOP_TICKERS]
        text = f"{text} Top holdings: {', '.join(top)}."
    return text


class EmbeddingsBlock:
    """Embeddings normalisés L2. Sans calculateur d'embeddings (Azure absent), le bloc est inactif."""

    name = "embeddings"

    def __init__(self, embedder, dimensions: int, include_top_tickers: bool = False):
        self.embedder = embedder
        self.dimensions = dimensions
        self.include_top_tickers = include_top_tickers

    def __getstate__(self):
        state = self.__dict__.copy()
        state["embedder"] = None  # le client Azure n'est jamais sauvegardé avec le pipeline
        return state

    def texts(self, data: pd.DataFrame) -> list[str]:
        compositions = data["composition"] if "composition" in data else [None] * len(data)
        return [embedding_text(n, d, c, self.include_top_tickers)
                for n, d, c in zip(data["name"], data["description"], compositions)]

    def fit(self, data: pd.DataFrame) -> "EmbeddingsBlock":
        return self  # rien à apprendre : le modèle d'embeddings est déjà entraîné

    def transform(self, data: pd.DataFrame) -> sparse.csr_matrix:
        if self.embedder is None:
            raise RuntimeError("Bloc embeddings sans accès Azure : recharger le pipeline avec un embedder")
        vectors = self.embedder.embed(self.texts(data))
        return sparse.csr_matrix(np.asarray(vectors, dtype=np.float32))

    def feature_names(self) -> list[str]:
        return [f"emb:{i}" for i in range(self.dimensions)]
