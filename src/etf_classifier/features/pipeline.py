"""FeaturePipeline: fits the blocks on the training set, then applies them to any dataset.

Everything that is learned (vocabularies, IDF, scaling, list of categories) comes from the training
set only: no information from the test set leaks into the features.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from etf_classifier.config import resolve_path
from etf_classifier.features.embeddings import EmbeddingsBlock
from etf_classifier.features.holdings import CompositionStatsBlock, HoldingsBlock
from etf_classifier.features.labels import CategoryEncoder
from etf_classifier.features.text import TextTfidfBlock

logger = logging.getLogger(__name__)

PIPELINE_FILE = "pipeline.joblib"
METADATA_FILE = "metadata.json"


@dataclass
class FeatureSet:
    """Features of a dataset: one matrix per block, the weighted combined matrix and the column names."""

    blocks: dict[str, sparse.csr_matrix]
    matrix: sparse.csr_matrix
    feature_names: list[str]


def training_fingerprint(data: pd.DataFrame) -> str:
    """Fingerprint of the training set (names and categories), to trace which dataset was used."""
    content = "\n".join(f"{n}|{','.join(c)}" for n, c in zip(data["name"], data["categories"]))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Fingerprint of a whole prepared dataset (names, descriptions, compositions, categories), order-independent.

    Used by the daily retraining to detect whether anything changed since the last model.
    """
    rows = sorted(
        f"{n}|{d}|{sorted(c.items())}|{sorted(k)}"
        for n, d, c, k in zip(data["name"], data["description"], data["composition"], data["categories"])
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


class FeaturePipeline:
    def __init__(self, settings: dict, embedder=None):
        self.settings = settings
        cfg = settings["features"]
        self.weights = {name: float(block["weight"]) for name, block in cfg["blocks"].items()}
        self.blocks = self._make_blocks(cfg, embedder)
        self.labels = CategoryEncoder()
        self.metadata: dict = {}

    @property
    def active_blocks(self) -> list[str]:
        return [block.name for block in self.blocks]

    @staticmethod
    def _make_blocks(cfg: dict, embedder) -> list:
        enabled = {name for name, block in cfg["blocks"].items() if block["enabled"]}
        blocks = []
        if "text_tfidf" in enabled:
            text = cfg["text_tfidf"]
            blocks.append(TextTfidfBlock(text["ngram_max"], text["min_df"], text["max_features"]))
        if "holdings" in enabled:
            blocks.append(HoldingsBlock(cfg["holdings"]["min_df"]))
        if "composition_stats" in enabled:
            blocks.append(CompositionStatsBlock())
        if "embeddings" in enabled:
            if embedder is None:
                logger.warning("Embeddings block disabled: Azure OpenAI access not configured (see .env.example)")
            else:
                emb = cfg["embeddings"]
                blocks.append(EmbeddingsBlock(embedder, emb["dimensions"], emb["include_top_tickers"]))
        return blocks

    def fit(self, train: pd.DataFrame) -> "FeaturePipeline":
        for block in self.blocks:
            block.fit(train)
        self.labels.fit(train)
        self.metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_train": len(train),
            "training_fingerprint": training_fingerprint(train),
            "blocks": {b.name: {"weight": self.weights[b.name], "n_features": len(b.feature_names())}
                       for b in self.blocks},
            "categories": self.labels.classes,
            "settings": self.settings["features"],
        }
        return self

    def transform(self, data: pd.DataFrame) -> FeatureSet:
        matrices = {block.name: block.transform(data) for block in self.blocks}
        weighted = [matrices[name] * self.weights[name] for name in matrices]
        names = [n for block in self.blocks for n in block.feature_names()]
        combined = sparse.hstack(weighted, format="csr") if weighted else sparse.csr_matrix((len(data), 0))
        return FeatureSet(blocks=matrices, matrix=combined.astype(np.float32), feature_names=names)

    def fit_transform(self, train: pd.DataFrame) -> FeatureSet:
        return self.fit(train).transform(train)

    def transform_labels(self, data: pd.DataFrame) -> np.ndarray:
        return self.labels.transform(data)

    def save(self, path: str | Path | None = None) -> Path:
        """Saves to artifacts/features/<version>/ (without the Azure client)."""
        if path is None:
            version = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = resolve_path(self.settings["features"]["artifacts_dir"]) / version
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / PIPELINE_FILE)
        (path / METADATA_FILE).write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @property
    def uses_embeddings(self) -> bool:
        return any(isinstance(block, EmbeddingsBlock) for block in self.blocks)

    def attach_embedder(self, embedder) -> None:
        """Reattaches the embedder after reloading (it is never saved)."""
        for block in self.blocks:
            if isinstance(block, EmbeddingsBlock):
                block.embedder = embedder

    @classmethod
    def load(cls, path: str | Path, embedder=None) -> "FeaturePipeline":
        """Reloads a pipeline; provide an embedder if it contains the embeddings block."""
        pipeline: FeaturePipeline = joblib.load(Path(path) / PIPELINE_FILE)
        pipeline.attach_embedder(embedder)
        return pipeline
