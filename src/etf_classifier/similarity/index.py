"""Similarity index between ETFs: composition (cosine of the weights) and text (embeddings or TF-IDF).

The composition similarity is a cosine, hence symmetric: a small ETF whose holdings all appear in a
large ETF is not considered identical to it. The weighted overlap (data module) is only used to
explain the match and to flag near-duplicates.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from etf_classifier.data.dedup import weighted_overlap

logger = logging.getLogger(__name__)


@dataclass
class Neighbor:
    row_id: int
    ticker: str
    name: str
    categories: list[str]
    score: float
    composition_similarity: float
    text_similarity: float
    overlap_pct: float
    shared_holdings: list[tuple[str, float]]
    text_only: bool


@dataclass
class QueryResult:
    neighbors: list[Neighbor]
    near_duplicate: Neighbor | None
    text_only: bool  # the queried ETF has no composition: only the text counts


class SimilarityIndex:
    """Reference vectors and neighbour search. The pipeline is provided with every query."""

    def __init__(self):
        self.row_ids: np.ndarray = np.array([], dtype=int)
        self.tickers: list[str] = []
        self.names: list[str] = []
        self.categories: list[list[str]] = []
        self.compositions: list[dict[str, float]] = []
        self.vocabulary: dict[str, int] = {}
        self.composition_matrix = sparse.csr_matrix((0, 0))
        self.text_matrix = sparse.csr_matrix((0, 0))
        self.text_source: str | None = None
        self.weights = (0.5, 0.5)
        self.n_shared = 5
        self.duplicate_rule = (0.8, 3)
        self.chunk = 256

    # ---------- building ----------
    @classmethod
    def build(cls, reference: pd.DataFrame, pipeline, settings: dict) -> "SimilarityIndex":
        cfg, dedup = settings["similarity"], settings["dedup"]
        index = cls()
        index.row_ids = (reference["row_id"].to_numpy(dtype=int) if "row_id" in reference
                         else np.arange(len(reference)))
        index.tickers = list(reference["ticker"])
        index.names = list(reference["name"])
        index.categories = [list(c) for c in reference["categories"]]
        index.compositions = [dict(c) for c in reference["composition"]]
        tickers = sorted({t for comp in index.compositions for t in comp})
        index.vocabulary = {t: i for i, t in enumerate(tickers)}
        index.composition_matrix = index._composition_vectors(index.compositions)
        index.text_source = cls._text_block_name(pipeline)
        index.text_matrix = index._text_vectors(reference, pipeline)
        index.weights = (float(cfg["composition_weight"]), float(cfg["text_weight"]))
        index.n_shared = int(cfg["n_shared_holdings"])
        index.duplicate_rule = (float(dedup["overlap_threshold"]), int(dedup["min_common_holdings"]))
        index.chunk = int(cfg["query_chunk"])
        return index

    @staticmethod
    def _text_block_name(pipeline) -> str | None:
        names = pipeline.active_blocks
        if "embeddings" in names:
            return "embeddings"
        return "text_tfidf" if "text_tfidf" in names else None

    def _composition_vectors(self, compositions: list[dict[str, float]]) -> sparse.csr_matrix:
        """Weights of the tickers known to the index, divided by the norm of the WHOLE composition.

        An ETF with some holdings unknown to the reference thus keeps a lower similarity than an
        ETF that is fully covered.
        """
        rows, cols, values = [], [], []
        for row, comp in enumerate(compositions):
            norm = float(np.sqrt(sum(w * w for w in comp.values() if w > 0)))
            if norm == 0:
                continue
            for ticker, weight in comp.items():
                col = self.vocabulary.get(ticker)
                if col is not None and weight > 0:
                    rows.append(row)
                    cols.append(col)
                    values.append(weight / norm)
        return sparse.csr_matrix((values, (rows, cols)), shape=(len(compositions), max(len(self.vocabulary), 1)))

    def _text_vectors(self, data: pd.DataFrame, pipeline) -> sparse.csr_matrix:
        if self.text_source is None:
            return sparse.csr_matrix((len(data), 0))
        block = next(b for b in pipeline.blocks if b.name == self.text_source)
        return normalize(block.transform(data), norm="l2").tocsr().astype(np.float32)

    # ---------- search ----------
    def query(self, data: pd.DataFrame, pipeline, k: int = 5, exclude_row_ids=None) -> list[QueryResult]:
        """k neighbours per ETF, sorted by decreasing combined score.

        exclude_row_ids: row ids of the queried ETFs (same order as data) that are part of the
        reference; each one is excluded from its own neighbours.
        """
        data = data.reset_index(drop=True)
        compositions = [dict(c) for c in data["composition"]]
        query_comp = self._composition_vectors(compositions)
        query_text = self._text_vectors(data, pipeline)
        exclude = list(exclude_row_ids) if exclude_row_ids is not None else [None] * len(data)
        position = {int(r): i for i, r in enumerate(self.row_ids)}
        reference_has_comp = np.asarray(self.composition_matrix.getnnz(axis=1)) > 0
        k = min(k, len(self.row_ids) - (1 if exclude_row_ids is not None else 0))
        results: list[QueryResult] = []
        for start in range(0, len(data), self.chunk):
            stop = min(start + self.chunk, len(data))
            comp_sim = (query_comp[start:stop] @ self.composition_matrix.T).toarray()
            text_sim = ((query_text[start:stop] @ self.text_matrix.T).toarray() if self.text_source
                        else np.zeros_like(comp_sim))
            for offset in range(stop - start):
                i = start + offset
                results.append(self._neighbors_of(compositions[i], comp_sim[offset], text_sim[offset],
                                                  reference_has_comp, position.get(exclude[i]), k))
        return results

    def _neighbors_of(self, composition, comp_sim, text_sim, reference_has_comp, self_position, k) -> QueryResult:
        wc, wt = self.weights
        query_has_comp = bool(composition)
        both = reference_has_comp & query_has_comp
        # All neighbours of a query are scored on the same scale:
        # - query without a composition: text only for everyone;
        # - neighbour without a composition: its composition similarity counts as 0 in the combined score.
        if self.text_source is None:
            scores = comp_sim.copy()
        elif not query_has_comp:
            scores = text_sim.copy()
        else:
            scores = (wc * comp_sim + wt * text_sim) / (wc + wt or 1.0)
        if self_position is not None:
            scores[self_position] = -np.inf
        if k <= 0:
            return QueryResult([], None, not query_has_comp)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        neighbors = [self._explain(j, composition, scores[j], comp_sim[j], text_sim[j], not both[j]) for j in top]
        threshold, min_common = self.duplicate_rule
        # same rule as the data module: sufficient overlap AND minimum number of shared tickers
        duplicates = [n for j, n in zip(top, neighbors) if n.overlap_pct / 100 >= threshold - 1e-9
                      and len(set(composition) & set(self.compositions[j])) >= min_common]
        near_duplicate = max(duplicates, key=lambda n: n.overlap_pct) if duplicates else None
        return QueryResult(neighbors, near_duplicate, not query_has_comp)

    def _explain(self, j, composition, score, comp_sim, text_sim, text_only) -> Neighbor:
        reference = self.compositions[j]
        shared = sorted(((t, min(composition[t], reference[t])) for t in set(composition) & set(reference)),
                        key=lambda item: (-item[1], item[0]))
        return Neighbor(
            row_id=int(self.row_ids[j]), ticker=self.tickers[j], name=self.names[j], categories=self.categories[j],
            score=float(score),
            composition_similarity=float(comp_sim), text_similarity=float(text_sim),
            overlap_pct=100 * weighted_overlap(composition, reference) if composition and reference else 0.0,
            shared_holdings=shared[: self.n_shared], text_only=bool(text_only),
        )

    # ---------- saving ----------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "SimilarityIndex":
        return joblib.load(Path(path))

    @property
    def size_mb(self) -> float:
        nbytes = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
                     for m in (self.composition_matrix, self.text_matrix))
        return nbytes / 1e6
