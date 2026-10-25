"""Index de similarité entre ETF : composition (cosinus des poids) et texte (embeddings ou TF-IDF).

La similarité de composition est un cosinus, donc symétrique : un petit ETF dont toutes les lignes
figurent dans un grand ETF n'est pas jugé identique à lui. Le recouvrement pondéré (module data)
sert seulement à expliquer le rapprochement et à signaler les quasi-doublons.
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
    text_only: bool  # l'ETF recherché n'a pas de composition : seul le texte compte


class SimilarityIndex:
    """Vecteurs de référence et recherche des voisins. Le pipeline est fourni à chaque recherche."""

    def __init__(self):
        self.row_ids: np.ndarray = np.array([], dtype=int)
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

    # ---------- construction ----------
    @classmethod
    def build(cls, reference: pd.DataFrame, pipeline, settings: dict) -> "SimilarityIndex":
        cfg, dedup = settings["similarity"], settings["dedup"]
        index = cls()
        index.row_ids = (reference["row_id"].to_numpy(dtype=int) if "row_id" in reference
                         else np.arange(len(reference)))
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
        """Poids des tickers connus de l'index, divisés par la norme de TOUTE la composition.

        Un ETF dont une partie des lignes est inconnue de la référence garde ainsi une similarité
        plus faible qu'un ETF entièrement couvert.
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

    # ---------- recherche ----------
    def query(self, data: pd.DataFrame, pipeline, k: int = 5, exclude_row_ids=None) -> list[QueryResult]:
        """k voisins par ETF, triés par score combiné décroissant.

        exclude_row_ids : identifiants de ligne des ETF recherchés (même ordre que data) qui font
        partie de la référence ; chacun est exclu de ses propres voisins.
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
                results.append(self._neighbors_of(i, compositions[i], comp_sim[offset], text_sim[offset],
                                                  reference_has_comp, position.get(exclude[i]), k))
        return results

    def _neighbors_of(self, i, composition, comp_sim, text_sim, reference_has_comp, self_position, k) -> QueryResult:
        wc, wt = self.weights
        query_has_comp = bool(composition)
        both = reference_has_comp & query_has_comp
        # Tous les voisins d'une même recherche sont notés sur la même échelle :
        # - recherche sans composition : texte seul pour tout le monde ;
        # - voisin sans composition : sa similarité de composition vaut 0 dans le score combiné.
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
        # même règle que le module data : recouvrement suffisant ET nombre minimal de tickers communs
        duplicates = [n for n in neighbors if n.overlap_pct / 100 >= threshold - 1e-9
                      and self._common_count(composition, n) >= min_common]
        near_duplicate = max(duplicates, key=lambda n: n.overlap_pct) if duplicates else None
        return QueryResult(neighbors, near_duplicate, not query_has_comp)

    def _common_count(self, composition: dict, neighbor: Neighbor) -> int:
        position = int(np.flatnonzero(self.row_ids == neighbor.row_id)[0])
        return len(set(composition) & set(self.compositions[position]))

    def _explain(self, j, composition, score, comp_sim, text_sim, text_only) -> Neighbor:
        reference = self.compositions[j]
        shared = sorted(((t, min(composition[t], reference[t])) for t in set(composition) & set(reference)),
                        key=lambda item: (-item[1], item[0]))
        return Neighbor(
            row_id=int(self.row_ids[j]), name=self.names[j], categories=self.categories[j], score=float(score),
            composition_similarity=float(comp_sim), text_similarity=float(text_sim),
            overlap_pct=100 * weighted_overlap(composition, reference) if composition and reference else 0.0,
            shared_holdings=shared[: self.n_shared], text_only=bool(text_only),
        )

    # ---------- sauvegarde ----------
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
