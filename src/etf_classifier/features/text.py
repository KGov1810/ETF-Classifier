"""text_tfidf block: TF-IDF of the name and the description (English)."""

import logging

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def build_text(name: str, description: str) -> str:
    """Text of an ETF: "Name. Description" (the category is never included)."""
    name, description = (name or "").strip(), (description or "").strip()
    return f"{name}. {description}" if name and description else name or description


def texts_of(data: pd.DataFrame) -> list[str]:
    return [build_text(n, d) for n, d in zip(data["name"], data["description"])]


class TextTfidfBlock:
    """Words and word pairs weighted by TF-IDF, English stop words removed, L2-normalised rows."""

    name = "text_tfidf"

    def __init__(self, ngram_max: int = 2, min_df: int = 2, max_features: int = 50000):
        self.params = {"ngram_max": ngram_max, "min_df": min_df, "max_features": max_features}
        self._vectorizer: TfidfVectorizer | None = None

    def _make(self, min_df: int) -> TfidfVectorizer:
        return TfidfVectorizer(stop_words="english", ngram_range=(1, self.params["ngram_max"]), min_df=min_df,
                               max_features=self.params["max_features"], sublinear_tf=True, norm="l2")

    def fit(self, data: pd.DataFrame) -> "TextTfidfBlock":
        texts = texts_of(data)
        try:
            self._vectorizer = self._make(self.params["min_df"]).fit(texts)
        except ValueError:
            # Training set too small for min_df: accept terms seen only once
            logger.warning("Empty vocabulary with min_df=%s: falling back to min_df=1", self.params["min_df"])
            self._vectorizer = self._make(1).fit(texts)
        return self

    def transform(self, data: pd.DataFrame) -> sparse.csr_matrix:
        return self._vectorizer.transform(texts_of(data)).tocsr()

    def feature_names(self) -> list[str]:
        return [f"text:{term}" for term in self._vectorizer.get_feature_names_out()]
