import copy

import numpy as np

from etf_classifier.features.embeddings import create_embedder
from etf_classifier.features.pipeline import FeaturePipeline
from etf_classifier.similarity.index import SimilarityIndex


def test_vocabulary_covers_tickers_unknown_to_the_model(small_reference, isolated_settings):
    pipeline = FeaturePipeline(isolated_settings).fit(small_reference.iloc[:2])  # the model does not see NEM
    index = SimilarityIndex.build(small_reference, pipeline, isolated_settings)
    assert "NEM" in index.vocabulary


def test_text_source_is_tfidf_without_embeddings(small_reference, small_pipeline, isolated_settings):
    assert SimilarityIndex.build(small_reference, small_pipeline, isolated_settings).text_source == "text_tfidf"


def test_text_source_is_embeddings_when_the_model_uses_them(small_reference, isolated_settings, fake_client):
    cfg = copy.deepcopy(isolated_settings)
    cfg["features"]["blocks"]["embeddings"]["enabled"] = True
    pipeline = FeaturePipeline(cfg, embedder=create_embedder(cfg, client=fake_client)).fit(small_reference)
    assert SimilarityIndex.build(small_reference, pipeline, cfg).text_source == "embeddings"


def test_empty_composition_has_a_null_vector(small_reference, small_pipeline, isolated_settings):
    index = SimilarityIndex.build(small_reference, small_pipeline, isolated_settings)
    assert index.composition_matrix[3].nnz == 0
    assert np.allclose(np.linalg.norm(index.composition_matrix[:3].toarray(), axis=1), 1.0)


def test_row_ids_default_to_positions_when_absent(small_reference, small_pipeline, isolated_settings):
    reference = small_reference.drop(columns=["row_id"])
    index = SimilarityIndex.build(reference, small_pipeline, isolated_settings)
    assert list(index.row_ids) == [0, 1, 2, 3]
