import copy
import time

import numpy as np
import pandas as pd
import pytest

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.features.pipeline import FeaturePipeline


@pytest.fixture
def prepared(sample_csv, settings):
    return prepare_dataset(sample_csv, settings)


@pytest.fixture
def local_settings(settings):
    """Configuration sans Azure : les trois blocs locaux."""
    cfg = copy.deepcopy(settings)
    cfg["features"]["blocks"]["embeddings"]["enabled"] = False
    cfg["features"]["text_tfidf"]["min_df"] = 1
    return cfg


def test_pipeline_combines_the_three_local_blocks(prepared, local_settings):
    pipeline = FeaturePipeline(local_settings).fit(prepared.split.train)
    features = pipeline.transform(prepared.split.test)
    assert set(features.blocks) == {"text_tfidf", "holdings", "composition_stats"}
    assert features.matrix.shape == (len(prepared.split.test), len(features.feature_names))


def test_same_training_gives_the_same_matrix(prepared, local_settings):
    first = FeaturePipeline(local_settings).fit(prepared.split.train).transform(prepared.data).matrix
    second = FeaturePipeline(local_settings).fit(prepared.split.train).transform(prepared.data).matrix
    assert (first != second).nnz == 0


def test_save_then_load_gives_an_identical_transformation(prepared, local_settings, tmp_path):
    pipeline = FeaturePipeline(local_settings).fit(prepared.split.train)
    path = pipeline.save(tmp_path / "pipeline")
    reloaded = FeaturePipeline.load(path)
    before = pipeline.transform(prepared.data).matrix
    after = reloaded.transform(prepared.data).matrix
    assert (before != after).nnz == 0
    assert (path / "metadata.json").exists()


def test_disabled_block_disappears_from_the_combined_matrix(prepared, local_settings):
    local_settings["features"]["blocks"]["holdings"]["enabled"] = False
    features = FeaturePipeline(local_settings).fit(prepared.split.train).transform(prepared.data)
    assert "holdings" not in features.blocks
    assert not any(name.startswith("holding:") for name in features.feature_names)


def test_block_weight_scales_its_columns(prepared, local_settings):
    base = FeaturePipeline(local_settings).fit(prepared.split.train).transform(prepared.data)
    local_settings["features"]["blocks"]["holdings"]["weight"] = 2.0
    weighted = FeaturePipeline(local_settings).fit(prepared.split.train).transform(prepared.data)
    columns = [i for i, n in enumerate(base.feature_names) if n.startswith("holding:")]
    assert np.allclose(weighted.matrix[:, columns].toarray(), 2 * base.matrix[:, columns].toarray())


def test_embeddings_block_is_skipped_without_azure(prepared, settings, caplog):
    pipeline = FeaturePipeline(settings, embedder=None).fit(prepared.split.train)
    assert "embeddings" not in pipeline.active_blocks
    assert "Azure" in caplog.text


def test_labels_are_learned_on_training_only(prepared, local_settings):
    pipeline = FeaturePipeline(local_settings).fit(prepared.split.train)
    labels = pipeline.transform_labels(prepared.split.test)
    assert labels.shape == (len(prepared.split.test), len(pipeline.labels.classes))


@pytest.mark.slow
def test_local_blocks_on_10000_etfs_in_less_than_60_seconds(local_settings):
    rng = np.random.default_rng(0)
    words = [f"word{i}" for i in range(3000)]
    rows = []
    for i in range(10_000):
        size = int(rng.integers(10, 120))
        tickers = rng.choice(8000, size=size, replace=False)
        rows.append({
            "name": f"ETF {i}",
            "description": " ".join(rng.choice(words, size=30)),
            "composition": {f"T{t}": float(w) for t, w in zip(tickers, rng.dirichlet(np.ones(size)) * 80)},
            "categories": [f"Cat{i % 20}"],
        })
    data = pd.DataFrame(rows)
    start = time.perf_counter()
    FeaturePipeline(local_settings).fit(data.iloc[:8000]).transform(data)
    assert time.perf_counter() - start < 60
