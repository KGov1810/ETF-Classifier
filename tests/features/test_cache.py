import numpy as np

from etf_classifier.features.cache import EmbeddingCache, cache_key


def test_vectors_are_stored_and_read_back_exactly(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    vector = np.arange(8, dtype=np.float32) / 10
    key = cache_key("model", 8, "Clean energy ETF")
    cache.put_many({key: vector}, model="model", dimensions=8)
    assert np.array_equal(cache.get_many([key])[key], vector)


def test_key_depends_on_model_dimensions_and_text():
    base = cache_key("model", 8, "text")
    assert base != cache_key("model", 16, "text")
    assert base != cache_key("other", 8, "text")
    assert base != cache_key("model", 8, "another text")


def test_missing_keys_are_absent_from_the_result(tmp_path):
    assert EmbeddingCache(tmp_path / "emb.sqlite").get_many(["missing"]) == {}


def test_cache_persists_between_instances(tmp_path):
    path = tmp_path / "emb.sqlite"
    key = cache_key("model", 4, "text")
    EmbeddingCache(path).put_many({key: np.ones(4, dtype=np.float32)}, model="model", dimensions=4)
    assert key in EmbeddingCache(path).get_many([key])


def test_many_keys_are_read_in_chunks(tmp_path):
    cache = EmbeddingCache(tmp_path / "emb.sqlite")
    items = {cache_key("m", 2, str(i)): np.full(2, i, dtype=np.float32) for i in range(2500)}
    cache.put_many(items, model="m", dimensions=2)
    assert len(cache.get_many(list(items))) == 2500
