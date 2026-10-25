import yaml

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.model.registry import load_bundle
from etf_classifier.model.train import main as train_main
from etf_classifier.similarity.index import SimilarityIndex


def _train(sample_csv, settings, tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    assert train_main(["--csv", str(sample_csv), "--config", str(config), "--no-azure"]) == 0


def test_training_saves_the_similarity_index_in_the_model_folder(sample_csv, isolated_settings, tmp_path):
    _train(sample_csv, isolated_settings, tmp_path)
    bundle = load_bundle("current", isolated_settings)
    assert (bundle.path / "similarity.joblib").exists()
    assert bundle.similarity is not None


def test_reloaded_index_gives_the_same_neighbors_as_a_fresh_one(sample_csv, isolated_settings, tmp_path):
    _train(sample_csv, isolated_settings, tmp_path)
    bundle = load_bundle("current", isolated_settings)
    data = prepare_dataset(sample_csv, isolated_settings).data
    fresh = SimilarityIndex.build(data, bundle.pipeline, isolated_settings)
    loaded = bundle.similarity.query(data, bundle.pipeline, k=3, exclude_row_ids=data["row_id"])
    rebuilt = fresh.query(data, bundle.pipeline, k=3, exclude_row_ids=data["row_id"])
    assert [[n.row_id for n in r.neighbors] for r in loaded] == [[n.row_id for n in r.neighbors] for r in rebuilt]


def test_model_folder_without_index_still_loads(sample_csv, isolated_settings, tmp_path):
    _train(sample_csv, isolated_settings, tmp_path)
    bundle = load_bundle("current", isolated_settings)
    (bundle.path / "similarity.joblib").unlink()
    assert load_bundle("current", isolated_settings).similarity is None
