from pathlib import Path

import yaml

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.similarity.evaluate import main, run_similarity_evaluation


def test_evaluation_index_contains_no_test_etf(sample_csv, sim_settings):
    prepared = prepare_dataset(sample_csv, sim_settings)
    outcome = run_similarity_evaluation(prepared, sim_settings, embedder=None)
    assert set(outcome.index_row_ids).isdisjoint(prepared.split.test["row_id"])
    assert set(outcome.index_row_ids) == set(prepared.split.train["row_id"])


def test_report_compares_vote_model_and_baseline(sample_csv, sim_settings, tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump(sim_settings, allow_unicode=True), encoding="utf-8")
    assert main(["--csv", str(sample_csv), "--config", str(config)]) == 0
    report = (Path(sim_settings["model"]["reports_dir"]) / "similarity_evaluation.md").read_text(encoding="utf-8")
    for label in ["Neighbour vote (k = 5)", "Neighbour vote (k = 20)", "Model (logistic regression)",
                  "Trivial baseline", "fingerprint"]:
        assert label in report
