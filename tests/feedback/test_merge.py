import hashlib

from etf_classifier.data.loader import load_dataset, prepare_dataset


def test_validated_etf_joins_the_reference_dataset(store, new_etf, sample_csv, isolated_settings):
    store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    data = load_dataset(sample_csv, isolated_settings).data
    added = data[data["source"] == "validation"]
    assert list(added["name"]) == ["New Clean Energy"]
    assert added.iloc[0]["categories"] == ["Clean Energy"]
    assert (data["source"] == "reference").sum() == 15
    assert data["row_id"].is_unique


def test_cancelled_etf_is_not_in_the_reference_dataset(store, new_etf, sample_csv, isolated_settings):
    record = store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    store.cancel_validation(record["ValidationId"], cancelled_by="Marie")
    assert "New Clean Energy" not in set(load_dataset(sample_csv, isolated_settings).data["name"])


def test_original_csv_is_never_modified(store, new_etf, sample_csv, isolated_settings):
    before = hashlib.sha256(sample_csv.read_bytes()).hexdigest()
    store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    prepare_dataset(sample_csv, isolated_settings)
    assert hashlib.sha256(sample_csv.read_bytes()).hexdigest() == before


def test_new_etfs_to_predict_never_receive_the_validations(store, new_etf, sample_csv, isolated_settings):
    store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    data = load_dataset(sample_csv, isolated_settings, require_categories=False).data
    assert "New Clean Energy" not in set(data["name"])


def test_quality_rules_apply_to_journal_lines(store, new_etf, sample_csv, isolated_settings):
    store.record_validation(new_etf, ["Clean Energy"], validated_by="Kevin")
    with store.journal_path.open("a", encoding="utf-8") as handle:  # line damaged by hand
        handle.write("manual-1;Broken;Desc;[{'AAPL';['Gold'];Kevin;2026-10-25T10:00:00Z;;;no_proposal;;\n")
    result = load_dataset(sample_csv, isolated_settings)
    issue = next(i for i in result.log.exclusions() if i.source == "validation")
    assert issue.rule == "unreadable_composition"
    assert "Broken" not in set(result.data["name"])


def test_validated_etfs_reach_the_training_split(store, sample_csv, isolated_settings):
    for i in range(5):
        store.record_validation({"name": f"Clean {i}", "description": "clean energy",
                                 "composition": {"ENPH": 5.0 + i, "FSLR": 4.0, "VWS": 3.0}},
                                ["Clean Energy"], validated_by="Kevin")
    prepared = prepare_dataset(sample_csv, isolated_settings)
    assert len(prepared.split.train) + len(prepared.split.test) == 20
