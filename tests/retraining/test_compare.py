import json
from types import SimpleNamespace

import pandas as pd

from etf_classifier.data.loader import prepare_dataset
from etf_classifier.model.registry import load_bundle
from etf_classifier.model.train import train_and_save
from etf_classifier.retraining.compare import Scores, comparison_set, decide, score, shared_categories

CFG = {"f1_tolerance": 0.01, "category_drop_max": 0.20, "category_min_support": 5, "min_comparison_etfs": 20}


def _scores(macro, per_category):
    """per_category: {category: (f1, support)}"""
    frame = pd.DataFrame([{"category": c, "f1": f, "support": s} for c, (f, s) in per_category.items()])
    return Scores(f1_macro=macro, per_category=frame)


def _decide(current, candidate, n_etfs=50, candidate_f1=0.8, baseline_f1=0.3):
    return decide(current, candidate, candidate_test_f1=candidate_f1, baseline_test_f1=baseline_f1,
                  n_etfs=n_etfs, cfg=CFG)


def test_candidate_within_tolerance_is_promoted():
    current = _scores(0.80, {"Gold": (0.9, 10), "ESG": (0.7, 10)})
    candidate = _scores(0.795, {"Gold": (0.88, 10), "ESG": (0.71, 10)})
    decision = _decide(current, candidate)
    assert decision.promote
    assert all(ok for ok, _ in decision.checks)


def test_candidate_below_tolerance_is_refused():
    current = _scores(0.80, {"Gold": (0.9, 10)})
    candidate = _scores(0.78, {"Gold": (0.88, 10)})
    decision = _decide(current, candidate)
    assert not decision.promote
    assert any("macro F1" in reason for reason in decision.reasons)


def test_collapsing_category_is_refused_and_named():
    current = _scores(0.80, {"Gold": (0.9, 10), "ESG": (0.7, 10)})
    candidate = _scores(0.82, {"Gold": (0.95, 10), "ESG": (0.45, 10)})
    decision = _decide(current, candidate)
    assert not decision.promote
    assert any("ESG" in reason for reason in decision.reasons)


def test_small_categories_are_not_checked_for_collapse():
    current = _scores(0.80, {"Gold": (0.9, 10), "Tiny": (1.0, 2)})
    candidate = _scores(0.80, {"Gold": (0.9, 10), "Tiny": (0.0, 2)})
    assert _decide(current, candidate).promote


def test_candidate_not_better_than_the_trivial_baseline_is_refused():
    current = _scores(0.20, {"Gold": (0.2, 10)})
    candidate = _scores(0.25, {"Gold": (0.25, 10)})
    decision = _decide(current, candidate, candidate_f1=0.25, baseline_f1=0.30)
    assert not decision.promote
    assert any("trivial baseline" in reason for reason in decision.reasons)


def test_too_small_comparison_set_is_refused():
    current = _scores(0.80, {"Gold": (0.9, 10)})
    candidate = _scores(0.90, {"Gold": (0.95, 10)})
    decision = _decide(current, candidate, n_etfs=5)
    assert not decision.promote
    assert any("5 ETF" in reason for reason in decision.reasons)


def _bundle_with_names(tmp_path, names):
    path = tmp_path / "model"
    path.mkdir()
    if names is not None:
        (path / "train_names.json").write_text(json.dumps(names), encoding="utf-8")
    return SimpleNamespace(path=path)


def test_comparison_set_excludes_etfs_seen_by_the_current_model(tmp_path):
    test = pd.DataFrame({"name": ["A", "B", "C"]})
    result = comparison_set(test, _bundle_with_names(tmp_path, ["B", "Z"]))
    assert list(result.data["name"]) == ["A", "C"]
    assert result.n_seen_excluded == 1 and not result.whole_test_set


def test_comparison_set_without_training_names_uses_the_whole_test_set(tmp_path):
    test = pd.DataFrame({"name": ["A", "B"]})
    result = comparison_set(test, _bundle_with_names(tmp_path, None))
    assert list(result.data["name"]) == ["A", "B"] and result.whole_test_set


def test_shared_categories_lists_new_ones():
    current = SimpleNamespace(model=SimpleNamespace(classes=["Gold", "ESG"]))
    candidate = SimpleNamespace(model=SimpleNamespace(classes=["Gold", "ESG", "Space"]))
    shared, new, dropped = shared_categories(current, candidate)
    assert shared == ["ESG", "Gold"] and new == ["Space"] and dropped == []


def test_score_of_a_real_model_on_its_own_test_set(isolated_settings, sample_csv):
    prepared = prepare_dataset(sample_csv, isolated_settings)
    train_and_save(prepared, isolated_settings)
    bundle = load_bundle("current", isolated_settings)
    categories = sorted(bundle.model.classes)
    result = score(bundle, prepared.split.test, categories, isolated_settings)
    assert 0.0 <= result.f1_macro <= 1.0
    assert set(result.per_category["category"]) == set(categories)
    assert result.per_category["support"].sum() == sum(
        len(set(c) & set(categories)) for c in prepared.split.test["categories"])


def test_candidate_exactly_at_the_tolerance_is_promoted():
    cfg = {**CFG, "f1_tolerance": 0.25}  # 0.75 - 0.25 = 0.5 exactly in binary floating point
    current = _scores(0.75, {"Gold": (0.75, 10)})
    candidate = _scores(0.5, {"Gold": (0.6, 10)})
    assert decide(current, candidate, candidate_test_f1=0.8, baseline_test_f1=0.3, n_etfs=50, cfg=cfg).promote
