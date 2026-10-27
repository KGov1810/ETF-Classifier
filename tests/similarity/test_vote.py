import pytest

from etf_classifier.data.split import make_split
from etf_classifier.features.pipeline import FeaturePipeline
from etf_classifier.model.baseline import FrequencyBaseline
from etf_classifier.model.evaluate import evaluate
from etf_classifier.similarity.index import Neighbor, SimilarityIndex
from etf_classifier.similarity.vote import neighbor_vote, vote_matrix


def _neighbor(score, categories):
    return Neighbor(row_id=0, ticker="X", name="x", categories=categories, score=score, composition_similarity=score,
                    text_similarity=score, overlap_pct=0.0, shared_holdings=[], text_only=False)


def test_vote_computed_by_hand():
    votes = neighbor_vote([_neighbor(0.6, ["Gold"]), _neighbor(0.3, ["ESG"]), _neighbor(0.1, ["Tech"])])
    assert votes == pytest.approx({"Gold": 0.6, "ESG": 0.3, "Tech": 0.1})
    assert list(votes) == ["Gold", "ESG", "Tech"]


def test_vote_with_zero_scores_does_not_divide_by_zero():
    assert neighbor_vote([_neighbor(0.0, ["Gold"])]) == {"Gold": 0.0}


def test_neighbor_vote_clearly_beats_the_baseline(make_synthetic, isolated_settings):
    data = make_synthetic(400, seed=0).assign(row_id=lambda d: range(len(d)))
    split = make_split(data, test_size=0.2, seed=1)
    pipeline = FeaturePipeline(isolated_settings).fit(split.train)
    index = SimilarityIndex.build(split.train, pipeline, isolated_settings)
    classes = pipeline.labels.classes
    scores = vote_matrix(index.query(split.test, pipeline, k=10), classes)
    from etf_classifier.model.classifier import decide

    y_test = pipeline.transform_labels(split.test)
    vote_f1 = evaluate(y_test, decide(scores), classes).f1_macro
    baseline = evaluate(y_test, FrequencyBaseline().fit(pipeline.transform_labels(split.train)).predict(len(y_test)),
                        classes).f1_macro
    assert vote_f1 > baseline + 0.4


def test_save_then_load_gives_the_same_neighbors(small_reference, small_pipeline, isolated_settings, tmp_path):
    index = SimilarityIndex.build(small_reference, small_pipeline, isolated_settings)
    index.save(tmp_path / "similarity.joblib")
    reloaded = SimilarityIndex.load(tmp_path / "similarity.joblib")
    before = index.query(small_reference, small_pipeline, k=2, exclude_row_ids=small_reference["row_id"])
    after = reloaded.query(small_reference, small_pipeline, k=2, exclude_row_ids=small_reference["row_id"])
    assert [[(n.row_id, round(n.score, 9)) for n in r.neighbors] for r in before] == \
           [[(n.row_id, round(n.score, 9)) for n in r.neighbors] for r in after]
