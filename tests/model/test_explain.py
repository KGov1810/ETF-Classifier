from etf_classifier.data.split import make_split
from etf_classifier.model.classifier import tune
from etf_classifier.model.explain import top_features


def test_influential_features_are_the_category_signatures(synthetic_data, isolated_settings):
    split = make_split(synthetic_data, test_size=0.2, seed=1)
    result = tune(split.train, isolated_settings)
    names = result.pipeline.transform(split.train.head(1)).feature_names
    top = top_features(result.model, names, k=10)
    gold_signature = {"text:gold", "text:bullion", "text:miners", "holding:NEM", "holding:GOLD",
                      "holding:AEM", "holding:KGC"}
    found = {name for name, _ in top["Gold"]}
    assert len(found & gold_signature) >= 4
    assert all(weight > 0 for _, weight in top["Gold"])


def test_constant_category_has_no_influential_feature():
    class FakeModel:  # a single training category: nothing to learn, no coefficients
        classes = ["Always"]

        @staticmethod
        def coefficients():
            return {"Always": None}

    assert top_features(FakeModel(), ["a"], k=10) == {"Always": []}
