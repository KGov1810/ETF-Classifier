
import pandas as pd
import pytest

from etf_classifier.app import services
from etf_classifier.data.loader import load_dataset


def _etf(settings, name, description, holdings):
    checked = services.etf_from_inputs(name, description, pd.DataFrame(holdings, columns=["Ticker", "Weight (%)"]),
                                       settings)
    assert checked.errors == []
    return checked.data


def test_proposal_contains_categories_scores_neighbours_and_version(trained_settings):
    bundle = services.load_current_model(trained_settings)
    data = _etf(trained_settings, "New Gold", "Global gold miners", [["NEM", 11.0], ["GOLD", 9.0], ["AEM", 8.0]])
    proposal = services.propose(data, bundle, trained_settings)[0]
    assert 1 <= len(proposal.categories) <= trained_settings["model"]["max_categories"]
    assert len(proposal.scores) <= trained_settings["app"]["n_scores_shown"]
    assert proposal.scores == sorted(proposal.scores, key=lambda item: -item[1])
    assert len(proposal.neighbors) == trained_settings["similarity"]["n_neighbors"]
    assert proposal.model_version == bundle.version
    assert proposal.neighbors[0].name == "Gold Miners"


def test_copy_of_an_existing_etf_raises_the_near_duplicate_alert(trained_settings):
    bundle = services.load_current_model(trained_settings)
    data = _etf(trained_settings, "World Clone", "Tracks the MSCI World index",
                [["AAPL", 5.0], ["MSFT", 4.5], ["NVDA", 4.0], ["AMZN", 3.0], ["GOOGL", 2.5]])
    proposal = services.propose(data, bundle, trained_settings)[0]
    assert proposal.near_duplicate is not None
    assert proposal.near_duplicate.name.startswith("World Tracker")


def test_typed_etf_and_csv_etf_get_the_same_proposal(trained_settings, sample_csv):
    bundle = services.load_current_model(trained_settings)
    from_csv = load_dataset(sample_csv, trained_settings).data
    from_csv = from_csv[from_csv["name"] == "Europe Dividend"].reset_index(drop=True)
    typed = _etf(trained_settings, "Europe Dividend", "High-dividend European equities",
                 [["NESN", 4.0], ["TTE", 3.8], ["ALV", 3.5], ["SAN", 3.0]])
    a = services.propose(from_csv, bundle, trained_settings)[0]
    b = services.propose(typed, bundle, trained_settings)[0]
    assert a.categories == b.categories
    assert a.scores == pytest.approx(b.scores)


def test_local_model_needs_no_cost_confirmation(trained_settings):
    bundle = services.load_current_model(trained_settings)
    data = _etf(trained_settings, "X", "desc", [["AAPL", 5.0]])
    assert services.estimate(data, bundle, trained_settings) is None


def test_embeddings_model_never_calls_azure_without_confirmation(trained_emb_settings, fake_client):
    bundle = services.load_current_model(trained_emb_settings, client=fake_client)
    data = _etf(trained_emb_settings, "Brand New Theme", "Space exploration companies", [["RKLB", 6.0]])
    calls_before = len(fake_client.calls)
    cost = services.estimate(data, bundle, trained_emb_settings)
    assert cost.n_to_compute == 1
    with pytest.raises(services.CostConfirmationRequired):
        services.propose(data, bundle, trained_emb_settings)
    assert len(fake_client.calls) == calls_before
    services.propose(data, bundle, trained_emb_settings, confirmed=True)
    assert len(fake_client.calls) == calls_before + 1


def test_cached_texts_cost_nothing_and_need_no_confirmation(trained_emb_settings, fake_client, sample_csv):
    bundle = services.load_current_model(trained_emb_settings, client=fake_client)
    known = load_dataset(sample_csv, trained_emb_settings).data.head(1).reset_index(drop=True)
    cost = services.estimate(known, bundle, trained_emb_settings)
    assert cost.n_to_compute == 0
    services.propose(known, bundle, trained_emb_settings)  # no exception: nothing to pay


def test_embeddings_model_without_azure_gives_a_clear_error(trained_emb_settings, monkeypatch):
    for variable in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("etf_classifier.azure.load_dotenv", lambda *a, **k: None)
    with pytest.raises(services.AppError, match="Azure"):
        services.load_current_model(trained_emb_settings)


def test_no_current_model_gives_a_clear_error(isolated_settings):
    with pytest.raises(services.AppError, match="No current model"):
        services.load_current_model(isolated_settings)


def test_model_without_similarity_index_still_proposes(trained_settings):
    bundle = services.load_current_model(trained_settings)
    (bundle.path / "similarity.joblib").unlink()
    bundle = services.load_current_model(trained_settings)
    data = _etf(trained_settings, "X", "desc", [["AAPL", 5.0]])
    proposal = services.propose(data, bundle, trained_settings)[0]
    assert proposal.categories
    assert proposal.neighbors == [] and not proposal.similarity_available
