"""Page tests with Streamlit's AppTest (no browser). Settings point to a temporary folder."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from etf_classifier.feedback.store import FeedbackStore

APP_DIR = Path(__file__).parents[2] / "src" / "etf_classifier" / "app"
PAGES = ["Home.py", "pages/1_Single_ETF.py", "pages/2_Batch.py", "pages/3_History.py", "pages/4_Monitoring.py"]


@pytest.fixture
def use_settings(monkeypatch, tmp_path):
    """Writes the given settings to a YAML file and points the app to it."""
    def _use(settings: dict) -> dict:
        path = tmp_path / "app_settings.yaml"
        path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
        monkeypatch.setenv("ETF_CLASSIFIER_CONFIG", str(path))
        return settings
    return _use


def _app(page: str) -> AppTest:
    return AppTest.from_file(str(APP_DIR / page), default_timeout=60)


def _texts(at: AppTest, kind: str) -> str:
    return " ".join(element.value for element in getattr(at, kind))


def _existing_pages():
    return [page for page in PAGES if (APP_DIR / page).exists()]


@pytest.mark.parametrize("page", PAGES[:2])
def test_page_renders_without_a_model(page, app_settings, use_settings):
    use_settings(app_settings)
    at = _app(page).run()
    assert not at.exception
    assert "No current model" in _texts(at, "error") + _texts(at, "warning") + _texts(at, "info")


@pytest.mark.parametrize("page", PAGES[:2])
def test_page_renders_with_a_model(page, trained_settings, use_settings):
    use_settings(trained_settings)
    at = _app(page).run()
    assert not at.exception
    assert not at.error


def test_sidebar_offers_the_configured_validators(trained_settings, use_settings):
    use_settings(trained_settings)
    at = _app("Home.py").run()
    assert at.sidebar.selectbox(key="validator").options == ["—", "Alice Martin", "Bruno Leroy", "Chloe Dubois"]


def _fill_single_etf(at: AppTest) -> AppTest:
    at.session_state["single_holdings_init"] = pd.DataFrame(
        {"Ticker": ["NEM", "GOLD", "AEM"], "Weight (%)": [11.0, 9.0, 8.0]})
    at.run()
    at.text_input(key="single_name").input("New Gold Miners")
    at.text_area(key="single_description").input("Global gold mining companies")
    at.button(key="single_propose").click()
    return at.run()


def test_single_etf_propose_then_validate_writes_one_journal_line(trained_settings, use_settings):
    use_settings(trained_settings)
    at = _app("pages/1_Single_ETF.py").run()
    at.sidebar.selectbox(key="validator").select("Alice Martin")
    at = _fill_single_etf(at)
    assert not at.exception
    assert "Gold Miners" in _texts(at, "markdown")  # closest ETF shown
    at.button(key="single_validate").click()
    at.run()
    assert not at.exception
    journal = FeedbackStore(trained_settings).read_journal()
    assert list(journal["Name"]) == ["New Gold Miners"]
    assert journal.iloc[0]["ValidatedBy"] == "Alice Martin"
    assert "Validated" in _texts(at, "success")
    assert at.button(key="single_validate").disabled


def test_single_etf_cannot_be_validated_without_a_validator(trained_settings, use_settings):
    use_settings(trained_settings)
    at = _fill_single_etf(_app("pages/1_Single_ETF.py").run())
    assert at.button(key="single_validate").disabled
    assert "Choose a validator" in _texts(at, "warning") + _texts(at, "info")
    assert FeedbackStore(trained_settings).read_journal().empty


def test_single_etf_input_errors_are_shown(trained_settings, use_settings):
    use_settings(trained_settings)
    at = _app("pages/1_Single_ETF.py").run()
    at.session_state["single_holdings_init"] = pd.DataFrame({"Ticker": ["AAPL"], "Weight (%)": [-5.0]})
    at.run()
    at.text_input(key="single_name").input("Bad ETF")
    at.button(key="single_propose").click()
    at.run()
    assert "Negative weight" in _texts(at, "error")


def test_embeddings_model_asks_for_cost_confirmation_before_proposing(trained_emb_settings, use_settings,
                                                                      fake_client, monkeypatch):
    use_settings(trained_emb_settings)
    # Azure "configured" with the fake client: the app builds its client exactly as in production
    from etf_classifier.features.embeddings import ENV_VARIABLES

    for variable in ENV_VARIABLES:
        monkeypatch.setenv(variable, "test")
    monkeypatch.setattr("openai.AzureOpenAI", lambda **kwargs: fake_client)
    at = _app("pages/1_Single_ETF.py").run()
    calls_before = len(fake_client.calls)
    at = _fill_single_etf(at)
    assert at.button(key="single_confirm_cost")
    assert len(fake_client.calls) == calls_before
    at.button(key="single_confirm_cost").click()
    at.run()
    assert not at.exception
    assert len(fake_client.calls) == calls_before + 1
    assert at.button(key="single_validate")


def _batch_bytes(sample_csv) -> bytes:
    frame = pd.read_csv(sample_csv, sep=";", dtype=str, keep_default_na=False)
    frame = frame[frame["Name"].isin(["Gold Miners", "US Tech", "Negative Weight"])]
    frame = frame.assign(Name=frame["Name"] + " II")
    return frame.drop(columns=["Category"]).to_csv(sep=";", index=False).encode("utf-8")


def _upload_batch(at: AppTest, sample_csv) -> AppTest:
    at.file_uploader(key="batch_file").upload("new_etfs.csv", _batch_bytes(sample_csv), "text/csv")
    at.run()
    at.button(key="batch_propose").click()
    return at.run()


def test_page_batch_renders(trained_settings, use_settings):
    use_settings(trained_settings)
    at = _app("pages/2_Batch.py").run()
    assert not at.exception and not at.error


def test_batch_queue_validate_then_skip(trained_settings, use_settings, sample_csv):
    use_settings(trained_settings)
    at = _app("pages/2_Batch.py").run()
    at.sidebar.selectbox(key="validator").select("Bruno Leroy")
    at = _upload_batch(at, sample_csv)
    assert not at.exception
    assert "Negative Weight II" in _texts(at, "markdown") + _texts(at, "warning") + _texts(at, "error") or \
        any("Negative Weight II" in df.value.to_string() for df in at.dataframe)
    assert "1 / 2" in _texts(at, "markdown") + _texts(at, "caption")
    at.button(key="batch_validate").click()
    at.run()
    at.button(key="batch_next").click()
    at.run()
    assert "2 / 2" in _texts(at, "markdown") + _texts(at, "caption")
    at.button(key="batch_skip").click()
    at.run()
    assert not at.exception
    journal = FeedbackStore(trained_settings).read_journal()
    assert len(journal) == 1
    assert "1 validated" in _texts(at, "markdown") + _texts(at, "caption") + _texts(at, "success")


def _validate_directly(settings, name):
    from etf_classifier.app import services

    store = FeedbackStore(settings)
    return store.record_validation({"name": name, "description": "d", "composition": {"NEM": 5.0}},
                                   ["Gold"], "Alice Martin", predicted_categories=["Gold"], model_version="v1")


def test_history_page_empty_journal(trained_settings, use_settings):
    use_settings(trained_settings)
    at = _app("pages/3_History.py").run()
    assert not at.exception
    assert "No validation" in _texts(at, "info")


def test_history_page_lists_and_cancels(trained_settings, use_settings):
    use_settings(trained_settings)
    record = _validate_directly(trained_settings, "Gold Hist")
    at = _app("pages/3_History.py").run()
    assert any("Gold Hist" in df.value.to_string() for df in at.dataframe)
    at.sidebar.selectbox(key="validator").select("Alice Martin")
    at.run()
    at.selectbox(key="history_cancel_id").select(record["ValidationId"])
    at.text_input(key="history_cancel_comment").input("Wrong ETF")
    at.checkbox(key="history_cancel_confirm").check()
    at.run()  # the button is enabled once the confirmation is ticked
    at.button(key="history_cancel").click()
    at.run()
    assert not at.exception
    journal = FeedbackStore(trained_settings).read_journal()
    assert list(journal["Decision"]) == ["accepted", "cancellation"]
    assert "Cancelled" in _texts(at, "success")


def test_history_cancel_requires_a_comment(trained_settings, use_settings):
    use_settings(trained_settings)
    record = _validate_directly(trained_settings, "Gold Hist")
    at = _app("pages/3_History.py").run()
    at.sidebar.selectbox(key="validator").select("Alice Martin")
    at.run()
    at.selectbox(key="history_cancel_id").select(record["ValidationId"])
    at.checkbox(key="history_cancel_confirm").check()
    at.run()  # the button is enabled once the confirmation is ticked
    at.button(key="history_cancel").click()
    at.run()
    assert "comment" in _texts(at, "error")
    assert len(FeedbackStore(trained_settings).read_journal()) == 1


def test_monitoring_page_without_data(app_settings, use_settings):
    use_settings(app_settings)
    at = _app("pages/4_Monitoring.py").run()
    assert not at.exception
    assert "No validation" in _texts(at, "markdown") + _texts(at, "info")


def test_monitoring_page_shows_the_acceptance_rate(trained_settings, use_settings):
    use_settings(trained_settings)
    _validate_directly(trained_settings, "Gold Mon")
    at = _app("pages/4_Monitoring.py").run()
    assert not at.exception
    assert any(m.value == "100%" for m in at.metric)
