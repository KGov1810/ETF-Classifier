"""LLM justification in the interface: services and pages (fake chat client, no network)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from etf_classifier.app import services
from etf_classifier.feedback.store import FeedbackStore
from etf_classifier.llm.prompts import Justification
from etf_classifier.llm.store import JustificationStore

APP_DIR = Path(__file__).parents[2] / "src" / "etf_classifier" / "app"


def _justification(**changes):
    values = dict(summary="s", verdict="doubtful", reason="r", alternative="ESG", new_category=None,
                  inconsistency=None, confidence="high")
    values.update(changes)
    return Justification(**values)


def test_suggested_category_is_the_alternative():
    assert services.suggested_category("Gold", _justification()) == "ESG"


def test_without_alternative_the_selection_is_kept():
    assert services.suggested_category("Gold", _justification(alternative=None)) == "Gold"


def test_category_descriptions_include_the_historical_categories(trained_settings):
    descriptions = services.category_descriptions(trained_settings)
    assert {"Gold", "Global Equity"} <= set(descriptions)


def _proposal(settings):
    bundle = services.load_current_model(settings)
    checked = services.etf_from_inputs("NGLD", "New Gold", "Gold miners", pd.DataFrame(
        [["NEM", 11.0], ["GOLD", 9.0], ["AEM", 8.0]], columns=["Ticker", "Weight (%)"]), settings)
    return services.propose(checked.data, bundle, settings)[0]


def test_validation_is_linked_to_its_justification(trained_settings, chat_client):
    proposal = _proposal(trained_settings)
    justifier = services.llm_justifier(trained_settings, client=chat_client)
    result = services.llm_justify(justifier, proposal, trained_settings, confirmed=True)
    record = services.validate(proposal, proposal.category, "Alice Martin", trained_settings,
                               justification_id=result.justification_id)
    linked = JustificationStore(trained_settings).for_validation(record["ValidationId"])
    assert linked["id"] == result.justification_id


class _BrokenCompletions:
    def create(self, **_):
        raise ConnectionError("network unreachable")


def test_azure_errors_become_readable_messages(trained_settings):
    from types import SimpleNamespace

    proposal = _proposal(trained_settings)
    justifier = services.llm_justifier(trained_settings,
                                       client=SimpleNamespace(chat=SimpleNamespace(completions=_BrokenCompletions())))
    with pytest.raises(services.AppError, match="network unreachable"):
        services.llm_justify(justifier, proposal, trained_settings, confirmed=True)


# --- pages ------------------------------------------------------------------------------------------


@pytest.fixture
def single_page(trained_settings, monkeypatch, tmp_path):
    path = tmp_path / "isolated_settings.yaml"
    path.write_text(yaml.safe_dump(trained_settings, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("ETF_CLASSIFIER_CONFIG", str(path))
    at = AppTest.from_file(str(APP_DIR / "pages" / "1_Single_ETF.py"), default_timeout=60).run()
    at.sidebar.selectbox(key="validator").select("Alice Martin")
    at.session_state["single_holdings_init"] = pd.DataFrame(
        {"Ticker": ["NEM", "GOLD", "AEM"], "Weight (%)": [11.0, 9.0, 8.0]})
    at.run()
    at.text_input(key="single_ticker").input("NGMX")
    at.text_input(key="single_name").input("New Gold Miners")
    at.text_area(key="single_description").input("Global gold mining companies")
    at.button(key="single_propose").click()
    return at.run()


def _texts(at, kind):
    return " ".join(e.value for e in getattr(at, kind))


def test_explain_button_asks_for_confirmation_before_calling(azure_configured, single_page):
    at = single_page
    at.button(key="single_explain").click()
    at.run()
    assert azure_configured.calls == []
    assert "estimated cost" in _texts(at, "info")
    at.button(key="single_explain_confirm").click()
    at.run()
    assert not at.exception
    assert len(azure_configured.calls) == 1
    assert "Gold miners: ESG fits better" in _texts(at, "markdown")


def test_apply_suggestions_updates_the_selection(azure_configured, single_page):
    at = single_page
    at.button(key="single_explain").click()
    at.run()
    at.button(key="single_explain_confirm").click()
    at.run()
    at.button(key="single_apply").click()
    at.run()
    assert not at.exception
    assert at.selectbox(key="single_category").value == "ESG"
    assert at.checkbox(key="single_create").value is True
    assert at.text_input(key="single_new_name").value == "Precious Metals Miners"


def test_validating_after_an_explanation_links_it(azure_configured, single_page, trained_settings):
    at = single_page
    at.button(key="single_explain").click()
    at.run()
    at.button(key="single_explain_confirm").click()
    at.run()
    at.button(key="single_validate").click()
    at.run()
    validation_id = FeedbackStore(trained_settings).read_journal().iloc[0]["ValidationId"]
    assert JustificationStore(trained_settings).for_validation(validation_id) is not None


def test_explain_button_is_disabled_without_azure(single_page, monkeypatch):
    at = single_page
    assert at.button(key="single_explain").disabled
    assert "not configured" in _texts(at, "caption")


def test_history_shows_the_justification_of_a_validation(trained_settings, monkeypatch, tmp_path, llm_answer):
    store = FeedbackStore(trained_settings)
    record = store.record_validation({"ticker": "GLDJ", "name": "Gold J", "description": "d",
                                      "composition": {"NEM": 5.0}}, "Gold", "Alice Martin",
                                     predicted_category="Gold", model_version="v1")
    justifications = JustificationStore(trained_settings)
    justification_id = justifications.append(ticker="GLDJ", etf_name="Gold J", model_version="v1",
                                             deployment="gpt-6.1-sol",
                                             answer={**llm_answer, "notes": []}, input_tokens=1, output_tokens=1,
                                             cost_usd=0.01)
    justifications.link(justification_id, record["ValidationId"])
    path = tmp_path / "isolated_settings.yaml"
    path.write_text(yaml.safe_dump(trained_settings, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("ETF_CLASSIFIER_CONFIG", str(path))
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_History.py"), default_timeout=60).run()
    at.selectbox(key="history_justification_id").select(record["ValidationId"])
    at.run()
    assert not at.exception
    assert "Gold miners: ESG fits better" in _texts(at, "markdown")


def test_history_without_justification_says_so(trained_settings, monkeypatch, tmp_path):
    FeedbackStore(trained_settings).record_validation(
        {"ticker": "GLDK", "name": "Gold K", "description": "d", "composition": {"NEM": 5.0}}, "Gold",
        "Alice Martin")
    path = tmp_path / "isolated_settings.yaml"
    path.write_text(yaml.safe_dump(trained_settings, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("ETF_CLASSIFIER_CONFIG", str(path))
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_History.py"), default_timeout=60).run()
    assert "No LLM justification" in _texts(at, "caption")


def test_failed_justification_link_does_not_hide_a_saved_validation(trained_settings, chat_client, monkeypatch):
    proposal = _proposal(trained_settings)
    justifier = services.llm_justifier(trained_settings, client=chat_client)
    result = services.llm_justify(justifier, proposal, trained_settings, confirmed=True)

    def broken_link(*_):
        raise TimeoutError("justification file locked")

    monkeypatch.setattr(JustificationStore, "link", broken_link)
    record = services.validate(proposal, proposal.category, "Alice Martin", trained_settings,
                               justification_id=result.justification_id)
    assert len(FeedbackStore(trained_settings).read_journal()) == 1
    assert "could not be linked" in record["warning"]


def test_markdown_from_untrusted_text_is_neutralised():
    from etf_classifier.app.components import md_escape

    escaped = md_escape("See ![tracker](http://evil.example/a.png) and [link](http://evil.example) **bold**")
    assert "![" not in escaped and "](" not in escaped and "**" not in escaped
    assert md_escape("Gold & Silver Miners 2x") == "Gold & Silver Miners 2x"  # plain text unchanged


def test_llm_answer_cannot_inject_markdown(llm_answer):
    injected = {**llm_answer, "summary": "Fine. ![x](http://evil.example/leak.png)",
                "inconsistency": "[click](http://evil.example)", "notes": []}
    at = AppTest.from_string(
        "from etf_classifier.app import components\n"
        "from etf_classifier.llm.prompts import Justification\n"
        f"components.show_justification(Justification(**{injected!r}))\n").run()
    rendered = " ".join(e.value for e in at.markdown) + " ".join(e.value for e in at.warning)
    assert not at.exception
    assert "![" not in rendered and "](" not in rendered
