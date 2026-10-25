import csv

import pytest

from etf_classifier.llm.justify import ConfirmationRequired, Justifier, LLMUnavailable
from etf_classifier.llm.prompts import AnswerError
from etf_classifier.llm.store import JustificationStore


def _cost_lines(settings):
    with open(settings["costs"]["usage_log_path"], encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def test_estimate_is_given_before_any_call(isolated_settings, proposal, known_categories, valid_answer, fake_chat):
    client = fake_chat.FakeChatClient(valid_answer)
    justifier = Justifier(isolated_settings, client=client)
    estimate = justifier.estimate(proposal, known_categories)
    assert estimate.cost.input_tokens > 0 and estimate.cost.cost_usd > 0 and estimate.cost.approximate
    assert not estimate.cached
    with pytest.raises(ConfirmationRequired):
        justifier.justify(proposal, known_categories)
    assert client.calls == []


def test_confirmed_call_records_the_cost_and_the_justification(isolated_settings, proposal, known_categories,
                                                                valid_answer, fake_chat):
    client = fake_chat.FakeChatClient(valid_answer)
    result = Justifier(isolated_settings, client=client).justify(proposal, known_categories, confirmed=True)
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert result.justification.suggested_removals == ["Commodities"]
    lines = _cost_lines(isolated_settings)
    assert len(lines) == 1 and lines[0]["module"] == "llm-justify" and lines[0]["input_tokens"] == "1200"
    stored = JustificationStore(isolated_settings).get(result.justification_id)
    assert stored["etf_name"] == "Global Gold Miners" and stored["answer"]["confidence"] == "high"


def test_identical_request_is_served_by_the_cache(isolated_settings, proposal, known_categories, valid_answer,
                                                  fake_chat):
    client = fake_chat.FakeChatClient(valid_answer)
    first = Justifier(isolated_settings, client=client).justify(proposal, known_categories, confirmed=True)
    justifier = Justifier(isolated_settings, client=client)
    assert justifier.estimate(proposal, known_categories).cached
    second = justifier.justify(proposal, known_categories)  # no confirmation needed: nothing to pay
    assert len(client.calls) == 1 and second.cached and second.cost_usd == 0
    assert second.justification_id == first.justification_id
    assert len(_cost_lines(isolated_settings)) == 1


def test_fallback_to_json_mode_when_structured_outputs_are_refused(isolated_settings, proposal, known_categories,
                                                                    valid_answer, fake_chat):
    client = fake_chat.FakeChatClient(valid_answer, reject_schema=True)
    result = Justifier(isolated_settings, client=client).justify(proposal, known_categories, confirmed=True)
    assert [c["response_format"]["type"] for c in client.calls] == ["json_schema", "json_object"]
    assert result.justification.confidence == "high"


def test_invalid_answer_is_an_error_but_its_cost_is_recorded(isolated_settings, proposal, known_categories,
                                                             fake_chat):
    client = fake_chat.FakeChatClient("this is not json")
    with pytest.raises(AnswerError):
        Justifier(isolated_settings, client=client).justify(proposal, known_categories, confirmed=True)
    assert len(_cost_lines(isolated_settings)) == 1


def test_truncated_answer_gives_a_clear_error(isolated_settings, proposal, known_categories, valid_answer, fake_chat):
    client = fake_chat.FakeChatClient(valid_answer, finish_reason="length")
    with pytest.raises(AnswerError, match="max_completion_tokens"):
        Justifier(isolated_settings, client=client).justify(proposal, known_categories, confirmed=True)


def test_without_azure_configuration_nothing_is_called(isolated_settings, proposal, known_categories, monkeypatch):
    for variable in ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"]:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("etf_classifier.azure.load_dotenv", lambda *a, **k: None)
    justifier = Justifier(isolated_settings)
    assert not justifier.available
    with pytest.raises(LLMUnavailable, match="Azure OpenAI is not configured"):
        justifier.justify(proposal, known_categories, confirmed=True)


def test_deployment_missing_from_the_price_list_is_refused(isolated_settings, proposal, known_categories, valid_answer,
                                                            fake_chat, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "my-unknown-chat")
    justifier = Justifier(isolated_settings, client=fake_chat.FakeChatClient(valid_answer))
    assert not justifier.available
    assert "pricing.yaml" in justifier.unavailable_reason
