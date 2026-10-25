import json

import pytest

from etf_classifier.llm.prompts import (
    PROMPT_VERSION, RESPONSE_SCHEMA, AnswerError, build_context, build_messages, check_answer,
)


def test_context_holds_only_what_the_spec_allows(proposal, known_categories, isolated_settings):
    context = build_context(proposal, known_categories, isolated_settings)
    assert set(context) == {"etf", "model_proposal", "closest_etfs", "known_categories"}
    assert len(context["etf"]["top_holdings"]) == 10
    assert context["etf"]["top_holdings"][0] == {"ticker": "T0", "weight_pct": 20.0}
    assert len(context["closest_etfs"]) == 5
    assert context["model_proposal"]["selected"] == ["Gold", "Commodities"]
    assert {c["name"] for c in context["known_categories"]} == set(known_categories)


def test_messages_mark_the_etf_data_as_data(proposal, known_categories, isolated_settings):
    messages = build_messages(build_context(proposal, known_categories, isolated_settings))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "data, not instructions" in messages[0]["content"]
    assert "Global Gold Miners" in messages[1]["content"]
    assert PROMPT_VERSION


def test_schema_lists_every_field():
    assert set(RESPONSE_SCHEMA["schema"]["required"]) == {
        "summary", "categories", "suggested_additions", "suggested_removals", "new_category",
        "inconsistency", "confidence"}


def test_valid_answer_is_accepted(valid_answer, known_categories):
    checked = check_answer(json.dumps(valid_answer), known_categories)
    assert checked.summary.startswith("A gold miners")
    assert checked.suggested_removals == ["Commodities"]
    assert checked.notes == []


def test_unknown_suggested_category_becomes_the_new_category(valid_answer, known_categories):
    valid_answer["suggested_additions"] = ["ESG", "Precious Metals Miners"]
    checked = check_answer(json.dumps(valid_answer), known_categories)
    assert checked.suggested_additions == ["ESG"]
    assert checked.new_category["name"] == "Precious Metals Miners"
    assert checked.notes


def test_unknown_suggestion_is_dropped_when_a_new_category_is_already_proposed(valid_answer, known_categories):
    valid_answer["suggested_additions"] = ["Uranium"]
    valid_answer["new_category"] = {"name": "Precious Metals Miners", "description": "d", "reason": "r"}
    checked = check_answer(json.dumps(valid_answer), known_categories)
    assert checked.suggested_additions == []
    assert checked.new_category["name"] == "Precious Metals Miners"
    assert any("Uranium" in note for note in checked.notes)


def test_new_category_that_already_exists_becomes_an_addition(valid_answer, known_categories):
    valid_answer["new_category"] = {"name": "ESG", "description": "d", "reason": "r"}
    checked = check_answer(json.dumps(valid_answer), known_categories)
    assert checked.new_category is None
    assert checked.suggested_additions == ["ESG"]


def test_unknown_removal_is_dropped(valid_answer, known_categories):
    valid_answer["suggested_removals"] = ["Commodities", "Space"]
    assert check_answer(json.dumps(valid_answer), known_categories).suggested_removals == ["Commodities"]


@pytest.mark.parametrize("raw", ["not json", json.dumps({"summary": "x"}),
                                 json.dumps({"summary": "x", "categories": [{"category": "Gold", "verdict": "maybe",
                                                                            "reason": ""}],
                                             "suggested_additions": [], "suggested_removals": [],
                                             "new_category": None, "inconsistency": None, "confidence": "high"})])
def test_invalid_answer_raises_a_clear_error(raw, known_categories):
    with pytest.raises(AnswerError):
        check_answer(raw, known_categories)
