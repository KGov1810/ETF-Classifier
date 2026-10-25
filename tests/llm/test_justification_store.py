from etf_classifier.llm.store import JustificationStore


def _answer():
    return {"summary": "s", "confidence": "high"}


def test_appended_justification_reads_back(isolated_settings):
    store = JustificationStore(isolated_settings)
    record_id = store.append(etf_name="Gold ETF", model_version="v1", deployment="gpt-6.1-sol", answer=_answer(),
                             input_tokens=10, output_tokens=5, cost_usd=0.001)
    record = store.get(record_id)
    assert record["etf_name"] == "Gold ETF" and record["answer"] == _answer() and record["cost_usd"] == 0.001
    assert record["prompt_version"]


def test_link_adds_a_line_and_finds_the_justification_of_a_validation(isolated_settings):
    store = JustificationStore(isolated_settings)
    record_id = store.append(etf_name="Gold ETF", model_version="v1", deployment="d", answer=_answer(),
                             input_tokens=1, output_tokens=1, cost_usd=0.0)
    lines_before = store.path.read_text(encoding="utf-8").count("\n")
    store.link(record_id, "20261025-101010-abcd")
    assert store.path.read_text(encoding="utf-8").count("\n") == lines_before + 1
    assert store.for_validation("20261025-101010-abcd")["answer"] == _answer()
    assert store.for_validation("unknown") is None


def test_missing_file_gives_empty_results(isolated_settings):
    store = JustificationStore(isolated_settings)
    assert store.get("x") is None and store.for_validation("y") is None
