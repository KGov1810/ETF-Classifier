"""Fake Azure OpenAI chat client for the tests: prepared answer, token usage, no network call."""

import json
from types import SimpleNamespace


class SchemaNotSupported(Exception):
    """Mimics the 400 error of a deployment that does not support structured outputs."""

    status_code = 400


class FakeCompletions:
    def __init__(self, answer, reject_schema=False, finish_reason="stop"):
        self.answer = answer
        self.reject_schema = reject_schema
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    def create(self, model, messages, response_format, max_completion_tokens, **_):
        self.calls.append({"model": model, "messages": messages, "response_format": response_format})
        if self.reject_schema and response_format.get("type") == "json_schema":
            raise SchemaNotSupported("response_format json_schema is not supported by this deployment")
        content = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=300,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=0))
        choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeChatClient:
    def __init__(self, answer, **options):
        self.chat = SimpleNamespace(completions=FakeCompletions(answer, **options))

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls
