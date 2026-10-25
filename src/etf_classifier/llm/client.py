"""Azure OpenAI chat client (GPT-6.1 Sol): creation from .env and call with a structured answer."""

import logging
import os

from dotenv import load_dotenv

from etf_classifier.config import PROJECT_ROOT
from etf_classifier.features.embeddings import ENV_VARIABLES
from etf_classifier.llm.prompts import RESPONSE_SCHEMA

logger = logging.getLogger(__name__)


def chat_deployment(settings: dict) -> str:
    """Name of the chat deployment: AZURE_OPENAI_CHAT_DEPLOYMENT if set, otherwise settings.yaml."""
    return os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT") or settings["llm"]["deployment"]


def create_chat_client(settings: dict):
    """AzureOpenAI client, or None if Azure OpenAI is not configured (.env)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing = [name for name in ENV_VARIABLES if not os.environ.get(name)]
    if missing:
        logger.info("Azure OpenAI not configured (missing variables: %s): see .env.example", ", ".join(missing))
        return None
    from openai import AzureOpenAI

    return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                       max_retries=settings["llm"]["max_retries"])


def complete(client, deployment: str, messages: list[dict], max_completion_tokens: int):
    """Chat call asking for the JSON schema; falls back to JSON mode if the deployment refuses structured outputs.

    Returns the raw response (content, finish reason and token usage are read by the caller).
    """
    try:
        return client.chat.completions.create(
            model=deployment, messages=messages, max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA})
    except Exception as exc:  # noqa: BLE001 - only a 400 about the response format triggers the fallback
        about_format = "response_format" in str(exc) or "json_schema" in str(exc)
        if getattr(exc, "status_code", None) != 400 or not about_format:
            raise
        logger.warning("Structured outputs refused by deployment '%s': falling back to JSON mode", deployment)
    schema_hint = {"role": "system", "content": "Answer with one JSON object following this schema:\n"
                                                f"{RESPONSE_SCHEMA['schema']}"}
    return client.chat.completions.create(
        model=deployment, messages=[messages[0], schema_hint, *messages[1:]],
        max_completion_tokens=max_completion_tokens, response_format={"type": "json_object"})
