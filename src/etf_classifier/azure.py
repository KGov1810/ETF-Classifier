"""Azure OpenAI access shared by the embeddings and the LLM: environment variables (.env) and client.

The key is only read from the environment and only passed to the official client; it is never logged.
"""

import logging
import os

from dotenv import load_dotenv

from etf_classifier.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

ENV_VARIABLES = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION")


def azure_client(max_retries: int):
    """AzureOpenAI client from .env / the environment, or None if Azure OpenAI is not configured."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    missing = [name for name in ENV_VARIABLES if not os.environ.get(name)]
    if missing:
        logger.info("Azure OpenAI not configured (missing variables: %s): see .env.example", ", ".join(missing))
        return None
    from openai import AzureOpenAI

    return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                       api_version=os.environ["AZURE_OPENAI_API_VERSION"], max_retries=max_retries)


def deployment_name(variable: str, default: str) -> str:
    """Deployment name from the environment (e.g. AZURE_OPENAI_CHAT_DEPLOYMENT), otherwise the configured one."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return os.environ.get(variable) or default
