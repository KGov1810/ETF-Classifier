"""Token counting with tiktoken, with an approximate fallback when its tables are unavailable.

tiktoken downloads its tables on first use. On a restricted network this download can fail: we
then fall back to an approximation (about 4 characters per token), flagged as such. To work
offline, pre-download the tables and set TIKTOKEN_CACHE_DIR.
"""

import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4          # common approximation for English
SAFE_CHARS_PER_TOKEN = 3     # conservative truncation when counting is approximate


class TokenCounter:
    """Counts and truncates texts with a tiktoken encoding (or an approximation)."""

    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # blocked network, unknown encoding, missing library
            logger.warning(
                "Tokenizer %s unavailable (%s): approximate counting at %d characters per token",
                encoding_name, type(exc).__name__, CHARS_PER_TOKEN,
            )

    @property
    def exact(self) -> bool:
        return self._encoding is not None

    def count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def truncate(self, text: str, max_tokens: int) -> tuple[str, bool]:
        """Truncates the text to max_tokens; returns (text, truncated or not)."""
        if self.count(text) <= max_tokens:
            return text, False
        if self._encoding is not None:
            tokens = self._encoding.encode(text, disallowed_special=())[:max_tokens]
            return self._encoding.decode(tokens), True
        return text[: max_tokens * SAFE_CHARS_PER_TOKEN], True


@lru_cache(maxsize=None)
def get_counter(encoding_name: str) -> TokenCounter:
    """Counter shared per encoding (the tables are loaded only once)."""
    return TokenCounter(encoding_name)
