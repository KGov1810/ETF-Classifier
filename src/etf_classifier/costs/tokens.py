"""Comptage des tokens avec tiktoken, et repli approximatif si ses tables sont indisponibles.

tiktoken télécharge ses tables au premier usage. Sur un réseau restreint, ce téléchargement peut
échouer : on bascule alors sur une approximation (environ 4 caractères par token), signalée comme
telle. Pour travailler hors ligne, on peut pré-télécharger les tables et définir TIKTOKEN_CACHE_DIR.
"""

import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4          # approximation courante pour l'anglais
SAFE_CHARS_PER_TOKEN = 3     # troncature prudente quand le comptage est approximatif


class TokenCounter:
    """Compte et tronque des textes selon un encodage tiktoken (ou une approximation)."""

    def __init__(self, encoding_name: str):
        self.encoding_name = encoding_name
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # réseau bloqué, encodage inconnu, bibliothèque absente
            logger.warning(
                "Tokenizer %s indisponible (%s) : comptage approximatif à %d caractères par token",
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
        """Tronque le texte à max_tokens ; renvoie (texte, tronqué ou non)."""
        if self.count(text) <= max_tokens:
            return text, False
        if self._encoding is not None:
            tokens = self._encoding.encode(text, disallowed_special=())[:max_tokens]
            return self._encoding.decode(tokens), True
        return text[: max_tokens * SAFE_CHARS_PER_TOKEN], True


@lru_cache(maxsize=None)
def get_counter(encoding_name: str) -> TokenCounter:
    """Compteur partagé par encodage (les tables ne sont chargées qu'une fois)."""
    return TokenCounter(encoding_name)
