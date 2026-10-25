"""Faux client Azure OpenAI pour les tests : vecteurs déterministes, aucun appel réseau."""

import hashlib
from types import SimpleNamespace

import numpy as np


def fake_vector(text: str, dimensions: int) -> list[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(size=dimensions).tolist()


class FakeEmbeddingsAPI:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, model: str, input: list[str], dimensions: int, **_):
        self.calls.append({"model": model, "input": list(input), "dimensions": dimensions})
        data = [SimpleNamespace(index=i, embedding=fake_vector(text, dimensions)) for i, text in enumerate(input)]
        tokens = sum(max(1, len(text) // 4) for text in input)
        # ordre volontairement inversé : le client doit remettre les vecteurs dans l'ordre des index
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=tokens, total_tokens=tokens))


class FakeAzureClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()

    @property
    def calls(self) -> list[dict]:
        return self.embeddings.calls
