"""Cache SQLite des embeddings : un texte déjà calculé n'est jamais renvoyé (ni repayé) à Azure.

La clé combine le modèle, la dimension et le texte : changer l'un des trois ne réutilise jamais
un vecteur calculé dans d'autres conditions.
"""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHUNK = 500  # nombre de clés par requête (limite de variables SQLite)


def cache_key(model: str, dimensions: int, text: str) -> str:
    return hashlib.sha256(f"{model}|{dimensions}|{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")  # lectures et écritures concurrentes (app + job)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "key TEXT PRIMARY KEY, model TEXT NOT NULL, dimensions INTEGER NOT NULL, "
                "vector BLOB NOT NULL, created_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get_many(self, keys) -> dict[str, np.ndarray]:
        keys = list(dict.fromkeys(keys))
        found: dict[str, np.ndarray] = {}
        with self._connect() as connection:
            for start in range(0, len(keys), CHUNK):
                chunk = keys[start:start + CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = connection.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", chunk
                ).fetchall()
                found.update({key: np.frombuffer(blob, dtype=np.float32).copy() for key, blob in rows})
        return found

    def put_many(self, items: dict[str, np.ndarray], model: str, dimensions: int) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [(key, model, dimensions, np.asarray(vector, dtype=np.float32).tobytes(), now)
                for key, vector in items.items()]
        with self._connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)", rows)
