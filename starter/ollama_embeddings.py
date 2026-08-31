from __future__ import annotations

import json
import math
import sqlite3
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass


class OllamaSetupError(RuntimeError):
    """Raised when the required local Ollama model cannot be reached."""


@dataclass(frozen=True)
class OllamaConfig:
    model: str = "nomic-embed-text"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0


class OllamaEmbeddingClient:
    """Small standard-library client for Ollama's local embedding API."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        # Check the local service once so setup errors appear before evaluation.
        self.config = config or OllamaConfig()
        self._check_model()

    @property
    def model_name(self) -> str:
        # Expose the exact model tag used in cache and dense-index validation.
        return self.config.model

    def _request(self, path: str, payload: dict | None = None) -> dict:
        # Keep HTTP handling in one place so failures use the same clear message.
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise OllamaSetupError(
                "Threadline requires Ollama. Start it with 'ollama serve', then run "
                "'ollama pull nomic-embed-text'."
            ) from error

    def _check_model(self) -> None:
        # Ollama may be running without the required embedding model installed.
        response = self._request("/api/tags")
        available = {
            str(item.get("name", "")).split(":", maxsplit=1)[0]
            for item in response.get("models", [])
            if isinstance(item, dict)
        }
        if self.config.model.split(":", maxsplit=1)[0] not in available:
            raise OllamaSetupError(
                f"Ollama is running, but '{self.config.model}' is not installed. "
                f"Run 'ollama pull {self.config.model}'."
            )

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        # Send one batch request and normalize every returned vector.
        if not texts:
            return []
        response = self._request(
            "/api/embed",
            {"model": self.config.model, "input": texts, "truncate": True},
        )
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaSetupError("Ollama returned an invalid embedding response.")
        return [self._normalize(vector) for vector in embeddings]

    @staticmethod
    def _normalize(vector: object) -> tuple[float, ...]:
        # Unit vectors make dot products directly comparable as cosine scores.
        if not isinstance(vector, list) or not vector:
            raise OllamaSetupError("Ollama returned an empty embedding vector.")
        clean = tuple(float(value) for value in vector)
        length = math.sqrt(sum(value * value for value in clean))
        if length == 0:
            raise OllamaSetupError("Ollama returned a zero-length embedding vector.")
        return tuple(value / length for value in clean)


class EmbeddingCache:
    """Persistent product embeddings so repeat runs do not redo model work."""

    def __init__(self, path: str, model_name: str) -> None:
        # Namespacing by model prevents incompatible vectors from being mixed.
        self.model_name = model_name
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "model TEXT NOT NULL, parent_asin TEXT NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY (model, parent_asin))"
        )

    def get_many(self, identifiers: list[str]) -> dict[str, tuple[float, ...]]:
        # SQLite has a parameter limit, so candidate lists are fetched in chunks.
        if not identifiers:
            return {}
        placeholders = ",".join("?" for _ in identifiers)
        rows = self.connection.execute(
            f"SELECT parent_asin, vector FROM embeddings "
            f"WHERE model = ? AND parent_asin IN ({placeholders})",
            (self.model_name, *identifiers),
        ).fetchall()
        return {
            str(parent_asin): struct.unpack(f"<{len(blob) // 4}f", blob)
            for parent_asin, blob in rows
        }

    def put_many(self, vectors: dict[str, tuple[float, ...]]) -> None:
        # One transaction keeps warm-cache writes quick and recoverable.
        rows = [
            (
                self.model_name,
                parent_asin,
                struct.pack(f"<{len(vector)}f", *vector),
            )
            for parent_asin, vector in vectors.items()
        ]
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
            rows,
        )
        self.connection.commit()

    def close(self) -> None:
        # Explicit cleanup helps repeated evaluator runs in the same process.
        self.connection.close()
