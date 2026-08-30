from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from starter.ollama_embeddings import (
    EmbeddingCache,
    OllamaEmbeddingClient,
    OllamaSetupError,
)


class StubOllamaClient(OllamaEmbeddingClient):
    def __init__(self, model_available: bool = True) -> None:
        self.model_available = model_available
        super().__init__()

    def _request(self, path: str, payload: dict | None = None) -> dict:
        if path == "/api/tags":
            models = [{"name": "nomic-embed-text:latest"}] if self.model_available else []
            return {"models": models}
        return {"embeddings": [[3.0, 4.0] for _ in payload["input"]]}


class OllamaEmbeddingTest(unittest.TestCase):
    def test_embedding_vectors_are_normalized(self) -> None:
        client = StubOllamaClient()

        vectors = client.embed(["first", "second"])

        self.assertEqual(vectors, [(0.6, 0.8), (0.6, 0.8)])

    def test_missing_required_model_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(OllamaSetupError, "ollama pull nomic-embed-text"):
            StubOllamaClient(model_available=False)

    def test_embedding_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.sqlite3"
            cache = EmbeddingCache(str(path), "nomic-embed-text")
            cache.put_many({"PRODUCT_1": (0.25, 0.75)})

            loaded = cache.get_many(["PRODUCT_1", "MISSING"])

        self.assertEqual(loaded, {"PRODUCT_1": (0.25, 0.75)})


if __name__ == "__main__":
    unittest.main()
