from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from starter.dense_index import DenseIndex, DenseIndexError, save_dense_index
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
    def test_dense_index_round_trip_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text('{"parent_asin":"A"}\n{"parent_asin":"B"}\n')
            index_path = root / "dense.npz"
            save_dense_index(
                index_path,
                ["A", "B"],
                [(1.0, 0.0), (0.0, 1.0)],
                catalog,
                "nomic-embed-text",
            )
            index = DenseIndex.load(index_path, catalog, "nomic-embed-text")

            self.assertEqual(index.search((0.9, 0.1), 1)[0][0], "A")

    def test_dense_index_rejects_a_different_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text('{"parent_asin":"A"}\n')
            index_path = root / "dense.npz"
            save_dense_index(index_path, ["A"], [(1.0, 0.0)], catalog, "nomic-embed-text")
            catalog.write_text('{"parent_asin":"CHANGED"}\n')

            with self.assertRaisesRegex(DenseIndexError, "different catalogue"):
                DenseIndex.load(index_path, catalog, "nomic-embed-text")

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
            cache.close()

        self.assertEqual(loaded, {"PRODUCT_1": (0.25, 0.75)})


if __name__ == "__main__":
    unittest.main()
