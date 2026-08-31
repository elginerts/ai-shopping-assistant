from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


FORMAT_VERSION = 1


class DenseIndexError(RuntimeError):
    """Raised when the required dense index is missing or incompatible."""


def file_sha256(path: str | Path) -> str:
    # Stream large catalogues instead of loading the whole file into memory.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_product_text(product: dict) -> str:
    # Use the same labelled field order for building and verifying embeddings.
    # Labels help Nomic distinguish a category from a feature or description.
    def flatten(value: object) -> str:
        # Product fields may be nested, so reduce them to stable labelled text.
        if isinstance(value, dict):
            return " ".join(f"{key} {item}" for key, item in value.items())
        if isinstance(value, list):
            return " ".join(str(item) for item in value)
        return "" if value is None else str(value)

    labels = {
        "title": "title", "categories": "category", "features": "features",
        "details": "details", "description": "description",
    }
    parts = [
        f"{label}: {flatten(product.get(field))}"
        for field, label in labels.items()
        if product.get(field)
    ]
    return " ".join(parts)[:1400]


@dataclass(slots=True)
class DenseIndex:
    """Portable in-memory product vectors used as an independent entrance."""

    product_ids: np.ndarray
    vectors: np.ndarray
    model_name: str
    catalog_sha256: str
    positions: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Build constant-time ID lookup once after loading the matrix.
        self.positions = {
            str(parent_asin): position
            for position, parent_asin in enumerate(self.product_ids)
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        catalog_path: str | Path,
        model_name: str,
    ) -> "DenseIndex":
        # Validate provenance before accepting a large downloaded index.
        index_path = Path(path)
        if not index_path.exists():
            raise DenseIndexError(
                f"Dense index not found at '{index_path}'. Download the release asset "
                "or run 'python3 -m scripts.build_dense_index'."
            )
        try:
            with np.load(index_path, allow_pickle=False) as archive:
                version = int(archive["format_version"].item())
                stored_model = str(archive["model_name"].item())
                stored_checksum = str(archive["catalog_sha256"].item())
                product_ids = archive["product_ids"].copy()
                vectors = archive["vectors"].copy()
        except (KeyError, OSError, ValueError) as error:
            raise DenseIndexError(f"Dense index '{index_path}' is invalid.") from error

        if version != FORMAT_VERSION:
            raise DenseIndexError("Dense index format version does not match this code.")
        if stored_model != model_name:
            raise DenseIndexError(
                f"Dense index uses '{stored_model}', but the agent uses '{model_name}'."
            )
        if stored_checksum != file_sha256(catalog_path):
            raise DenseIndexError("Dense index was built from a different catalogue.")
        if vectors.dtype != np.float32 or vectors.ndim != 2:
            raise DenseIndexError("Dense vectors must be a two-dimensional float32 matrix.")
        with Path(catalog_path).open(encoding="utf-8") as handle:
            catalog_ids = {
                str(json.loads(line)["parent_asin"])
                for line in handle
                if line.strip()
            }
        index_ids = set(product_ids.tolist())
        if (
            len(product_ids) != len(vectors)
            or len(index_ids) != len(product_ids)
            or index_ids != catalog_ids
        ):
            raise DenseIndexError("Dense index product IDs are incomplete or duplicated.")
        return cls(product_ids, vectors, stored_model, stored_checksum)

    def search(self, query_vector: tuple[float, ...], limit: int = 50) -> list[tuple[str, float]]:
        # One vectorized matrix product scores all 50,000 catalogue items.
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],):
            raise DenseIndexError("Query embedding dimension does not match the dense index.")
        scores = self.vectors @ query
        count = min(max(1, limit), len(scores))
        positions = np.argpartition(scores, -count)[-count:]
        positions = positions[np.argsort(scores[positions])[::-1]]
        return [(str(self.product_ids[pos]), float(scores[pos])) for pos in positions]

    def scores_for(self, product_ids: list[str], query_vector: tuple[float, ...]) -> dict[str, float]:
        # Slice only requested rows when reranking an existing candidate slate.
        query = np.asarray(query_vector, dtype=np.float32)
        return {
            parent_asin: float(self.vectors[self.positions[parent_asin]] @ query)
            for parent_asin in product_ids
            if parent_asin in self.positions
        }


def save_dense_index(
    path: str | Path,
    product_ids: list[str],
    vectors: list[tuple[float, ...]],
    catalog_path: str | Path,
    model_name: str,
) -> None:
    # Store vectors with the metadata needed for portable startup checks.
    if not product_ids or len(product_ids) != len(vectors):
        raise DenseIndexError("Product IDs and vectors must be non-empty and aligned.")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise DenseIndexError("All dense vectors must have the same dimension.")
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            format_version=np.asarray(FORMAT_VERSION, dtype=np.int16),
            model_name=np.asarray(model_name),
            catalog_sha256=np.asarray(file_sha256(catalog_path)),
            product_ids=np.asarray(product_ids),
            vectors=matrix,
        )
    temporary.replace(target)
