from __future__ import annotations

import argparse
import json
from pathlib import Path

from starter.dense_index import save_dense_index, semantic_product_text
from starter.ollama_embeddings import EmbeddingCache, OllamaEmbeddingClient


def main() -> None:
    # Build resumably so a long local embedding job can continue after interruption.
    parser = argparse.ArgumentParser(description="Build Threadline's full Nomic index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default=".threadline_cache/dense_index.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    catalog_path = Path(args.catalog)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OllamaEmbeddingClient()
    cache = EmbeddingCache(
        str(output_path.parent / "product_embeddings.sqlite3"),
        client.model_name,
    )
    with catalog_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    identifiers = [str(row["parent_asin"]) for row in rows]
    vectors: dict[str, tuple[float, ...]] = {}

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        batch_ids = identifiers[start:start + args.batch_size]
        cached = cache.get_many(batch_ids)
        missing_rows = [row for row in batch_rows if str(row["parent_asin"]) not in cached]
        if missing_rows:
            embedded = client.embed([
                f"search_document: {semantic_product_text(row)}" for row in missing_rows
            ])
            fresh = {
                str(row["parent_asin"]): vector
                for row, vector in zip(missing_rows, embedded)
            }
            cache.put_many(fresh)
            cached.update(fresh)
        vectors.update(cached)
        print(f"Embedded {min(start + len(batch_rows), len(rows)):,}/{len(rows):,}", flush=True)

    save_dense_index(
        output_path,
        identifiers,
        [vectors[parent_asin] for parent_asin in identifiers],
        catalog_path,
        client.model_name,
    )
    cache.close()
    print(f"Dense index ready: {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
