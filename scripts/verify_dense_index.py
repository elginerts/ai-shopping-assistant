from __future__ import annotations

import argparse

from starter.dense_index import DenseIndex


def main() -> None:
    # Load performs checksum, model, dimension, and catalogue-order validation.
    parser = argparse.ArgumentParser(description="Verify a Threadline dense index")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--index", default=".threadline_cache/dense_index.npz")
    parser.add_argument("--model", default="nomic-embed-text")
    args = parser.parse_args()
    index = DenseIndex.load(args.index, args.catalog, args.model)
    print(f"Verified {len(index.product_ids):,} products with {index.vectors.shape[1]} dimensions.")


if __name__ == "__main__":
    main()
