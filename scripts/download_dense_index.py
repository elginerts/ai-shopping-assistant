from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request
from pathlib import Path

from starter.dense_index import DenseIndex, DenseIndexError


def main() -> None:
    # Download to a temporary file, then verify it before replacing the local index.
    parser = argparse.ArgumentParser(description="Download Threadline's release index")
    parser.add_argument("--url", default=os.getenv("THREADLINE_DENSE_INDEX_URL"))
    parser.add_argument("--output", default=".threadline_cache/dense_index.npz")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()
    if not args.url:
        parser.error("provide --url or set THREADLINE_DENSE_INDEX_URL")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".download")
    try:
        urllib.request.urlretrieve(args.url, temporary)
        index = DenseIndex.load(temporary, args.catalog, "nomic-embed-text")
        temporary.replace(output)
    except (DenseIndexError, OSError, urllib.error.URLError):
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified {len(index.product_ids):,} product embeddings.")


if __name__ == "__main__":
    main()
