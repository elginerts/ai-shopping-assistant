from __future__ import annotations

import argparse
from pathlib import Path

from starter.dense_index import file_sha256


def main() -> None:
    # A portable Python check works on macOS and Linux without different hash tools.
    parser = argparse.ArgumentParser(description="Verify the frozen competition catalogue")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--checksum", default="data/catalog.sha256")
    args = parser.parse_args()

    checksum_path = Path(args.checksum)
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = file_sha256(args.catalog)
    if actual != expected:
        raise SystemExit(
            f"Catalogue checksum mismatch: expected {expected}, received {actual}"
        )
    print(f"Verified catalogue checksum: {actual}")


if __name__ == "__main__":
    main()
