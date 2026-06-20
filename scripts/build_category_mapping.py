"""Create a processed item-category artifact for the demo without runtime raw-data access."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/retailrocket"),
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("data/processed/item_id_mapping.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/item_category_mapping.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from data.category_mapping import (
        build_item_category_mapping,
        save_item_category_mapping,
    )

    args = parse_args(argv)
    item_mapping = json.loads(args.mapping_path.read_text(encoding="utf-8"))
    property_paths = [
        args.raw_dir / "item_properties_part1.csv",
        args.raw_dir / "item_properties_part2.csv",
    ]
    category_mapping = build_item_category_mapping(
        property_paths,
        item_mapping,
        chunk_size=args.chunk_size,
    )
    output_path = save_item_category_mapping(category_mapping, args.output)
    train_vocabulary_size = len(item_mapping) - 1
    coverage = len(category_mapping) / train_vocabulary_size if train_vocabulary_size else 0.0
    print(
        f"Saved categories for {len(category_mapping):,}/{train_vocabulary_size:,} "
        f"training items ({coverage:.2%}) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
