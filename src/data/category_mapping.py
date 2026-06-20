"""Build a processed item-to-category mapping from RetailRocket property shards."""

import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def build_item_category_mapping(
    property_paths: Sequence[Path],
    item_id_mapping: dict[str, int],
    chunk_size: int = 1_000_000,
) -> dict[int, int]:
    """Return latest known category per train-vocabulary item.

    Property files are streamed in chunks. Category timestamps are compared
    globally across both shards, so file order does not override newer values.
    """
    if not property_paths:
        raise ValueError("At least one item-properties path is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    category_rows = []
    source_order = 0
    for path in property_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing item-properties file: '{path}'")
        chunks = pd.read_csv(
            path,
            usecols=["timestamp", "itemid", "property", "value"],
            dtype={"property": "string", "value": "string"},
            chunksize=chunk_size,
        )
        for chunk in chunks:
            categories = chunk.loc[chunk["property"] == "categoryid"].copy()
            if categories.empty:
                source_order += len(chunk)
                continue
            categories["model_item_id"] = categories["itemid"].astype(str).map(item_id_mapping)
            categories["categoryid"] = pd.to_numeric(categories["value"], errors="coerce")
            categories["timestamp"] = pd.to_numeric(categories["timestamp"], errors="coerce")
            categories = categories.dropna(
                subset=["model_item_id", "categoryid", "timestamp"]
            )
            categories["_source_order"] = range(
                source_order,
                source_order + len(categories),
            )
            category_rows.append(
                categories.loc[
                    :, ["model_item_id", "categoryid", "timestamp", "_source_order"]
                ]
            )
            source_order += len(chunk)

    if not category_rows:
        return {}
    combined = pd.concat(category_rows, ignore_index=True)
    latest = (
        combined.sort_values(["timestamp", "_source_order"], kind="mergesort")
        .drop_duplicates("model_item_id", keep="last")
        .sort_values("model_item_id")
    )
    return {
        int(model_item_id): int(categoryid)
        for model_item_id, categoryid in zip(
            latest["model_item_id"],
            latest["categoryid"],
        )
    }


def save_item_category_mapping(mapping: dict[int, int], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
