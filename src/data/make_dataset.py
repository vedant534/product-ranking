"""Create leakage-safe next-item examples from chronologically split sessions."""

import json
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

REQUIRED_COLUMNS = ("session_id", "timestamp", "session_position", "itemid", "split")
EXAMPLE_COLUMNS = ("prefix_items", "target_item", "session_id", "split", "prefix_length")
SPLIT_NAMES = ("train", "validation", "test")
UNK_ITEM_TOKEN = "<UNK_ITEM>"
UNK_ITEM_INDEX = 0


def _validate_input(events: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(
            "Split session dataframe is missing required column(s): "
            f"{', '.join(missing)}. Required columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    null_required = [column for column in REQUIRED_COLUMNS if events[column].isna().any()]
    if null_required:
        raise ValueError(
            "Split session dataframe contains null values in required column(s): "
            f"{', '.join(null_required)}"
        )

    unexpected_splits = sorted(set(events["split"].astype(str).unique()) - set(SPLIT_NAMES))
    if unexpected_splits:
        raise ValueError(f"Unexpected split value(s): {unexpected_splits}")

    split_counts = events.groupby("session_id", sort=False)["split"].nunique()
    leaking_sessions = split_counts[split_counts > 1].index.tolist()
    if leaking_sessions:
        preview = leaking_sessions[:5]
        raise ValueError(f"Session IDs appear in multiple splits: {preview}")


def _validate_max_sequence_length(max_sequence_length: int) -> None:
    if (
        isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
        or max_sequence_length < 1
    ):
        raise ValueError("max_sequence_length must be a positive integer")


def _normalize_timestamps(timestamps: pd.Series) -> pd.Series:
    try:
        if is_datetime64_any_dtype(timestamps.dtype):
            return pd.to_datetime(timestamps, utc=True, errors="raise")
        if is_numeric_dtype(timestamps.dtype):
            return pd.to_datetime(timestamps, unit="ms", utc=True, errors="raise")
        return pd.to_datetime(timestamps, utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Split session dataframe has invalid timestamps; numeric timestamps "
            "must be Unix milliseconds"
        ) from error


def _as_item_id(item: object) -> int:
    try:
        converted = int(item)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Item IDs must be integer-compatible; found {item!r}") from error
    if converted != item:
        raise ValueError(f"Item IDs must be integers; found {item!r}")
    return converted


def create_prefix_target_examples(
    split_events: pd.DataFrame,
    max_sequence_length: int = 50,
) -> pd.DataFrame:
    """Create raw immediate-next-item examples without applying vocabulary indices."""
    _validate_input(split_events)
    _validate_max_sequence_length(max_sequence_length)

    has_event = "event" in split_events.columns
    working_columns = list(REQUIRED_COLUMNS) + (["event"] if has_event else [])
    output_columns = list(EXAMPLE_COLUMNS) + (["target_event"] if has_event else [])
    working = split_events.loc[:, working_columns].copy()
    working["timestamp"] = _normalize_timestamps(working["timestamp"])
    working["_source_order"] = range(len(working))
    working = working.sort_values(
        ["session_id", "timestamp", "session_position", "_source_order"],
        kind="mergesort",
        ignore_index=True,
    )

    examples = []
    for session_id, session in working.groupby("session_id", sort=False):
        item_ids = [_as_item_id(item) for item in session["itemid"].tolist()]
        split_name = str(session["split"].iloc[0])
        for target_position in range(1, len(item_ids)):
            prefix_start = max(0, target_position - max_sequence_length)
            prefix = item_ids[prefix_start:target_position]
            example = {
                "prefix_items": prefix,
                "target_item": item_ids[target_position],
                "session_id": session_id,
                "split": split_name,
                "prefix_length": len(prefix),
            }
            if has_event:
                example["target_event"] = str(session.iloc[target_position]["event"])
            examples.append(example)

    if not examples:
        return pd.DataFrame(columns=output_columns).astype(
            {"target_item": "int64", "split": "string", "prefix_length": "int64"}
        )
    return pd.DataFrame(examples, columns=output_columns).astype(
        {"target_item": "int64", "split": "string", "prefix_length": "int64"}
    )


def build_item_id_mapping(split_events: pd.DataFrame) -> dict[str, int]:
    """Fit a deterministic item-to-index mapping from training events only."""
    _validate_input(split_events)
    training_items = {
        _as_item_id(item)
        for item in split_events.loc[split_events["split"] == "train", "itemid"].tolist()
    }
    mapping = {UNK_ITEM_TOKEN: UNK_ITEM_INDEX}
    mapping.update(
        {str(item_id): index for index, item_id in enumerate(sorted(training_items), start=1)}
    )
    return mapping


def map_examples_to_indices(
    examples: pd.DataFrame,
    item_id_mapping: dict[str, int],
) -> pd.DataFrame:
    """Map raw example items to train-fitted indices, using UNK for unseen items."""
    if item_id_mapping.get(UNK_ITEM_TOKEN) != UNK_ITEM_INDEX:
        raise ValueError(f"item_id_mapping must map {UNK_ITEM_TOKEN} to {UNK_ITEM_INDEX}")

    mapped = examples.copy()
    mapped["prefix_items"] = mapped["prefix_items"].apply(
        lambda prefix: [
            item_id_mapping.get(str(_as_item_id(item)), UNK_ITEM_INDEX) for item in prefix
        ]
    )
    mapped["target_item"] = mapped["target_item"].apply(
        lambda item: item_id_mapping.get(str(_as_item_id(item)), UNK_ITEM_INDEX)
    )
    return mapped


def build_dataset_stats(
    raw_examples: pd.DataFrame,
    mapped_examples: pd.DataFrame,
    item_id_mapping: dict[str, int],
    max_sequence_length: int,
) -> dict[str, object]:
    """Build JSON-serializable statistics for generated examples."""
    split_stats = {}
    for split_name in SPLIT_NAMES:
        raw_split = raw_examples[raw_examples["split"] == split_name]
        mapped_split = mapped_examples[mapped_examples["split"] == split_name]
        prefix_lengths = mapped_split["prefix_length"]
        unknown_prefix_items = sum(
            item == UNK_ITEM_INDEX
            for prefix in mapped_split["prefix_items"]
            for item in prefix
        )
        seen_known_targets = sum(
            int(target) != UNK_ITEM_INDEX and int(target) in set(prefix)
            for prefix, target in zip(
                mapped_split["prefix_items"],
                mapped_split["target_item"],
            )
        )
        split_stats[split_name] = {
            "number_of_examples": int(len(mapped_split)),
            "number_of_sessions": int(raw_split["session_id"].nunique()),
            "average_prefix_length": (
                float(prefix_lengths.mean()) if not prefix_lengths.empty else 0.0
            ),
            "max_prefix_length": (
                int(prefix_lengths.max()) if not prefix_lengths.empty else 0
            ),
            "unknown_prefix_items": int(unknown_prefix_items),
            "unknown_targets": int((mapped_split["target_item"] == UNK_ITEM_INDEX).sum()),
            "seen_known_target_examples": int(seen_known_targets),
            "ranking_eligible_examples": int(len(mapped_split) - seen_known_targets),
        }

    return {
        "max_sequence_length": max_sequence_length,
        "unknown_item_token": UNK_ITEM_TOKEN,
        "unknown_item_index": UNK_ITEM_INDEX,
        "training_vocabulary_size": len(item_id_mapping) - 1,
        "vocabulary_size_including_unknown": len(item_id_mapping),
        "splits": split_stats,
    }


def build_next_item_dataset(
    split_events: pd.DataFrame,
    max_sequence_length: int = 50,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, object]]:
    """Create raw examples, fit the train vocabulary, and return mapped examples."""
    raw_examples = create_prefix_target_examples(split_events, max_sequence_length)
    item_id_mapping = build_item_id_mapping(split_events)
    mapped_examples = map_examples_to_indices(raw_examples, item_id_mapping)
    stats = build_dataset_stats(
        raw_examples,
        mapped_examples,
        item_id_mapping,
        max_sequence_length,
    )
    return mapped_examples, item_id_mapping, stats


def save_dataset_artifacts(
    examples: pd.DataFrame,
    item_id_mapping: dict[str, int],
    dataset_stats: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    """Write per-split Parquet examples and JSON metadata artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_examples.parquet",
        "validation": output_dir / "validation_examples.parquet",
        "test": output_dir / "test_examples.parquet",
        "item_id_mapping": output_dir / "item_id_mapping.json",
        "dataset_stats": output_dir / "dataset_stats.json",
    }

    for split_name in SPLIT_NAMES:
        split_examples = examples[examples["split"] == split_name].reset_index(drop=True)
        try:
            split_examples.to_parquet(paths[split_name], index=False, engine="pyarrow")
        except ImportError as error:
            raise RuntimeError(
                "Writing Parquet artifacts requires the pyarrow dependency"
            ) from error

    with paths["item_id_mapping"].open("w", encoding="utf-8") as mapping_file:
        json.dump(item_id_mapping, mapping_file, indent=2)
        mapping_file.write("\n")
    with paths["dataset_stats"].open("w", encoding="utf-8") as stats_file:
        json.dump(dataset_stats, stats_file, indent=2, sort_keys=True)
        stats_file.write("\n")
    return paths


def make_and_save_dataset(
    split_events: pd.DataFrame,
    output_dir: Path,
    max_sequence_length: int = 50,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, object], dict[str, Path]]:
    """Build the mapped next-item dataset and save all required artifacts."""
    examples, item_id_mapping, dataset_stats = build_next_item_dataset(
        split_events,
        max_sequence_length=max_sequence_length,
    )
    paths = save_dataset_artifacts(examples, item_id_mapping, dataset_stats, output_dir)
    return examples, item_id_mapping, dataset_stats, paths
