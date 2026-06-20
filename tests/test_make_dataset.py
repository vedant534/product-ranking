"""Tests for leakage-safe next-item dataset creation."""

import json
from pathlib import Path

import pandas as pd

from data.make_dataset import (
    EXAMPLE_COLUMNS,
    UNK_ITEM_INDEX,
    UNK_ITEM_TOKEN,
    build_next_item_dataset,
    create_prefix_target_examples,
    save_dataset_artifacts,
)


def _split_events(sessions: list[tuple[int, str, list[int]]]) -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00:00+00:00")
    rows = []
    for day, (session_id, split_name, item_ids) in enumerate(sessions):
        for position, item_id in enumerate(item_ids):
            rows.append(
                {
                    "session_id": session_id,
                    "timestamp": base_time
                    + pd.Timedelta(days=day, minutes=position * 5),
                    "session_position": position,
                    "itemid": item_id,
                    "split": split_name,
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def test_toy_session_creates_immediate_next_item_examples() -> None:
    events = _split_events([(1, "train", [10, 20, 30])])

    examples = create_prefix_target_examples(events)

    assert tuple(examples.columns) == EXAMPLE_COLUMNS
    assert examples["prefix_items"].tolist() == [[10], [10, 20]]
    assert examples["target_item"].tolist() == [20, 30]
    assert examples["prefix_length"].tolist() == [1, 2]
    assert examples["session_id"].tolist() == [1, 1]
    assert examples["split"].tolist() == ["train", "train"]


def test_max_sequence_length_keeps_the_most_recent_items() -> None:
    events = _split_events([(1, "train", [10, 20, 30, 40])])

    examples = create_prefix_target_examples(events, max_sequence_length=2)

    assert examples["prefix_items"].tolist() == [[10], [10, 20], [20, 30]]
    assert examples["target_item"].tolist() == [20, 30, 40]
    assert examples["prefix_length"].tolist() == [1, 2, 2]


def test_target_event_is_retained_when_available() -> None:
    events = _split_events([(1, "train", [10, 20, 30])])
    events["event"] = ["view", "addtocart", "transaction"]

    examples = create_prefix_target_examples(events)

    assert examples["target_event"].tolist() == ["addtocart", "transaction"]


def test_unseen_validation_and_test_items_map_to_unknown() -> None:
    events = _split_events(
        [
            (1, "train", [10, 20]),
            (2, "validation", [10, 99]),
            (3, "test", [99, 20]),
        ]
    )

    examples, mapping, stats = build_next_item_dataset(events)
    validation = examples[examples["split"] == "validation"].iloc[0]
    test = examples[examples["split"] == "test"].iloc[0]

    assert mapping[UNK_ITEM_TOKEN] == UNK_ITEM_INDEX
    assert validation["prefix_items"] == [mapping["10"]]
    assert validation["target_item"] == UNK_ITEM_INDEX
    assert test["prefix_items"] == [UNK_ITEM_INDEX]
    assert test["target_item"] == mapping["20"]
    assert stats["splits"]["validation"]["unknown_targets"] == 1
    assert stats["splits"]["test"]["unknown_prefix_items"] == 1
    assert stats["splits"]["train"]["ranking_eligible_examples"] == 1


def test_item_mapping_is_fitted_from_training_split_only() -> None:
    events = _split_events(
        [
            (1, "train", [10, 20]),
            (2, "validation", [30, 40]),
            (3, "test", [50, 60]),
        ]
    )

    _, mapping, _ = build_next_item_dataset(events)

    assert mapping == {UNK_ITEM_TOKEN: 0, "10": 1, "20": 2}
    assert all(str(item) not in mapping for item in (30, 40, 50, 60))


def test_required_artifacts_are_saved_and_readable(tmp_path: Path) -> None:
    events = _split_events(
        [
            (1, "train", [10, 20]),
            (2, "validation", [10, 99]),
            (3, "test", [99, 20]),
        ]
    )
    examples, mapping, stats = build_next_item_dataset(events)

    paths = save_dataset_artifacts(examples, mapping, stats, tmp_path)

    assert {path.name for path in paths.values()} == {
        "train_examples.parquet",
        "validation_examples.parquet",
        "test_examples.parquet",
        "item_id_mapping.json",
        "dataset_stats.json",
    }
    train_examples = pd.read_parquet(paths["train"])
    assert train_examples.iloc[0]["prefix_items"].tolist() == [mapping["10"]]
    assert json.loads(paths["item_id_mapping"].read_text(encoding="utf-8")) == mapping
    saved_stats = json.loads(paths["dataset_stats"].read_text(encoding="utf-8"))
    assert saved_stats["training_vocabulary_size"] == 2
