"""Tests for chronological session-level data splitting."""

import json
from pathlib import Path

import pandas as pd
import pytest

from data.split import SPLIT_NAMES, chronological_split, save_split_summary


def _sessionized_events(number_of_sessions: int = 100) -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01 00:00:00+00:00")
    rows = []
    for session_id in range(number_of_sessions):
        session_start = base_time + pd.Timedelta(days=session_id)
        for position in range(2):
            rows.append(
                {
                    "session_id": session_id,
                    "visitorid": session_id,
                    "timestamp": session_start + pd.Timedelta(minutes=position * 5),
                    "event": "view",
                    "itemid": session_id * 2 + position,
                    "transactionid": None,
                    "session_position": position,
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture
def split_events() -> pd.DataFrame:
    return chronological_split(_sessionized_events())


def test_no_session_appears_in_more_than_one_split(split_events: pd.DataFrame) -> None:
    split_count_by_session = split_events.groupby("session_id")["split"].nunique()

    assert split_count_by_session.eq(1).all()
    assert len(split_events) == 200


def test_train_session_dates_are_earlier_than_validation(
    split_events: pd.DataFrame,
) -> None:
    session_starts = split_events.groupby(["session_id", "split"])["timestamp"].min()

    assert session_starts.xs("train", level="split").max() < session_starts.xs(
        "validation", level="split"
    ).min()


def test_validation_session_dates_are_earlier_than_test(
    split_events: pd.DataFrame,
) -> None:
    session_starts = split_events.groupby(["session_id", "split"])["timestamp"].min()

    assert session_starts.xs("validation", level="split").max() < session_starts.xs(
        "test", level="split"
    ).min()


def test_default_split_proportions_are_seventy_fifteen_fifteen(
    split_events: pd.DataFrame,
) -> None:
    session_splits = split_events[["session_id", "split"]].drop_duplicates()
    proportions = session_splits["split"].value_counts(normalize=True)

    assert proportions["train"] == pytest.approx(0.70)
    assert proportions["validation"] == pytest.approx(0.15)
    assert proportions["test"] == pytest.approx(0.15)


def test_split_summary_contains_requested_statistics(split_events: pd.DataFrame) -> None:
    summary = split_events.attrs["split_summary"]

    assert tuple(summary) == SPLIT_NAMES
    assert summary["train"]["number_of_sessions"] == 70
    assert summary["train"]["number_of_events"] == 140
    assert summary["train"]["number_of_unique_items"] == 140
    assert summary["train"]["number_of_unique_visitors"] == 70
    assert summary["validation"]["number_of_sessions"] == 15
    assert summary["test"]["number_of_sessions"] == 15
    assert summary["train"]["date_range"]["start"] < summary["train"]["date_range"]["end"]
    assert (
        summary["train"]["session_start_time_range"]["end"]
        < summary["validation"]["session_start_time_range"]["start"]
    )


def test_split_summary_is_persisted_with_iso_timestamps(
    split_events: pd.DataFrame,
    tmp_path: Path,
) -> None:
    output_path = save_split_summary(
        split_events.attrs["split_summary"],
        tmp_path / "split_summary.json",
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["splits"]["train"]["number_of_sessions"] == 70
    assert saved["splits"]["validation"]["number_of_sessions"] == 15
    assert saved["splits"]["train"]["date_range"]["start"].startswith("2024-01-01T")


def test_overlapping_boundary_sessions_are_purged() -> None:
    events = _sessionized_events(10)
    validation_boundary = pd.Timestamp("2024-01-08 00:00:00+00:00")
    events.loc[events["session_id"] == 6, "timestamp"] = [
        pd.Timestamp("2024-01-07 00:00:00+00:00"),
        validation_boundary + pd.Timedelta(minutes=5),
    ]

    split = chronological_split(events)

    assert 6 not in set(split["session_id"])
    assert split.attrs["purge_summary"]["purged_train_sessions"] == 1
    train_end = split.loc[split["split"] == "train", "timestamp"].max()
    validation_start = split.loc[split["split"] == "validation", "timestamp"].min()
    assert train_end < validation_start


def test_default_assignment_is_deterministic_not_random() -> None:
    events = _sessionized_events()
    first = chronological_split(events)
    second = chronological_split(events.sample(frac=1, random_state=7))

    first_mapping = first.groupby("session_id")["split"].first().sort_index()
    second_mapping = second.groupby("session_id")["split"].first().sort_index()
    pd.testing.assert_series_equal(first_mapping, second_mapping)


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        chronological_split(
            _sessionized_events(10),
            train_fraction=0.7,
            validation_fraction=0.2,
            test_fraction=0.2,
        )
