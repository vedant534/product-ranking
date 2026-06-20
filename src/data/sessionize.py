"""Convert anonymous RetailRocket events into deterministic sessions."""

import json
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

REQUIRED_COLUMNS = ("timestamp", "visitorid", "event", "itemid", "transactionid")
OUTPUT_COLUMNS = (
    "session_id",
    "visitorid",
    "timestamp",
    "event",
    "itemid",
    "transactionid",
    "session_position",
)
DEFAULT_ALLOWED_EVENTS = ("view", "addtocart", "transaction")


def _validate_parameters(gap_minutes: Real, min_session_length: int) -> None:
    if isinstance(gap_minutes, bool) or not isinstance(gap_minutes, Real) or gap_minutes < 0:
        raise ValueError("gap_minutes must be a non-negative number")
    if (
        isinstance(min_session_length, bool)
        or not isinstance(min_session_length, int)
        or min_session_length < 1
    ):
        raise ValueError("min_session_length must be a positive integer")


def _validate_columns(events: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(
            "Events dataframe is missing required column(s): "
            f"{', '.join(missing)}. Required columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    null_required = [
        column
        for column in ("timestamp", "visitorid", "itemid")
        if events[column].isna().any()
    ]
    if null_required:
        raise ValueError(
            "Events dataframe contains null values in required column(s): "
            f"{', '.join(null_required)}"
        )


def _normalize_timestamps(timestamps: pd.Series) -> pd.Series:
    """Return timezone-aware UTC timestamps, treating numeric values as milliseconds."""
    try:
        if is_datetime64_any_dtype(timestamps.dtype):
            return pd.to_datetime(timestamps, utc=True, errors="raise")
        if is_numeric_dtype(timestamps.dtype):
            return pd.to_datetime(timestamps, unit="ms", utc=True, errors="raise")
        return pd.to_datetime(timestamps, utc=True, errors="raise")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "Events dataframe has invalid timestamps; numeric timestamps must be Unix milliseconds"
        ) from error


def summarize_sessions(sessionized_events: pd.DataFrame) -> dict[str, object]:
    """Calculate aggregate session lengths and event counts."""
    if sessionized_events.empty:
        return {
            "number_of_sessions": 0,
            "average_session_length": 0.0,
            "median_session_length": 0.0,
            "max_session_length": 0,
            "event_type_distribution": {},
        }

    session_lengths = sessionized_events.groupby("session_id", sort=False).size()
    event_distribution = sessionized_events["event"].value_counts().sort_index()
    return {
        "number_of_sessions": int(session_lengths.size),
        "average_session_length": float(session_lengths.mean()),
        "median_session_length": float(session_lengths.median()),
        "max_session_length": int(session_lengths.max()),
        "event_type_distribution": {
            str(event): int(count) for event, count in event_distribution.items()
        },
    }


def save_session_summary(summary: dict[str, object], output_path: Path) -> Path:
    """Persist JSON-serializable aggregate session diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def sessionize_events(
    events: pd.DataFrame,
    gap_minutes: Real = 30,
    min_session_length: int = 2,
    allowed_events: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Sort, filter, and assign anonymous sessions to RetailRocket events.

    Summary statistics are stored in the returned dataframe's ``summary_stats``
    attribute. A gap of exactly ``gap_minutes`` remains in the same session;
    only a strictly greater gap creates a new session.
    """
    _validate_parameters(gap_minutes, min_session_length)
    _validate_columns(events)

    selected_events = DEFAULT_ALLOWED_EVENTS if allowed_events is None else tuple(allowed_events)
    working = events.loc[:, REQUIRED_COLUMNS].copy()
    working["timestamp"] = _normalize_timestamps(working["timestamp"])
    working = working[working["event"].isin(selected_events)].copy()

    working["_source_position"] = range(len(working))
    working = working.sort_values(
        ["visitorid", "timestamp", "_source_position"],
        kind="mergesort",
        ignore_index=True,
    )

    visitor_changed = working["visitorid"].ne(working["visitorid"].shift())
    time_gap = working.groupby("visitorid", sort=False)["timestamp"].diff()
    session_started = visitor_changed | time_gap.gt(pd.Timedelta(minutes=float(gap_minutes)))
    working["_session_key"] = session_started.cumsum()

    session_lengths = working.groupby("_session_key", sort=False)["itemid"].transform("size")
    working = working[session_lengths >= min_session_length].copy()

    if working.empty:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)
        result = result.astype({"session_id": "int64", "session_position": "int64"})
    else:
        working["session_id"] = pd.factorize(working["_session_key"], sort=False)[0]
        working["session_position"] = working.groupby("session_id", sort=False).cumcount()
        result = working.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)

    result.attrs["summary_stats"] = summarize_sessions(result)
    return result
