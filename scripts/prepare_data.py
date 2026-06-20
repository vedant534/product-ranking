"""Prepare RetailRocket data for session-based recommendation."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEFAULT_ALLOWED_EVENTS = ["view", "addtocart", "transaction"]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--step",
        choices=("sessionize", "make-dataset"),
        default="sessionize",
    )
    parser.add_argument(
        "--sample-n",
        type=_positive_int,
        default=None,
        help="Load only the first N events before sessionization",
    )
    return parser.parse_args(argv)


def _load_config(config_path: Path) -> dict[str, object]:
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as error:
        raise FileNotFoundError(f"Could not read config '{config_path}': {error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Config '{config_path}' must contain a YAML mapping")
    return config


def _resolve_sessionization_settings(
    config: dict[str, object],
) -> tuple[Path, float, int, list[str]]:
    """Support both the project config and RetailRocket dataset config shapes."""
    if "dataset" in config:
        dataset = config["dataset"]
        if not isinstance(dataset, dict):
            raise ValueError("The dataset config section must be a mapping")
        files = dataset.get("files")
        if not isinstance(files, dict) or "events" not in files:
            raise ValueError("The dataset config must define dataset.files.events")
        return Path(str(files["events"])), 30.0, 2, DEFAULT_ALLOWED_EVENTS.copy()

    paths = config.get("paths")
    data = config.get("data")
    if not isinstance(paths, dict) or not isinstance(data, dict):
        raise ValueError("The project config must define paths and data mappings")

    event_path = Path(str(paths["raw_data"])) / str(data["events_file"])
    gap_minutes = float(data.get("session_gap_minutes", 30))
    min_session_length = int(data.get("min_session_length", 2))
    allowed_events = [str(event) for event in data.get("event_types", DEFAULT_ALLOWED_EVENTS)]
    return event_path, gap_minutes, min_session_length, allowed_events


def _session_integrity(
    sessionized: pd.DataFrame,
    min_session_length: int,
) -> dict[str, bool]:
    if sessionized.empty:
        return {
            "minimum_length": False,
            "timestamps_sorted": True,
            "single_visitor": True,
        }

    session_lengths = sessionized.groupby("session_id", sort=False).size()
    timestamps_sorted = (
        sessionized.groupby("session_id", sort=False)["timestamp"]
        .apply(lambda timestamps: timestamps.is_monotonic_increasing)
        .all()
    )
    visitors_per_session = sessionized.groupby("session_id", sort=False)["visitorid"].nunique()
    return {
        "minimum_length": bool(session_lengths.min() >= min_session_length),
        "timestamps_sorted": bool(timestamps_sorted),
        "single_visitor": bool(visitors_per_session.le(1).all()),
    }


def _print_diagnostics(sessionized: pd.DataFrame, min_session_length: int) -> None:
    session_lengths = sessionized.groupby("session_id", sort=False).size()
    print("\nSession length distribution:")
    if session_lengths.empty:
        print("No sessions retained.")
    else:
        description = session_lengths.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
        print(description.to_string())

    checks = _session_integrity(sessionized, min_session_length)
    minimum = "N/A" if session_lengths.empty else str(int(session_lengths.min()))
    print("\nIntegrity checks:")
    print(
        f"  minimum session length >= {min_session_length}: "
        f"{'PASS' if checks['minimum_length'] else 'FAIL'} (minimum={minimum})"
    )
    print(
        "  timestamps sorted within sessions: "
        f"{'PASS' if checks['timestamps_sorted'] else 'FAIL'}"
    )
    print(
        "  sessions contain one visitor: "
        f"{'PASS' if checks['single_visitor'] else 'FAIL'}"
    )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Session integrity check(s) failed: {', '.join(failed)}")


def run_sessionization_step(
    config_path: Path,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    from data.load_data import load_events
    from data.sessionize import save_session_summary, sessionize_events

    config = _load_config(config_path)
    event_path, gap_minutes, min_session_length, allowed_events = (
        _resolve_sessionization_settings(config)
    )
    events = load_events(str(event_path), sample_n=sample_n)
    print(f"Loaded {len(events):,} events from '{event_path}'.")

    sessionized = sessionize_events(
        events,
        gap_minutes=gap_minutes,
        min_session_length=min_session_length,
        allowed_events=allowed_events,
    )
    summary = sessionized.attrs["summary_stats"]
    print(
        f"Retained {len(sessionized):,} events across "
        f"{summary['number_of_sessions']:,} sessions."
    )
    print(f"Event type distribution: {summary['event_type_distribution']}")
    _print_diagnostics(sessionized, min_session_length)
    paths = config.get("paths", {})
    interim_dir = (
        Path(str(paths.get("interim_data", "data/interim")))
        if isinstance(paths, dict)
        else Path("data/interim")
    )
    summary_path = save_session_summary(summary, interim_dir / "session_summary.json")
    print(f"Saved session summary: {summary_path}")
    return sessionized


def _resolve_dataset_settings(
    config: dict[str, object],
) -> tuple[float, float, float, int, Path]:
    if "dataset" in config:
        return 0.70, 0.15, 0.15, 50, Path("data/processed")

    split = config.get("split")
    data = config.get("data")
    paths = config.get("paths")
    if not isinstance(split, dict) or not isinstance(data, dict) or not isinstance(paths, dict):
        raise ValueError("The project config must define split, data, and paths mappings")
    return (
        float(split.get("train_fraction", 0.70)),
        float(split.get("validation_fraction", 0.15)),
        float(split.get("test_fraction", 0.15)),
        int(data.get("max_sequence_length", 50)),
        Path(str(paths.get("processed_data", "data/processed"))),
    )


def run_make_dataset_step(
    config_path: Path,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    from data.make_dataset import make_and_save_dataset
    from data.split import chronological_split, save_split_summary

    sessionized = run_sessionization_step(config_path, sample_n=sample_n)
    config = _load_config(config_path)
    train_fraction, validation_fraction, test_fraction, max_sequence_length, output_dir = (
        _resolve_dataset_settings(config)
    )
    split_events = chronological_split(
        sessionized,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    split_summary_path = save_split_summary(
        split_events.attrs["split_summary"],
        output_dir / "split_summary.json",
        purge_summary=split_events.attrs["purge_summary"],
    )
    examples, item_id_mapping, dataset_stats, paths = make_and_save_dataset(
        split_events,
        output_dir=output_dir,
        max_sequence_length=max_sequence_length,
    )
    dataset_stats["source"] = {
        "events_row_limit": sample_n,
        "sampled": sample_n is not None,
    }
    paths["dataset_stats"].write_text(
        json.dumps(dataset_stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nTrain-only vocabulary size: {len(item_id_mapping) - 1:,} items")
    for split_name in ("train", "validation", "test"):
        stats = dataset_stats["splits"][split_name]
        print(
            f"{split_name}: {stats['number_of_examples']:,} examples, "
            f"{stats['unknown_targets']:,} unknown targets, "
            f"{stats['unknown_prefix_items']:,} unknown prefix items"
        )
        preview = examples[examples["split"] == split_name].head(2)
        if not preview.empty:
            print(preview.loc[:, ["prefix_items", "target_item", "session_id", "prefix_length"]])

    print("\nSaved dataset artifacts:")
    for path in paths.values():
        print(f"  {path}")
    print(f"  {split_summary_path}")
    return examples


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.step == "sessionize":
        run_sessionization_step(args.config, sample_n=args.sample_n)
    elif args.step == "make-dataset":
        run_make_dataset_step(args.config, sample_n=args.sample_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
