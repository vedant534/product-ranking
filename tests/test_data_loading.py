"""Tests for RetailRocket CSV loading and validation."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from data.load_data import (
    load_category_tree,
    load_events,
    load_item_properties,
    load_item_properties_parts,
    main,
)


@pytest.fixture
def sample_data_dir(repository_root: Path) -> Path:
    return repository_root / "data" / "sample"


def test_load_events_converts_timestamps_and_sorts_by_visitor(
    sample_data_dir: Path,
) -> None:
    events = load_events(str(sample_data_dir / "events_sample.csv"))

    assert len(events) == 18
    assert str(events["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert events.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 00:00:00+00:00")
    assert events["visitorid"].tolist() == sorted(events["visitorid"].tolist())
    timestamps_are_sorted = events.groupby("visitorid")["timestamp"].apply(
        lambda values: values.is_monotonic_increasing
    )
    assert timestamps_are_sorted.all()
    assert events["transactionid"].dtype == pd.StringDtype()


def test_load_events_preserves_numeric_transaction_ids_as_text(tmp_path: Path) -> None:
    csv_path = tmp_path / "numeric_transaction.csv"
    csv_path.write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "1704067200000,1001,transaction,501,123\n"
        "1704067260000,1001,view,502,\n",
        encoding="utf-8",
    )

    events = load_events(str(csv_path))

    assert events.loc[0, "transactionid"] == "123"
    assert pd.isna(events.loc[1, "transactionid"])


def test_load_item_properties_uses_contract_dtypes(sample_data_dir: Path) -> None:
    properties = load_item_properties(str(sample_data_dir / "item_properties_sample.csv"))

    assert len(properties) == 14
    assert str(properties["timestamp"].dtype) == "datetime64[ns, UTC]"
    assert properties["property"].dtype == pd.StringDtype()
    assert properties["value"].dtype == pd.StringDtype()
    assert properties.iloc[-1]["value"] == "0"


def test_load_item_properties_parts_concatenates_in_path_order(
    sample_data_dir: Path,
    tmp_path: Path,
) -> None:
    source = pd.read_csv(sample_data_dir / "item_properties_sample.csv")
    part1_path = tmp_path / "item_properties_part1.csv"
    part2_path = tmp_path / "item_properties_part2.csv"
    source.iloc[:7].to_csv(part1_path, index=False)
    source.iloc[7:].to_csv(part2_path, index=False)

    properties = load_item_properties_parts([str(part1_path), str(part2_path)])

    assert len(properties) == len(source)
    assert properties["itemid"].tolist() == source["itemid"].tolist()
    assert str(properties["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_sample_n_limits_each_item_property_part(
    sample_data_dir: Path,
    tmp_path: Path,
) -> None:
    source = pd.read_csv(sample_data_dir / "item_properties_sample.csv")
    part1_path = tmp_path / "item_properties_part1.csv"
    part2_path = tmp_path / "item_properties_part2.csv"
    source.iloc[:7].to_csv(part1_path, index=False)
    source.iloc[7:].to_csv(part2_path, index=False)

    properties = load_item_properties_parts(
        [str(part1_path), str(part2_path)],
        sample_n=2,
    )

    expected_itemids = pd.concat([source.iloc[:2], source.iloc[7:9]])["itemid"].tolist()
    assert len(properties) == 4
    assert properties["itemid"].tolist() == expected_itemids


def test_load_category_tree_supports_nullable_roots(sample_data_dir: Path) -> None:
    categories = load_category_tree(str(sample_data_dir / "category_tree_sample.csv"))

    assert len(categories) == 10
    assert str(categories["categoryid"].dtype) == "int64"
    assert str(categories["parentid"].dtype) == "Int64"
    assert categories["parentid"].isna().sum() == 3


def test_cli_loads_user_supplied_events_path(
    sample_data_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sample_path = sample_data_dir / "events_sample.csv"

    main([str(sample_path)])

    output = capsys.readouterr().out
    assert "Loaded 18 events rows" in output
    assert "datetime64[ns, UTC]" in output
    assert str(sample_path) in output


@pytest.mark.parametrize(
    ("loader", "contents", "missing_columns"),
    [
        (
            load_events,
            "timestamp,visitorid\n1704067200000,1001\n",
            ("event", "itemid", "transactionid"),
        ),
        (load_item_properties, "timestamp,itemid\n1704067200000,501\n", ("property", "value")),
        (load_category_tree, "categoryid\n10\n", ("parentid",)),
    ],
)
def test_loaders_report_missing_required_columns(
    tmp_path: Path,
    loader: Callable[[str], pd.DataFrame],
    contents: str,
    missing_columns: tuple[str, ...],
) -> None:
    csv_path = tmp_path / "incomplete.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError) as error_info:
        loader(str(csv_path))

    message = str(error_info.value)
    assert str(csv_path) in message
    assert "missing required column(s)" in message
    assert all(column in message for column in missing_columns)


def test_events_loader_reports_malformed_millisecond_timestamp(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_timestamp.csv"
    csv_path.write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "not-a-timestamp,1,view,10,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected Unix epoch milliseconds"):
        load_events(str(csv_path))


def test_events_loader_reports_incompatible_identifier(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_visitor.csv"
    csv_path.write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "1704067200000,visitor-one,view,10,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible with its schema"):
        load_events(str(csv_path))
