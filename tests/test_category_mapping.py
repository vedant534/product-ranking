from pathlib import Path

import pandas as pd

from data.category_mapping import build_item_category_mapping, save_item_category_mapping


def test_latest_category_is_selected_across_property_parts(tmp_path: Path) -> None:
    part1 = tmp_path / "item_properties_part1.csv"
    part2 = tmp_path / "item_properties_part2.csv"
    pd.DataFrame(
        {
            "timestamp": [1000, 2000, 3000],
            "itemid": [10, 20, 10],
            "property": ["categoryid", "categoryid", "available"],
            "value": ["1", "2", "1"],
        }
    ).to_csv(part1, index=False)
    pd.DataFrame(
        {
            "timestamp": [3000, 4000],
            "itemid": [10, 99],
            "property": ["categoryid", "categoryid"],
            "value": ["3", "9"],
        }
    ).to_csv(part2, index=False)

    mapping = build_item_category_mapping(
        [part1, part2],
        {"<UNK_ITEM>": 0, "10": 1, "20": 2},
        chunk_size=2,
    )

    assert mapping == {1: 3, 2: 2}
    output_path = save_item_category_mapping(mapping, tmp_path / "categories.json")
    assert output_path.read_text(encoding="utf-8").startswith("{\n")
