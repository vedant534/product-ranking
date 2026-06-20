from pathlib import Path

import pandas as pd

from evaluation.segment_analysis import (
    add_segment_columns,
    analyze_segments,
    build_popularity_segments,
    prefix_length_segment,
    render_segment_report,
    save_segment_report,
)


class FixedRecommender:
    def __init__(self, recommendations: list[int]) -> None:
        self.recommendations = recommendations

    def recommend(self, prefix_items: list[int], k: int) -> list[int]:
        return self.recommendations[:k]


def _examples(include_event: bool = False) -> pd.DataFrame:
    data = {
        "prefix_items": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5, 6]],
        "target_item": [2, 3, 9, 0],
        "session_id": ["a", "a", "b", "c"],
        "split": ["test"] * 4,
        "prefix_length": [1, 2, 3, 6],
    }
    if include_event:
        data["target_event"] = ["view", "view", "addtocart", "transaction"]
    return pd.DataFrame(data)


def test_prefix_length_segments_cover_required_boundaries() -> None:
    assert [prefix_length_segment(length) for length in (1, 2, 3, 5, 6)] == [
        "1",
        "2",
        "3-5",
        "3-5",
        ">5",
    ]


def test_popularity_segments_are_ranked_from_training_interactions() -> None:
    segments = build_popularity_segments(
        [[1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    )

    assert segments[1] == "head"
    assert segments[2] == "head"
    assert segments[3] == "mid"
    assert segments[5] == "mid"
    assert segments[6] == "tail"

    segmented, event_column = add_segment_columns(_examples(), segments)
    assert event_column is None
    assert segmented.loc[3, "target_popularity_segment"] == "tail"


def test_analyze_segments_slices_predictions_and_supports_events() -> None:
    analysis = analyze_segments(
        {
            "Popularity": FixedRecommender([2, 8]),
            "Markov": FixedRecommender([3, 9]),
        },
        _examples(include_event=True),
        [[1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        total_items=10,
        k=2,
    )

    assert analysis["event_column"] == "target_event"
    assert "Event type" in analysis["dimensions"]
    prefix_one = analysis["dimensions"]["Prefix length"]["1"]
    assert prefix_one["Popularity"]["examples"] == 1
    assert prefix_one["Popularity"]["recall"] == 1.0
    prefix_two = analysis["dimensions"]["Prefix length"]["2"]
    assert prefix_two["Markov"]["recall"] == 1.0


def test_report_lists_all_models_and_explains_missing_event_data(tmp_path: Path) -> None:
    models = {
        name: FixedRecommender([2, 3, 9])
        for name in ("Popularity", "Markov", "Item-KNN", "GRU4Rec", "GRU4Rec + MMR")
    }
    analysis = analyze_segments(
        models,
        _examples(),
        [[1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        total_items=10,
        k=3,
    )

    report = render_segment_report(analysis, "test")
    assert "| 1 | Popularity |" in report
    assert "| 1 | GRU4Rec + MMR |" in report
    assert "Event-type analysis is unavailable" in report
    assert "## Observations" in report

    output_path = save_segment_report(analysis, "test", tmp_path / "segments.md")
    assert output_path.read_text(encoding="utf-8") == report
