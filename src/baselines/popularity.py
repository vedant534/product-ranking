"""Train-only most-popular recommendation baseline."""

from collections import Counter
from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Union

import pandas as pd

TrainData = Union[pd.DataFrame, Iterable[Iterable[object]]]


def _item_sort_key(item: object) -> tuple[str, str]:
    return type(item).__name__, repr(item)


def _validate_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    return int(k)


def prepare_train_sessions(train_data: TrainData) -> list[list[object]]:
    """Normalize train sessions and enforce train filtering for DataFrame input."""
    if isinstance(train_data, pd.DataFrame):
        required = {"session_id", "itemid"}
        missing = sorted(required - set(train_data.columns))
        if missing:
            raise ValueError(f"Training interactions are missing column(s): {', '.join(missing)}")

        interactions = train_data
        if "split" in interactions.columns:
            interactions = interactions[interactions["split"] == "train"]
        sort_columns = ["session_id"]
        sort_columns.extend(
            column
            for column in ("timestamp", "session_position")
            if column in interactions.columns
        )
        interactions = interactions.sort_values(sort_columns, kind="mergesort")
        return [
            group["itemid"].tolist()
            for _, group in interactions.groupby("session_id", sort=False)
        ]

    return [list(session) for session in train_data]


class PopularityRecommender:
    """Rank items by train interaction count with stable tie-breaking."""

    def __init__(self) -> None:
        self.item_counts: Counter[object] = Counter()
        self.ranked_items: list[object] = []
        self._is_fitted = False

    def fit(self, train_data: TrainData) -> "PopularityRecommender":
        sessions = prepare_train_sessions(train_data)
        self.item_counts = Counter(item for session in sessions for item in session)
        self.ranked_items = sorted(
            self.item_counts,
            key=lambda item: (-self.item_counts[item], _item_sort_key(item)),
        )
        self._is_fitted = True
        return self

    def recommend(self, prefix_items: Sequence[object], k: int) -> list[object]:
        if not self._is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before recommendation")
        cutoff = _validate_k(k)
        seen = set(prefix_items)
        recommendations = []
        for item in self.ranked_items:
            if item not in seen:
                recommendations.append(item)
            if len(recommendations) == cutoff:
                break
        return recommendations

    def score_items(self, item_ids: Sequence[object]) -> dict[object, float]:
        """Return raw train interaction counts for requested catalog items."""
        if not self._is_fitted:
            raise RuntimeError("PopularityRecommender must be fitted before scoring")
        return {
            item: float(self.item_counts[item])
            for item in item_ids
            if item in self.item_counts
        }


MostPopularRecommender = PopularityRecommender
