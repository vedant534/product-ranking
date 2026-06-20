"""Artifact-only GRU loading and next-item inference helpers."""

from pathlib import Path

from models.gru4rec import GRU4Rec
from models.train_gru import resolve_device


def load_gru_model(checkpoint_path: Path, device: str = "auto") -> GRU4Rec:
    """Load a trained checkpoint and move it to the requested inference device."""
    resolved_device = resolve_device(device)
    model = GRU4Rec.load_checkpoint(checkpoint_path, map_location="cpu")
    model.to(resolved_device)
    model.eval()
    return model


def recommend_next_items(
    checkpoint_path: Path,
    prefix_items: list[int],
    k: int,
    device: str = "auto",
) -> list[int]:
    """Load a frozen checkpoint and return top-k mapped item indices."""
    return load_gru_model(checkpoint_path, device=device).recommend(prefix_items, k)
