"""Simple GRU4Rec-style next-item prediction model."""

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRU4Rec(nn.Module):
    """Embed a prefix, encode it with a GRU, and score the next item."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 100,
        hidden_dim: int = 100,
        num_layers: int = 1,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
        max_sequence_length: int = 50,
        unknown_idx: int = 0,
    ) -> None:
        super().__init__()
        if num_items < 2:
            raise ValueError("num_items must include UNK and at least one train item")
        if embedding_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("embedding_dim, hidden_dim, and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")

        resolved_padding_idx = num_items if padding_idx is None else padding_idx
        if resolved_padding_idx < num_items:
            raise ValueError("padding_idx must be outside the predicted item classes")
        if not 0 <= unknown_idx < num_items:
            raise ValueError("unknown_idx must be a predicted item class")

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_probability = float(dropout)
        self.padding_idx = resolved_padding_idx
        self.max_sequence_length = max_sequence_length
        self.unknown_idx = unknown_idx
        self.checkpoint_metadata: dict[str, object] = {}

        self.embedding = nn.Embedding(
            num_embeddings=resolved_padding_idx + 1,
            embedding_dim=embedding_dim,
            padding_idx=resolved_padding_idx,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, num_items)

    def encode(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return the final GRU state for each right-padded prefix."""
        embedded = self.embedding_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return full-catalog next-item logits for a right-padded batch."""
        return self.output(self.encode(input_ids, lengths))

    def predict_topk(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
        exclude_input_items: bool = False,
    ) -> torch.Tensor:
        """Return top-k catalog indices while excluding the synthetic UNK class."""
        available_items = self.num_items - 1
        if available_items <= 0:
            return torch.empty((len(input_ids), 0), dtype=torch.long, device=input_ids.device)
        cutoff = min(k, available_items)
        logits = self(input_ids, lengths)
        logits[:, self.unknown_idx] = -torch.inf
        if exclude_input_items:
            for row_index, length in enumerate(lengths.detach().cpu().tolist()):
                seen_items = input_ids[row_index, :length]
                seen_items = seen_items[
                    (seen_items >= 0) & (seen_items < self.num_items)
                ]
                logits[row_index, seen_items] = -torch.inf
        return torch.topk(logits, k=cutoff, dim=1).indices

    def recommend(self, prefix_items: list[int], k: int) -> list[int]:
        """Recommend from one mapped prefix using batch-size-one inference."""
        if k <= 0:
            raise ValueError("k must be positive")
        prefix = [
            item if 0 <= int(item) < self.num_items else self.unknown_idx
            for item in prefix_items[-self.max_sequence_length :]
        ]
        if not prefix:
            prefix = [self.unknown_idx]
        device = next(self.parameters()).device
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        lengths = torch.tensor([len(prefix)], dtype=torch.long, device=device)
        seen = {
            item
            for item in prefix
            if 0 <= item < self.num_items and item != self.unknown_idx
        }
        cutoff = min(k, self.num_items - 1 - len(seen))
        if cutoff <= 0:
            return []
        self.eval()
        with torch.no_grad():
            predictions = self.predict_topk(
                input_ids,
                lengths,
                cutoff,
                exclude_input_items=True,
            )
        return [
            item
            for item in predictions[0].detach().cpu().tolist()
            if item not in seen and item != self.unknown_idx
        ][:cutoff]

    def recommend_batch(
        self,
        prefixes: Sequence[Sequence[int]],
        k: int,
        batch_size: int = 512,
    ) -> list[list[int]]:
        """Recommend for many prefixes with batched full-catalog inference."""
        if k <= 0 or batch_size <= 0:
            raise ValueError("k and batch_size must be positive")
        normalized = [
            [
                int(item) if 0 <= int(item) < self.num_items else self.unknown_idx
                for item in list(prefix)[-self.max_sequence_length :]
            ]
            or [self.unknown_idx]
            for prefix in prefixes
        ]
        device = next(self.parameters()).device
        predictions = []
        self.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                batch = normalized[start : start + batch_size]
                lengths = torch.tensor(
                    [len(prefix) for prefix in batch],
                    dtype=torch.long,
                    device=device,
                )
                input_ids = torch.full(
                    (len(batch), int(lengths.max())),
                    fill_value=self.padding_idx,
                    dtype=torch.long,
                    device=device,
                )
                for row_index, prefix in enumerate(batch):
                    input_ids[row_index, : len(prefix)] = torch.tensor(
                        prefix,
                        dtype=torch.long,
                        device=device,
                    )
                maximum_seen = max(
                    len(
                        {
                            item
                            for item in prefix
                            if 0 <= item < self.num_items and item != self.unknown_idx
                        }
                    )
                    for prefix in batch
                )
                top_items = self.predict_topk(
                    input_ids,
                    lengths,
                    min(k + maximum_seen, self.num_items - 1),
                    exclude_input_items=True,
                )
                for prefix, row in zip(batch, top_items.detach().cpu().tolist()):
                    seen = set(prefix)
                    predictions.append(
                        [
                            item
                            for item in row
                            if item not in seen and item != self.unknown_idx
                        ][:k]
                    )
        return predictions

    def score_items(
        self,
        prefix_items: list[int],
        candidate_items: list[int],
    ) -> dict[int, float]:
        """Return neural logits for valid candidate item indices."""
        valid_candidates = [
            int(item)
            for item in candidate_items
            if 0 <= int(item) < self.num_items and int(item) != self.unknown_idx
        ]
        if not valid_candidates:
            return {}
        prefix = [
            int(item) if 0 <= int(item) < self.num_items else self.unknown_idx
            for item in prefix_items[-self.max_sequence_length :]
        ] or [self.unknown_idx]
        device = next(self.parameters()).device
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        lengths = torch.tensor([len(prefix)], dtype=torch.long, device=device)
        self.eval()
        with torch.no_grad():
            logits = self(input_ids, lengths)[0]
            candidate_tensor = torch.tensor(
                valid_candidates,
                dtype=torch.long,
                device=device,
            )
            candidate_scores = logits[candidate_tensor].detach().cpu().tolist()
        return {
            item: float(score)
            for item, score in zip(valid_candidates, candidate_scores)
        }

    def model_config(self) -> dict[str, object]:
        return {
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_probability,
            "padding_idx": self.padding_idx,
            "max_sequence_length": self.max_sequence_length,
            "unknown_idx": self.unknown_idx,
        }

    def save_checkpoint(
        self,
        path: Path,
        metadata: Optional[dict[str, object]] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_config": self.model_config(),
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        map_location: str = "cpu",
    ) -> "GRU4Rec":
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.checkpoint_metadata = checkpoint.get("metadata", {})
        model.eval()
        return model
