"""Candidate-aware GRU scoring for hard-negative training and reranking."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch.nn import functional

from models.gru4rec import GRU4Rec


class CandidateGRU(GRU4Rec):
    """GRU4Rec architecture trained and evaluated against per-prefix candidates."""

    def candidate_logits(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score a different candidate set for each prefix in a batch."""
        if candidate_ids.ndim != 2 or candidate_mask.shape != candidate_ids.shape:
            raise ValueError("candidate_ids and candidate_mask must have matching 2D shapes")
        hidden = self.encode(input_ids, lengths)
        safe_ids = candidate_ids.clamp(min=0, max=self.num_items - 1)
        weights = self.output.weight[safe_ids]
        biases = self.output.bias[safe_ids]
        logits = (weights * hidden.unsqueeze(1)).sum(dim=-1) + biases
        return logits.masked_fill(~candidate_mask, -torch.inf)

    def candidate_loss(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy where column zero is the explicitly included positive."""
        logits = self.candidate_logits(input_ids, lengths, candidate_ids, candidate_mask)
        targets = torch.zeros(len(input_ids), dtype=torch.long, device=input_ids.device)
        return functional.cross_entropy(logits, targets)

    def score_items(
        self,
        prefix_items: list[int],
        candidate_items: list[int],
    ) -> dict[int, float]:
        """Score only supplied candidates without materializing full-catalog logits."""
        valid = [
            int(item)
            for item in candidate_items
            if 0 <= int(item) < self.num_items and int(item) != self.unknown_idx
        ]
        if not valid:
            return {}
        prefix = [
            int(item) if 0 <= int(item) < self.num_items else self.unknown_idx
            for item in prefix_items[-self.max_sequence_length :]
        ] or [self.unknown_idx]
        device = next(self.parameters()).device
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        lengths = torch.tensor([len(prefix)], dtype=torch.long, device=device)
        candidate_ids = torch.tensor([valid], dtype=torch.long, device=device)
        candidate_mask = torch.ones_like(candidate_ids, dtype=torch.bool)
        self.eval()
        with torch.no_grad():
            scores = self.candidate_logits(
                input_ids, lengths, candidate_ids, candidate_mask
            )[0].cpu().tolist()
        return {item: float(score) for item, score in zip(valid, scores)}

    def recommend_from_candidates(
        self,
        prefix_items: Sequence[int],
        candidate_items: Sequence[int],
        k: int,
    ) -> list[int]:
        scores = self.score_items(list(prefix_items), list(candidate_items))
        return sorted(scores, key=lambda item: (-scores[item], item))[:k]

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        map_location: str = "cpu",
    ) -> "CandidateGRU":
        return super().load_checkpoint(path, map_location=map_location)
