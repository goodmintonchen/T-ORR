from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_mean(sequence: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ContrastiveIncongruityRouting(nn.Module):
    """Build literal and incongruous states and classify their semantic shift."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.resonance_transform = nn.Linear(hidden_dim, hidden_dim)
        self.dissonance_gate = nn.Linear(hidden_dim, 1)
        self.literal_norm = nn.LayerNorm(hidden_dim)
        self.incongruous_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text: Tensor,
        resonance: Tensor,
        dissonance: Tensor,
        text_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        literal = self.literal_norm(text + self.resonance_transform(resonance))
        gate = torch.sigmoid(self.dissonance_gate(dissonance))
        incongruous = self.incongruous_norm(text * gate)

        literal_global = masked_mean(literal, text_mask)
        incongruous_global = masked_mean(incongruous, text_mask)
        conflict = torch.cat(
            [
                literal_global,
                incongruous_global,
                (literal_global - incongruous_global).abs(),
            ],
            dim=-1,
        )
        return {
            "logits": self.classifier(conflict).squeeze(-1),
            "gate": gate,
            "literal": literal,
            "incongruous": incongruous,
            "conflict": conflict,
        }


def gate_sparsity_loss(gate: Tensor, labels: Tensor, text_mask: Tensor | None = None) -> Tensor:
    per_token = gate.squeeze(-1)
    if text_mask is None:
        per_sample = per_token.mean(dim=1)
    else:
        weights = text_mask.to(per_token.dtype)
        per_sample = (per_token * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    sincere_weight = 1.0 - labels.to(per_sample.dtype)
    return (sincere_weight * per_sample).mean()

