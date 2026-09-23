from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class StructurePreservingGeometricDecomposition(nn.Module):
    """Split mapped non-verbal features into resonance and dissonance."""

    def __init__(self, hidden_dim: int, basis_rank: int) -> None:
        super().__init__()
        if not 0 < basis_rank <= hidden_dim:
            raise ValueError("basis_rank must be in [1, hidden_dim]")
        self.hidden_dim = hidden_dim
        self.basis_rank = basis_rank
        self.modality_map = nn.Linear(hidden_dim, hidden_dim)
        self.basis_generator = nn.Linear(hidden_dim, hidden_dim * basis_rank)

    def forward(self, text: Tensor, aligned: Tensor) -> dict[str, Tensor]:
        mapped = self.modality_map(aligned)
        batch, length, _ = text.shape
        basis = self.basis_generator(text).view(
            batch, length, self.hidden_dim, self.basis_rank
        )
        orthonormal_basis, _ = torch.linalg.qr(basis, mode="reduced")

        mapped_column = mapped.unsqueeze(-1)
        coordinates = torch.matmul(orthonormal_basis.transpose(-2, -1), mapped_column)
        resonance = torch.matmul(orthonormal_basis, coordinates).squeeze(-1)
        dissonance = mapped - resonance
        return {
            "mapped": mapped,
            "resonance": resonance,
            "dissonance": dissonance,
            "basis": orthonormal_basis,
        }


def _masked_mean(sequence: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return sequence.mean(dim=1)
    weights = mask.to(sequence.dtype).unsqueeze(-1)
    return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def topology_preservation_loss(
    mapped: Tensor,
    resonance: Tensor,
    text_mask: Tensor | None = None,
) -> Tensor:
    """Preserve pairwise cosine geometry inside the resonance space."""

    mapped_global = F.normalize(_masked_mean(mapped, text_mask), dim=-1)
    resonance_global = F.normalize(_masked_mean(resonance, text_mask), dim=-1)
    mapped_similarity = mapped_global @ mapped_global.transpose(0, 1)
    resonance_similarity = resonance_global @ resonance_global.transpose(0, 1)
    return F.mse_loss(resonance_similarity, mapped_similarity)

