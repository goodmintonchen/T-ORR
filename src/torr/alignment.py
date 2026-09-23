from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DynamicLocalityConstrainedAlignment(nn.Module):
    """Align a non-verbal sequence to text tokens with a learnable local window.

    The implementation follows Equations (1)-(3) in the paper. A semantic shift
    predicts the center of a Gaussian mask for every token. The mask is added to
    scaled dot-product attention before normalisation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        max_shift: float = 0.15,
        initial_sigma: float = 3.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_shift = max_shift

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.shift = nn.Linear(hidden_dim, 1)
        self.raw_sigma = nn.Parameter(torch.tensor(float(initial_sigma)))
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        text: Tensor,
        modality: Tensor,
        modality_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, text_length, _ = text.shape
        modality_length = modality.size(1)

        q = self._split_heads(self.query(text))
        k = self._split_heads(self.key(modality))
        v = self._split_heads(self.value(modality))

        base = torch.linspace(0.0, 1.0, text_length, device=text.device, dtype=text.dtype)
        base = base.view(1, text_length).expand(batch, -1)
        relative_shift = self.max_shift * torch.tanh(self.shift(text).squeeze(-1))
        centers = (base + relative_shift).clamp(0.0, 1.0) * max(modality_length - 1, 1)

        frames = torch.arange(modality_length, device=text.device, dtype=text.dtype)
        distance = frames.view(1, 1, modality_length) - centers.unsqueeze(-1)
        sigma = F.softplus(self.raw_sigma).clamp_min(1e-4)
        locality_mask = -(distance.square()) / (2.0 * sigma.square())

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + locality_mask.unsqueeze(1)
        if modality_mask is not None:
            scores = scores.masked_fill(~modality_mask[:, None, None, :].bool(), -1e4)

        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        aligned = torch.matmul(attention, v).transpose(1, 2).contiguous()
        aligned = aligned.view(batch, text_length, self.hidden_dim)
        return self.output(aligned), attention, centers

