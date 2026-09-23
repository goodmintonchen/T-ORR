from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .alignment import DynamicLocalityConstrainedAlignment
from .config import ModelConfig
from .decomposition import (
    StructurePreservingGeometricDecomposition,
    topology_preservation_loss,
)
from .routing import ContrastiveIncongruityRouting, gate_sparsity_loss


class TORR(nn.Module):
    """Text-Anchored Orthogonal Residual Rectification network.

    The module consumes frozen encoder features and implements the trainable
    fusion strategy described in the paper.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim

        self.text_projection = nn.Linear(config.text_input_dim, dim)
        self.audio_projection = nn.Linear(config.audio_input_dim, dim)
        self.video_projection = nn.Linear(config.video_input_dim, dim)

        alignment_kwargs = dict(
            hidden_dim=dim,
            num_heads=config.attention_heads,
            max_shift=config.max_alignment_shift,
            dropout=config.dropout,
        )
        self.audio_alignment = DynamicLocalityConstrainedAlignment(**alignment_kwargs)
        self.video_alignment = DynamicLocalityConstrainedAlignment(**alignment_kwargs)
        self.audio_decomposition = StructurePreservingGeometricDecomposition(
            dim, config.basis_rank
        )
        self.video_decomposition = StructurePreservingGeometricDecomposition(
            dim, config.basis_rank
        )
        self.routing = ContrastiveIncongruityRouting(dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        text_features: Tensor,
        audio_features: Tensor,
        video_features: Tensor,
        text_mask: Tensor | None = None,
        audio_mask: Tensor | None = None,
        video_mask: Tensor | None = None,
        labels: Tensor | None = None,
        lambda_structure: float = 0.5,
        lambda_gate_sparsity: float = 0.1,
    ) -> dict[str, Tensor]:
        text = self.dropout(self.text_projection(text_features))
        audio = self.dropout(self.audio_projection(audio_features))
        video = self.dropout(self.video_projection(video_features))

        aligned_audio, audio_attention, audio_centers = self.audio_alignment(
            text, audio, audio_mask
        )
        aligned_video, video_attention, video_centers = self.video_alignment(
            text, video, video_mask
        )
        audio_parts = self.audio_decomposition(text, aligned_audio)
        video_parts = self.video_decomposition(text, aligned_video)

        resonance = audio_parts["resonance"] + video_parts["resonance"]
        dissonance = audio_parts["dissonance"] + video_parts["dissonance"]
        routed = self.routing(text, resonance, dissonance, text_mask)

        output = {
            **routed,
            "audio_attention": audio_attention,
            "video_attention": video_attention,
            "audio_centers": audio_centers,
            "video_centers": video_centers,
            "resonance": resonance,
            "dissonance": dissonance,
        }
        if labels is not None:
            labels = labels.to(routed["logits"].dtype)
            task_loss = F.binary_cross_entropy_with_logits(routed["logits"], labels)
            structure_loss = topology_preservation_loss(
                audio_parts["mapped"], audio_parts["resonance"], text_mask
            ) + topology_preservation_loss(
                video_parts["mapped"], video_parts["resonance"], text_mask
            )
            sparsity_loss = gate_sparsity_loss(routed["gate"], labels, text_mask)
            total_loss = (
                task_loss
                + lambda_structure * structure_loss
                + lambda_gate_sparsity * sparsity_loss
            )
            output.update(
                {
                    "loss": total_loss,
                    "task_loss": task_loss,
                    "structure_loss": structure_loss,
                    "gate_sparsity_loss": sparsity_loss,
                }
            )
        return output

