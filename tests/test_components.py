import torch

from torr.alignment import DynamicLocalityConstrainedAlignment
from torr.decomposition import StructurePreservingGeometricDecomposition


def test_alignment_respects_padding_mask() -> None:
    layer = DynamicLocalityConstrainedAlignment(
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
    )
    text = torch.randn(2, 5, 16)
    modality = torch.randn(2, 9, 16)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )

    aligned, attention, centers = layer(text, modality, mask)

    assert aligned.shape == (2, 5, 16)
    assert centers.shape == (2, 5)
    assert torch.allclose(attention[0, :, :, 6:], torch.zeros_like(attention[0, :, :, 6:]))
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)))


def test_orthogonal_split_reconstructs_mapped_features() -> None:
    layer = StructurePreservingGeometricDecomposition(hidden_dim=16, basis_rank=4)
    output = layer(torch.randn(2, 6, 16), torch.randn(2, 6, 16))

    reconstructed = output["resonance"] + output["dissonance"]
    assert torch.allclose(reconstructed, output["mapped"], atol=1e-6)

    basis = output["basis"]
    gram = basis.transpose(-2, -1) @ basis
    identity = torch.eye(4).view(1, 1, 4, 4).expand_as(gram)
    assert torch.allclose(gram, identity, atol=1e-5)

