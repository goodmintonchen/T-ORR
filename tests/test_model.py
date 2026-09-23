import torch

from torr import ModelConfig, TORR


def test_forward_and_losses_are_finite() -> None:
    config = ModelConfig(
        text_input_dim=32,
        audio_input_dim=24,
        video_input_dim=16,
        hidden_dim=32,
        basis_rank=8,
        attention_heads=4,
        dropout=0.0,
    )
    model = TORR(config)
    output = model(
        text_features=torch.randn(3, 7, 32),
        audio_features=torch.randn(3, 19, 24),
        video_features=torch.randn(3, 13, 16),
        text_mask=torch.ones(3, 7, dtype=torch.bool),
        audio_mask=torch.ones(3, 19, dtype=torch.bool),
        video_mask=torch.ones(3, 13, dtype=torch.bool),
        labels=torch.tensor([0.0, 1.0, 1.0]),
    )
    assert output["logits"].shape == (3,)
    assert output["audio_attention"].shape == (3, 4, 7, 19)
    assert output["video_attention"].shape == (3, 4, 7, 13)
    assert torch.isfinite(output["loss"])


def test_backward_reaches_trainable_modules() -> None:
    model = TORR(
        ModelConfig(
            text_input_dim=16,
            audio_input_dim=16,
            video_input_dim=16,
            hidden_dim=16,
            basis_rank=4,
            attention_heads=4,
            dropout=0.0,
        )
    )
    output = model(
        torch.randn(2, 5, 16),
        torch.randn(2, 9, 16),
        torch.randn(2, 8, 16),
        labels=torch.tensor([0.0, 1.0]),
    )
    output["loss"].backward()
    assert model.routing.classifier[-1].weight.grad is not None

