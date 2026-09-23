from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, TypeVar


T = TypeVar("T")


def _from_mapping(cls: type[T], values: Mapping[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**dict(values))


@dataclass(frozen=True)
class ModelConfig:
    text_input_dim: int = 1024
    audio_input_dim: int = 1024
    video_input_dim: int = 1024
    hidden_dim: int = 256
    basis_rank: int = 32
    attention_heads: int = 8
    max_alignment_shift: float = 0.15
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        return _from_mapping(cls, values)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lambda_structure: float = 0.5
    lambda_gate_sparsity: float = 0.1
    early_stopping_patience: int = 5
    decision_threshold: float = 0.5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        return _from_mapping(cls, values)

