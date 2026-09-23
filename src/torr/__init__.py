"""T-ORR: text-anchored orthogonal residual rectification."""

from .config import ModelConfig, TrainingConfig
from .model import TORR

__all__ = ["ModelConfig", "TrainingConfig", "TORR"]
__version__ = "0.1.0"

