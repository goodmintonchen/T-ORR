from __future__ import annotations

import torch
from torch import Tensor


def binary_metrics(logits: Tensor, labels: Tensor, threshold: float = 0.5) -> dict[str, float]:
    predictions = (torch.sigmoid(logits) >= threshold).long()
    labels = labels.long()
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    correct = int((predictions == labels).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / max(labels.numel(), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

