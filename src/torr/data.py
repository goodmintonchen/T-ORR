from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class FeatureDataset(Dataset[dict[str, Any]]):
    """Dataset for pre-extracted frozen encoder features stored with torch.save."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records = torch.load(self.path, map_location="cpu", weights_only=False)
        if isinstance(records, dict) and "records" in records:
            records = records["records"]
        if not isinstance(records, list):
            raise TypeError("Expected a list of sample dictionaries or {'records': list}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        required = {"text", "audio", "video", "label"}
        missing = required - set(record)
        if missing:
            raise KeyError(f"Sample {index} is missing keys: {sorted(missing)}")
        return record


def _pad(items: list[Tensor]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([item.size(0) for item in items], dtype=torch.long)
    padded = pad_sequence(items, batch_first=True)
    positions = torch.arange(padded.size(1)).unsqueeze(0)
    mask = positions < lengths.unsqueeze(1)
    return padded, mask


def collate_features(records: list[dict[str, Any]]) -> dict[str, Any]:
    text, text_mask = _pad([record["text"].float() for record in records])
    audio, audio_mask = _pad([record["audio"].float() for record in records])
    video, video_mask = _pad([record["video"].float() for record in records])
    labels = torch.tensor([record["label"] for record in records], dtype=torch.float32)
    return {
        "ids": [record.get("id", str(index)) for index, record in enumerate(records)],
        "text_features": text,
        "audio_features": audio,
        "video_features": video,
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "video_mask": video_mask,
        "labels": labels,
    }

