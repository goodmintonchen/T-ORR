from __future__ import annotations

import argparse
from pathlib import Path

import torch


def make_records(count: int, dims: tuple[int, int, int], seed: int) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    text_dim, audio_dim, video_dim = dims
    records = []
    for index in range(count):
        records.append(
            {
                "id": f"synthetic-{index:04d}",
                "text": torch.randn(8 + index % 7, text_dim, generator=generator),
                "audio": torch.randn(40 + index % 13, audio_dim, generator=generator),
                "video": torch.randn(24 + index % 11, video_dim, generator=generator),
                "label": index % 2,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--dims", type=int, nargs=3, default=(1024, 1024, 1024))
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = make_records(args.count, tuple(args.dims), seed=42)
    train_end = int(len(records) * 0.7)
    validation_end = int(len(records) * 0.85)
    torch.save(records[:train_end], output / "train.pt")
    torch.save(records[train_end:validation_end], output / "validation.pt")
    torch.save(records[validation_end:], output / "test.pt")


if __name__ == "__main__":
    main()

