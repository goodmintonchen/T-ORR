from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .data import FeatureDataset, collate_features
from .model import TORR
from .train import choose_device, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a T-ORR checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TORR(ModelConfig.from_mapping(saved["model_config"]))
    model.load_state_dict(saved["model"])
    model.to(device)
    training = TrainingConfig.from_mapping(saved["training_config"])
    loader = DataLoader(
        FeatureDataset(args.data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_features,
    )
    print(json.dumps(evaluate(model, loader, device, training), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

