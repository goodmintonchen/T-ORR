from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .data import FeatureDataset, collate_features
from .metrics import binary_metrics
from .model import TORR


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: TORR, loader: DataLoader, device: torch.device, config: TrainingConfig) -> dict:
    model.eval()
    logits, labels, losses = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(
            **{key: value for key, value in batch.items() if key != "ids"},
            lambda_structure=config.lambda_structure,
            lambda_gate_sparsity=config.lambda_gate_sparsity,
        )
        logits.append(output["logits"].cpu())
        labels.append(batch["labels"].cpu())
        losses.append(float(output["loss"]))
    metrics = binary_metrics(torch.cat(logits), torch.cat(labels), config.decision_threshold)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train T-ORR on pre-extracted features")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    random.seed(raw.get("seed", 42))
    torch.manual_seed(raw.get("seed", 42))

    model_config = ModelConfig.from_mapping(raw["model"])
    training_config = TrainingConfig.from_mapping(raw["training"])
    device = choose_device(raw.get("device", "auto"))
    model = TORR(model_config).to(device)

    train_loader = DataLoader(
        FeatureDataset(raw["data"]["train_file"]),
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate_features,
    )
    validation_loader = DataLoader(
        FeatureDataset(raw["data"]["validation_file"]),
        batch_size=training_config.batch_size,
        shuffle=False,
        collate_fn=collate_features,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    output_dir = Path(raw["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / raw["output"]["checkpoint_name"]
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                **{key: value for key, value in batch.items() if key != "ids"},
                lambda_structure=training_config.lambda_structure,
                lambda_gate_sparsity=training_config.lambda_gate_sparsity,
            )
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(output["loss"])

        validation = evaluate(model, validation_loader, device, training_config)
        report = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        print(json.dumps(report, sort_keys=True))

        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": raw["model"],
                    "training_config": raw["training"],
                    "epoch": epoch,
                    "validation": validation,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}")
                break


if __name__ == "__main__":
    main()

