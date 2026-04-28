from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from amazon_planet.config import load_config
from amazon_planet.data import make_dataloaders
from amazon_planet.metrics import fbeta_score_multilabel, optimise_threshold
from amazon_planet.models import build_model


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            targets.append(labels.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    y_true = np.concatenate(targets)
    y_prob = np.concatenate(probabilities)
    f2 = fbeta_score_multilabel(y_true, y_prob >= threshold)
    return float(np.mean(losses)), f2, y_true, y_prob


def train(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["seed"])

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config["training"]["device"])
    train_loader, val_loader = make_dataloaders(config)

    model = build_model(
        name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_f2 = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_losses: list[float] = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=np.mean(train_losses))

        val_loss, val_f2, y_true, y_prob = validate(
            model,
            val_loader,
            criterion,
            device,
            config["training"]["threshold"],
        )
        best_threshold, tuned_f2 = optimise_threshold(y_true, y_prob)

        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_f2": val_f2,
            "best_threshold": best_threshold,
            "tuned_val_f2": tuned_f2,
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        if tuned_f2 > best_f2:
            best_f2 = tuned_f2
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_threshold": best_threshold,
                    "best_f2": best_f2,
                },
                config["outputs"]["checkpoint"],
            )

    with Path(config["outputs"]["metrics_file"]).open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
