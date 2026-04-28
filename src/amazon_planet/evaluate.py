from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from amazon_planet import LABELS
from amazon_planet.config import load_config
from amazon_planet.data import make_dataloaders
from amazon_planet.metrics import fbeta_score_multilabel, per_label_metrics
from amazon_planet.models import build_model
from amazon_planet.train import resolve_device, validate


def evaluate(
    config_path: str,
    checkpoint_path: str,
    num_workers: int | None = 0,
    per_label_output: str | None = None,
) -> None:
    config = load_config(config_path)
    if num_workers is not None:
        config["data"]["num_workers"] = num_workers
    device = resolve_device(config["training"]["device"])
    _, val_loader = make_dataloaders(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    threshold = float(checkpoint.get("best_threshold", config["training"]["threshold"]))
    val_loss, _, y_true, y_prob = validate(model, val_loader, criterion, device, threshold)
    y_pred = y_prob >= threshold
    val_f2 = fbeta_score_multilabel(y_true, y_pred)
    label_metrics = per_label_metrics(y_true, y_pred, LABELS)

    metrics = {
        "val_loss": val_loss,
        "threshold": threshold,
        "val_f2": val_f2,
        "num_validation_samples": int(y_true.shape[0]),
        "mean_predicted_labels": float((y_prob >= threshold).sum(axis=1).mean()),
        "mean_true_labels": float(np.asarray(y_true).sum(axis=1).mean()),
        "best_label_by_f2": str(label_metrics.iloc[0]["label"]),
        "worst_label_by_f2": str(label_metrics.iloc[-1]["label"]),
    }
    print(json.dumps(metrics, indent=2))
    print("\nPer-label metrics:")
    print(label_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if per_label_output:
        label_metrics.to_csv(per_label_output, index=False)
        print(f"\nSaved per-label metrics to {per_label_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--per-label-output", default="outputs/per_label_metrics.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.checkpoint, args.num_workers, args.per_label_output)
