from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from amazon_planet import LABELS
from amazon_planet.config import load_config

WEATHER_LABELS = ["clear", "partly_cloudy", "haze", "cloudy"]
COMMON_LAND_LABELS = ["primary", "agriculture", "water", "road", "cultivation", "habitation"]


def build_label_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tags in dataframe["tags"]:
        present = set(tags.split())
        rows.append({label: int(label in present) for label in LABELS})
    return pd.DataFrame(rows)


def save_label_frequency(label_matrix: pd.DataFrame, output_dir: Path) -> pd.Series:
    counts = label_matrix.sum().sort_values(ascending=False)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(x=counts.values, y=counts.index, color="#2f7f8f")
    plt.xlabel("Number of images")
    plt.ylabel("Label")
    plt.title("Label frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_label_frequency.png", dpi=220)
    plt.close()

    return counts


def save_labels_per_image(label_matrix: pd.DataFrame, output_dir: Path) -> None:
    label_counts = label_matrix.sum(axis=1)

    plt.figure(figsize=(7, 4.5))
    sns.countplot(x=label_counts, color="#d39c3f")
    plt.xlabel("Number of labels per image")
    plt.ylabel("Number of images")
    plt.title("Multi-label density")
    plt.tight_layout()
    plt.savefig(output_dir / "viz_labels_per_image.png", dpi=220)
    plt.close()


def save_cooccurrence(label_matrix: pd.DataFrame, labels: list[str], output_path: Path, title: str) -> None:
    cooccurrence = label_matrix[labels].T.dot(label_matrix[labels])

    plt.figure(figsize=(max(6, len(labels) * 0.55), max(5, len(labels) * 0.45)))
    sns.heatmap(cooccurrence, cmap="mako", square=True, cbar_kws={"shrink": 0.7})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def load_rgb_image(image_dir: Path, image_name: str) -> Image.Image:
    return Image.open(image_dir / f"{image_name}.jpg").convert("RGB")


def wrap_title(text: str, max_chars: int = 42) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def save_random_sample_grid(
    dataframe: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    sample_count: int,
    seed: int,
) -> None:
    samples = dataframe.sample(n=min(sample_count, len(dataframe)), random_state=seed).reset_index(drop=True)
    cols = 4
    rows = math.ceil(len(samples) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.25))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, row in zip(axes_list, samples.itertuples(index=False)):
        image = load_rgb_image(image_dir, row.image_name)
        axis.imshow(image)
        axis.set_title(wrap_title(row.tags), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "viz_random_samples.png", dpi=220)
    plt.close()


def save_label_example_grid(
    dataframe: pd.DataFrame,
    image_dir: Path,
    output_dir: Path,
    labels: list[str],
    seed: int,
) -> None:
    rng = random.Random(seed)
    examples = []

    for label in labels:
        matches = dataframe[dataframe["tags"].str.split().apply(lambda tags: label in tags)]
        if matches.empty:
            continue
        row = matches.iloc[rng.randrange(len(matches))]
        examples.append((label, row["image_name"], row["tags"]))

    cols = 4
    rows = math.ceil(len(examples) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.35))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis in axes_list:
        axis.axis("off")

    for axis, (label, image_name, tags) in zip(axes_list, examples):
        image = load_rgb_image(image_dir, image_name)
        axis.imshow(image)
        axis.set_title(f"{label}\n{wrap_title(tags, 36)}", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "viz_examples_by_label.png", dpi=220)
    plt.close()


def save_dataset_summary(dataframe: pd.DataFrame, label_matrix: pd.DataFrame, output_dir: Path) -> None:
    label_counts = label_matrix.sum(axis=1)
    counts = label_matrix.sum().sort_values(ascending=False)
    rare_labels = counts[counts < 1000]

    lines = [
        "# Dataset Visualisation Summary",
        "",
        f"- Images: {len(dataframe)}",
        f"- Labels: {len(LABELS)}",
        f"- Mean labels per image: {label_counts.mean():.3f}",
        f"- Median labels per image: {label_counts.median():.3f}",
        f"- Most frequent label: {counts.index[0]} ({int(counts.iloc[0])})",
        f"- Least frequent label: {counts.index[-1]} ({int(counts.iloc[-1])})",
        f"- Labels with fewer than 1000 examples: {', '.join(rare_labels.index)}",
        "",
    ]
    (output_dir / "viz_dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")

    summary = pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percentage": counts.values / len(dataframe) * 100,
        }
    )
    summary.to_csv(output_dir / "viz_label_summary.csv", index=False)


def visualize_dataset(config_path: str, output_dir: str, sample_count: int, seed: int) -> None:
    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(config["data"]["csv_file"])
    image_dir = Path(config["data"]["image_dir"])
    label_matrix = build_label_matrix(dataframe)

    counts = save_label_frequency(label_matrix, output_path)
    save_labels_per_image(label_matrix, output_path)
    save_cooccurrence(label_matrix, LABELS, output_path / "viz_cooccurrence_all.png", "All label co-occurrence")
    save_cooccurrence(
        label_matrix,
        WEATHER_LABELS,
        output_path / "viz_cooccurrence_weather.png",
        "Weather label co-occurrence",
    )
    save_cooccurrence(
        label_matrix,
        COMMON_LAND_LABELS,
        output_path / "viz_cooccurrence_common_land.png",
        "Common land label co-occurrence",
    )

    rare_labels = counts[counts < 1000].index.tolist()
    if len(rare_labels) >= 2:
        save_cooccurrence(
            label_matrix,
            rare_labels,
            output_path / "viz_cooccurrence_rare.png",
            "Rare label co-occurrence",
        )

    save_random_sample_grid(dataframe, image_dir, output_path, sample_count, seed)
    save_label_example_grid(dataframe, image_dir, output_path, LABELS, seed)
    save_dataset_summary(dataframe, label_matrix, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualisations for the Planet Amazon dataset.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_dataset(args.config, args.output_dir, args.sample_count, args.seed)
