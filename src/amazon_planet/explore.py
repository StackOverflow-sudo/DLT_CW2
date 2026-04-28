from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from amazon_planet import LABELS
from amazon_planet.config import load_config


def build_label_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    label_rows = []
    for tags in dataframe["tags"]:
        present = set(tags.split())
        label_rows.append({label: int(label in present) for label in LABELS})
    return pd.DataFrame(label_rows)


def run_exploration(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(config["data"]["csv_file"])
    label_matrix = build_label_matrix(dataframe)

    counts = label_matrix.sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#307c8e")
    plt.xlabel("Number of images")
    plt.ylabel("Label")
    plt.title("Planet Amazon label frequency")
    plt.tight_layout()
    plt.savefig(output_path / "label_frequency.png", dpi=200)
    plt.close()

    cooccurrence = label_matrix.T.dot(label_matrix)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cooccurrence, cmap="viridis")
    plt.title("Label co-occurrence")
    plt.tight_layout()
    plt.savefig(output_path / "label_cooccurrence.png", dpi=200)
    plt.close()

    summary = pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percentage": counts.values / len(dataframe) * 100,
        }
    )
    summary.to_csv(output_path / "label_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="reports/figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_exploration(args.config, args.output_dir)
