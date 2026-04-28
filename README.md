# COMP6252 CW2 - Planet Amazon Multi-label Classification

This project uses the Kaggle competition **Planet: Understanding the Amazon from Space** as a practical deep learning coursework task.

The task is multi-label satellite image classification. Given an Amazon satellite image, the model predicts one or more labels describing weather, land cover, and land use.

## Project Structure

```text
.
├── config/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── reports/
│   └── report_outline.md
├── scripts/
│   └── download_data.ps1
├── src/
│   └── amazon_planet/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── metrics.py
│       ├── models.py
│       └── train.py
├── requirements.txt
└── .gitignore
```

## Data Setup

Download the Kaggle data into `data/raw/`.

Expected minimum layout:

```text
data/raw/
├── train_v2.csv
└── train-jpg/
    ├── train_0.jpg
    ├── train_1.jpg
    └── ...
```

If Kaggle CLI is configured, run:

```powershell
.\scripts\download_data.ps1
```

## Environment

Create and activate a Python environment, then install dependencies:

```powershell
pip install -r requirements.txt
pip install -e .
```

## Train

```powershell
python -m amazon_planet.train --config config/default.yaml
```

The best checkpoint is saved to `outputs/best_model.pt`.

## Train Baseline

```powershell
python -m amazon_planet.train --config config/baseline_cnn.yaml
```

The baseline is a small CNN trained from scratch. Use it to compare against the ImageNet-pretrained ResNet18 main model.

## Visualise Dataset

```powershell
python -m amazon_planet.visualize_dataset --config config/default.yaml --output-dir reports/figures
```

This creates report-ready figures such as label frequency, label co-occurrence, random image samples, examples by label, and label-density statistics.

## Evaluate

```powershell
python -m amazon_planet.evaluate --config config/default.yaml --checkpoint outputs/best_model.pt
```

## Coursework Direction

Recommended report experiments:

1. Dataset visualisation: label frequency, example images, label co-occurrence.
2. Baseline CNN or ResNet18 from scratch.
3. Transfer learning with ImageNet-pretrained ResNet18/ResNet50.
4. Threshold tuning for multi-label prediction.
5. Discussion of F2 score, recall, rare labels, and class imbalance.
