from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from amazon_planet import LABELS


class PlanetAmazonDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_to_index = {label: index for index, label in enumerate(LABELS)}

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / f"{row.image_name}.jpg"
        image = Image.open(image_path).convert("RGB")

        target = torch.zeros(len(LABELS), dtype=torch.float32)
        for label in row.tags.split():
            target[self.label_to_index[label]] = 1.0

        if self.transform is not None:
            image = self.transform(image)

        return image, target


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def make_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    train_config = config["training"]

    dataframe = pd.read_csv(data_config["csv_file"])
    train_df, val_df = train_test_split(
        dataframe,
        test_size=data_config["val_size"],
        random_state=config["seed"],
        shuffle=True,
    )

    train_transform, val_transform = get_transforms(train_config["image_size"])

    train_dataset = PlanetAmazonDataset(train_df, data_config["image_dir"], train_transform)
    val_dataset = PlanetAmazonDataset(val_df, data_config["image_dir"], val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader
