from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class GlyphDataset(Dataset):
    def __init__(self, csv_path: Path, split: str):
        df = pd.read_csv(csv_path)
        self.df = df[df.split == split].reset_index(drop=True)

        if split == "train":
            self.tf = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                    T.RandomRotation(10),
                    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )
        else:  # val / test
            self.tf = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert("RGB")
        return self.tf(img), int(row.label_id)


class GlyphDataModule(pl.LightningDataModule):
    def __init__(
        self, csv_path, batch_size=64, num_workers=4, persistent_workers=False
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent = persistent_workers

    def setup(self, stage=None):
        self.train_ds = GlyphDataset(self.csv_path, "train")
        self.val_ds = GlyphDataset(self.csv_path, "val")
        self.num_classes = len(set(self.train_ds.df.label_id))

        label_counts = self.train_ds.df.label_id.value_counts()
        weights = 1.0 / label_counts[self.train_ds.df.label_id].values
        self.sample_weights = torch.DoubleTensor(weights)

    def train_dataloader(self):
        sampler = WeightedRandomSampler(
            self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,  # ← replaces shuffle=True
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=True,
        )
