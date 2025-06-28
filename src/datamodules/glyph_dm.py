from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GlyphDataset(Dataset):
    def __init__(self, csv_path: Path, split: str):
        df = pd.read_csv(csv_path)
        self.df = df[df.split == split].reset_index(drop=True)
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

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent,
            pin_memory=True,
        )
