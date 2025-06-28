from multiprocessing import freeze_support, set_start_method

import pytorch_lightning as pl
import torch.multiprocessing as mp
from pytorch_lightning.callbacks import ModelCheckpoint

from src.datamodules.glyph_dm import GlyphDataModule
from src.models.vit_classifier import ViTClassifier


# This trains a Vision Transformer (ViT) model on a dataset of glyphs.
def main():
    import torch

    set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("high")  # optimized for RTX 5090 tensor cores

    # force the spawn method once
    mp.set_start_method("spawn", force=True)

    dm = GlyphDataModule(
        "data/processed/train_val_strat.csv",
        batch_size=128,
        num_workers=4,
        persistent_workers=True,  # use persistent workers for faster data loading
    )
    dm.setup()

    model = ViTClassifier(num_classes=dm.num_classes)

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[ModelCheckpoint(dirpath="checkpoints", save_last=True)],
        accelerator="gpu",  # explicit is clearer
        devices=1,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    freeze_support()  # needed only on frozen EXEs; harmless otherwise
    main()
