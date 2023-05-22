# Fixing example from:
# https://torchgeo.readthedocs.io/en/stable/tutorials/pretrained_weights.html

import os
import csv
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import timm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from torchgeo.datamodules import EuroSATDataModule
from torchgeo.trainers import ClassificationTask
from torchgeo.models import ResNet50_Weights, ViTSmall16_Weights
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def main():
    # we set a flag to check to see whether the notebook is currently being run by PyTest, if this is the case then
    # we'll skip the expensive training.
    in_tests = "PYTEST_CURRENT_TEST" in os.environ
    root = os.path.join(tempfile.gettempdir(), "eurosat")

    datamodule = EuroSATDataModule(root=root, batch_size=64, num_workers=4, download=True)

    weights = ResNet50_Weights.SENTINEL2_ALL_MOCO

    task = ClassificationTask(
        model="resnet50",
        loss="ce",
        weights=weights,
        in_channels=13,
        num_classes=10,
        learning_rate=0.001,
        learning_rate_schedule_patience=5,
    )

    # in_chans = weights.meta["in_chans"]
    # model = timm.create_model("resnet50", in_chans=in_chans, num_classes=10)
    # model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    experiment_dir = os.path.join(tempfile.gettempdir(), "eurosat_results")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=experiment_dir, save_top_k=1, save_last=True
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10)

    tensorboard_logger = TensorBoardLogger(save_dir=experiment_dir, name="pretrained_weights_logs")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[tensorboard_logger],
        default_root_dir=experiment_dir,
        min_epochs=1,
        max_epochs=10,
        fast_dev_run=in_tests,
    )

    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    main()
