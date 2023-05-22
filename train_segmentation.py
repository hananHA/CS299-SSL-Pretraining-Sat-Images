import os
import csv
import tempfile

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import pytorch_lightning as pl
import timm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader, Dataset

from torchgeo.datamodules import LandCoverAIDataModule
from torchgeo.trainers import SemanticSegmentationTask, ClassificationTask
from torchgeo.models import ResNet50_Weights, ViTSmall16_Weights, ResNet18_Weights
from torchgeo.trainers import utils
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datamodules.geo import NonGeoDataModule
import ssl
import argparse
from typing import Optional


ssl._create_default_https_context = ssl._create_unverified_context


def main(cfg):
    root = cfg.data_root
    if not os.path.exists(cfg.data_root):
        os.makedirs(cfg.data_root)

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    datamodule = LandCoverAIDataModule(root=root, batch_size=cfg.batch_size, num_workers=cfg.workers, download=True)

    # Pretrained weights
    # https://torchgeo.readthedocs.io/en/stable/api/models.html
    # https://github.com/facebookresearch/moco#models

    if cfg.segm_backbone == "resnet50":
        weights = ResNet50_Weights.SENTINEL2_RGB_MOCO if cfg.encoder_weights == "ResNet50_Weights.SENTINEL2_RGB_MOCO" else None
    elif cfg.segm_backbone == "resnet18":
        weights = ResNet18_Weights.SENTINEL2_RGB_MOCO if cfg.encoder_weights == "ResNet18_Weights.SENTINEL2_RGB_MOCO" else None
    else:
        weights = None

    task = SemanticSegmentationTask(
        model=cfg.segm_model,
        backbone=cfg.segm_backbone,
        loss=cfg.segm_loss,
        weights=None,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        learning_rate=cfg.learning_rate,
        learning_rate_schedule_patience=cfg.learning_rate_schedule_patience,
        ignore_index=cfg.ignore_index
    )

    # To instante a model without the need to create a task. Can be used in a usual pytorch training way
    # in_chans = weights.meta["in_chans"]
    # model = timm.create_model("resnet50", in_chans=in_chans, num_classes=10)
    # model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    if cfg.train_mode == "SSL4EO-pretrain":
        task.model.encoder.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    if cfg.train_mode == "fs-imagenet-pretrain":
        task.weights = "imagenet"

    if cfg.train_mode == "ssl-imagenet-pretrain":
        weights = torch.load('weights/moco_v2_200ep_pretrain.pth.tar')
        task.model.encoder.load_state_dict(weights, strict=False)
    
    if cfg.eval_mode == "linear":
        for name, param in task.model.encoder.named_parameters():
            param.requires_grad = False
        # for name, param in task.model.encoder.named_parameters():
        #     if not name.startswith("layer4"):
        #         param.requires_grad = False
    
    experiment_dir = cfg.output_dir

    # TODO: maybe monitor val_MulticlassAccuracy (with mode 'max') instead of val_loss (with mode 'min') to save
    #  only the best model
    monitor_metric = cfg.monitor_metric
    mode = cfg.mode_monitor_metric
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode=mode,
        filename="checkpoint-epoch{epoch:02d}-val_loss{val_loss:.2f}",
        dirpath=experiment_dir,
        save_top_k=1,
        save_last=True
    )

    early_stopping_callback = EarlyStopping(monitor=monitor_metric, min_delta=0.00, patience=18, mode=mode)

    wandb_logger = WandbLogger(project="SSL-Pretraining-Sat-Images", save_dir=experiment_dir)

    # tensorboard_logger = TensorBoardLogger(save_dir=experiment_dir, name="pretrained_weights_logs")

    wandb_logger.experiment.config.update({"dataset": cfg.dataset,
                                           "batch_size": cfg.batch_size,
                                           "train_mode": cfg.train_mode,
                                           "eval_mode": cfg.eval_mode,
                                           "num_workers": cfg.workers,
                                           "model": cfg.segm_model,
                                           "backbone": cfg.segm_backbone,
                                           "loss": cfg.segm_loss,
                                           "weights": cfg.encoder_weights,
                                           "in_channels": cfg.in_channels,
                                           "num_classes": cfg.num_classes,
                                           "learning_rate": cfg.learning_rate,
                                           "learning_rate_schedule_patience": cfg.learning_rate_schedule_patience,
                                           "ignore_index": cfg.ignore_index,
                                           "monitor_metric": cfg.monitor_metric,
                                           "mode_monitor_metric": cfg.mode_monitor_metric
                                           })

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[wandb_logger],
        default_root_dir=experiment_dir,
        min_epochs=cfg.min_epochs,
        max_epochs=cfg.epochs,
        fast_dev_run=False,
        gpus=cfg.num_gpus
    )
    print("=========================================")
    print("Starting training...")
    print(f"[Train mode: {cfg.train_mode}]")
    print("=========================================")
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSL Pretraining Satellite Images Arguments")

    # Adding command-line arguments
    parser.add_argument("-dr", "--data-root", type=str, default="data/landcoverai", help="Root Directory")
    parser.add_argument("-d", "--dataset", type=str, default="landcoverai", help="Dataset")
    parser.add_argument("-o", "--output-dir", type=str, default="output/landcoverai", help="Output Directory")
    parser.add_argument("-me", "--min-epochs", type=int, default=20, help="Minimum number of epochs")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of Epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch Size")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("-sm", "--segm-model", type=str, default="deeplabv3+", help="Segmentation Model")
    parser.add_argument("-sb", "--segm-backbone", type=str, default="resnet50", help="Segmentation Backbone")
    parser.add_argument("-sl", "--segm-loss", type=str, default="ce", help="Segmentation Loss")
    parser.add_argument("-ew", "--encoder-weights", type=str, default="ResNet50_Weights.SENTINEL2_RGB_MOCO", help="Backbone Encoder Weights")
    parser.add_argument("-ic", "--in-channels", type=int, default=3, help="Input Channels")
    parser.add_argument("-nc", "--num-classes", type=int, default=5, help="Number of Classes")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("-lrsp", "--learning-rate-schedule-patience", type=int, default=6, help="Learning Rate Schedule Patience")
    parser.add_argument("-ii", "--ignore-index", type=int, default=None, help="Ignore Index")
    parser.add_argument("-mm", "--monitor-metric", type=str, default="val_loss", help="Monitor Metric")
    parser.add_argument("-mmode", "--mode-monitor-metric", type=str, default="min", help="Mode Monitor Metric")
    parser.add_argument("-ng", "--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("-tm", "--train-mode", type=str, default="SSL4EO-pretrain",
                        choices=["scratch", "SSL4EO-pretrain", "fs-imagenet-pretrain", "ssl-imagenet-pretrain"], help="Train Mode")
    parser.add_argument("-em", "--eval-mode", type=str, default="finetune",
                        choices=["finetune", "linear"], help="Evaluation mode")

    # parser add_argument with options example
    # parser.add_argument("-m", "--model", type=str, default="resnet50", choices=["resnet50", "resnet101", "resnet152"],

    # Parsing the arguments
    args = parser.parse_args()
    print(args)

    main(args)
