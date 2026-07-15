from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from src.data.imbalance import make_long_tailed_indices
from src.transforms import build_eval_transforms, build_train_transforms


class CIFAR10ImbalancedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        num_classes: int = 10,
        img_size: int = 64,
        batch_size: int = 32,
        num_workers: int = 2,
        imbalance_factor: float = 0.1,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        train_full = CIFAR10(
            self.hparams.data_dir, train=True, transform=build_train_transforms(img_size=self.hparams.img_size)
        )
        val_source = CIFAR10(
            self.hparams.data_dir, train=True, transform=build_eval_transforms(img_size=self.hparams.img_size)
        )
        self.test_set = CIFAR10(
            self.hparams.data_dir, train=False, transform=build_eval_transforms(img_size=self.hparams.img_size)
        )

        targets = train_full.targets
        n_val = int(len(targets) * self.hparams.val_split)
        generator = torch.Generator().manual_seed(self.hparams.seed)
        perm = torch.randperm(len(targets), generator=generator).tolist()
        val_indices = perm[:n_val]
        remaining_indices = perm[n_val:]

        remaining_targets = [targets[i] for i in remaining_indices]
        long_tailed_relative = make_long_tailed_indices(
            remaining_targets,
            num_classes=self.hparams.num_classes,
            imbalance_factor=self.hparams.imbalance_factor,
            seed=self.hparams.seed,
        )
        train_indices = [remaining_indices[i] for i in long_tailed_relative]

        self.train_set = Subset(train_full, train_indices)
        self.val_set = Subset(val_source, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
