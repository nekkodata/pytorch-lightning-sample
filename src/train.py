import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.datamodule import CIFAR10ImbalancedDataModule
from src.models.classifier import EfficientNetClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet + Focal Loss image classification (CPU)")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--imbalance-factor", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    datamodule = CIFAR10ImbalancedDataModule(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        imbalance_factor=args.imbalance_factor,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = EfficientNetClassifier(
        num_classes=args.num_classes,
        gamma=args.gamma,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained=args.pretrained,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1),
        EarlyStopping(monitor="val_f1", mode="max", patience=5),
    ]

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
