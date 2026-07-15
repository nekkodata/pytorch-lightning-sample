import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.losses.focal_loss import FocalLoss


class EfficientNetClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        gamma: float = 2.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

        self.criterion = FocalLoss(gamma=gamma)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits.softmax(dim=1), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits.softmax(dim=1), y)
        self.val_f1(logits.softmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
