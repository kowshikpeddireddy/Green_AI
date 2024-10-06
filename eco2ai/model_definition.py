import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
class LigResNeXt(pl.LightningModule):
    def __init__(self, lr, num_class, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_class)

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)

        self.train_acc(torch.argmax(logits, dim=1), y)

        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)

        self.val_acc(torch.argmax(logits, dim=1), y)

        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)

        self.test_acc(torch.argmax(logits, dim=1), y)

        self.log('test_loss', loss.item(), on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        return preds
