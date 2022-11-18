import torch.nn as nn
from torch.optim import Adam
import lightning as pl
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


class ModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.precision_score = MulticlassPrecision(2)
        self.recall_score = MulticlassRecall(2)
        self.f1_score = MulticlassF1Score(2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        self.precision_score(y_hat, y)
        self.recall_score(y_hat, y)
        self.f1_score(y_hat, y)

        self.log('val_precision', self.precision_score)
        self.log('val_recall', self.recall_score)
        self.log('val_f1', self.f1_score)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        self.precision_score(y_hat, y)
        self.recall_score(y_hat, y)
        self.f1_score(y_hat, y)

        self.log('test_precision', self.precision_score)
        self.log('test_recall', self.recall_score)
        self.log('test_f1', self.f1_score)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
