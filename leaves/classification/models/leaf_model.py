import torch.nn as nn
from torch.optim import Adam
import lightning as pl
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
)
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


class LeafModel(pl.LightningModule):
    def __init__(self, lr, weight_decay, model, **kwargs):
        super().__init__()

        weights = {
            'mobilenetv3s': MobileNet_V3_Small_Weights,
            'mobilenetv3l': MobileNet_V3_Large_Weights,
            'efficientnetv2s': EfficientNet_V2_S_Weights,
            'efficientnetv2m': EfficientNet_V2_M_Weights,
        }[model]
        
        m = {
            'mobilenetv3s': mobilenet_v3_small,
            'mobilenetv3l': mobilenet_v3_large,
            'efficientnetv2s': efficientnet_v2_s,
            'efficientnetv2m': efficientnet_v2_m,
        }[model]
        
        self.model = m(weights=weights.DEFAULT)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 2)

        self.loss = nn.CrossEntropyLoss()

        self.val_precision = MulticlassPrecision(2)
        self.val_recall = MulticlassRecall(2)
        self.val_f1 = MulticlassF1Score(2)

        self.test_precision = MulticlassPrecision(2)
        self.test_recall = MulticlassRecall(2)
        self.test_f1 = MulticlassF1Score(2)
        
        self.save_hyperparameters()

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
        
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)

        self.log('val_precision', self.val_precision)
        self.log('val_recall', self.val_recall)
        self.log('val_f1', self.val_f1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        self.log('test_f1', self.test_f1)

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )
