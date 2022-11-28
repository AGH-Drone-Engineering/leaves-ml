import pytorch_lightning as pl
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torch.optim import Adam
from torchmetrics.detection.mean_ap import MeanAveragePrecision


TRANSFORM = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms()


class MyModelModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ssdlite320_mobilenet_v3_large(num_classes=2)
        self.val_mAP = MeanAveragePrecision()
        self.test_mAP = MeanAveragePrecision()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.val_mAP.update(y_hat, y)

    def validation_epoch_end(self, outputs):
        m = self.val_mAP.compute()
        m = {f'val_{k}': v for k, v in m.items()}
        self.log_dict(m, prog_bar=True)
        self.val_mAP.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.test_mAP.update(y_hat, y)

    def test_epoch_end(self, outputs):
        m = self.test_mAP.compute()
        m = {f'test_{k}': v for k, v in m.items()}
        self.log_dict(m, prog_bar=True)
        self.test_mAP.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
