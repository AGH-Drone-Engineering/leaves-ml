import pytorch_lightning as pl
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torch.optim import Adam


TRANSFORM = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms()


class MyModelModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ssdlite320_mobilenet_v3_large()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
