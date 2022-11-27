from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
)
import lightning as pl


class LeafData(pl.LightningDataModule):
    def __init__(self, batch_size, model, aug, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.aug = aug
        self.model = model

    def setup(self, stage):
        policy = {
            'none': None,
            'imagenet': transforms.AutoAugmentPolicy.IMAGENET,
            'cifar': transforms.AutoAugmentPolicy.CIFAR10,
            'svhn': transforms.AutoAugmentPolicy.SVHN,
        }[self.aug]
        
        weights = {
            'mobilenetv3s': MobileNet_V3_Small_Weights,
            'mobilenetv3l': MobileNet_V3_Large_Weights,
            'efficientnetv2s': EfficientNet_V2_S_Weights,
            'efficientnetv2m': EfficientNet_V2_M_Weights,
        }[self.model]
        
        train_transform = transforms.Compose([
            transforms.AutoAugment(policy) if policy is not None else lambda x: x,
            weights.DEFAULT.transforms(),
        ])

        test_transform = weights.DEFAULT.transforms()

        target_transform = target_transform=lambda x: 1 - x

        self.train_dataset = ImageFolder('data/raw/plant_village', transform=train_transform, target_transform=target_transform)

        beans = ImageFolder('data/raw/beans', transform=test_transform, target_transform=target_transform)
        self.val_dataset, self.test_dataset = random_split(beans, (0.5, 0.5))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=12)
