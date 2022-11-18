from torch.utils.data import DataLoader
from torchvision import transforms
import lightning as pl


class DataWrapper(pl.LightningDataModule):
    def __init__(self, train_dataset_fn, test_dataset_fn, batch_size=16):
        super().__init__()
        self.train_dataset_fn = train_dataset_fn
        self.test_dataset_fn = test_dataset_fn
        self.batch_size = batch_size

    def setup(self, stage):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.Resize(64),
            # transforms.ColorJitter(0.3, 0.3, 0.3),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.GaussianBlur(3),
            # transforms.RandomPerspective(0.4, 0.7),
            transforms.Normalize(0.5, 0.5),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.Resize(64),
            transforms.Normalize(0.5, 0.5),
        ])

        
        self.train_dataset = self.train_dataset_fn(split='train[:80%]', transform=train_transform)
        self.val_dataset = self.train_dataset_fn(split='train[80%:]', transform=test_transform)
        self.test_dataset = self.test_dataset_fn(transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    