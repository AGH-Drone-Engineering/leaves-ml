import torch
import pytorch_lightning as pl
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

from my_model_module import TRANSFORM


def bbox_transform(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def target_transform(target):
    return {
        'boxes': torch.tensor([bbox_transform(t['bbox']) for t in target]).reshape(-1, 4),
        'labels': torch.tensor([t['category_id'] for t in target]).long(),
        'image_id': torch.tensor([t['image_id'] for t in target]),
        'area': torch.tensor([t['area'] for t in target]),
        'iscrowd': torch.tensor([t['iscrowd'] for t in target]),
    }


def collate_fn(batch):
    x = [t[0] for t in batch]
    y = [t[1] for t in batch]
    return x, y


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = CocoDetection(
            root='dataset/train',
            annFile='dataset/train/_annotations.coco.json',
            transform=TRANSFORM,
            target_transform=target_transform,
        )

        self.val_dataset = CocoDetection(
            root='dataset/valid',
            annFile='dataset/valid/_annotations.coco.json',
            transform=TRANSFORM,
            target_transform=target_transform,
        )

        self.test_dataset = CocoDetection(
            root='dataset/test',
            annFile='dataset/test/_annotations.coco.json',
            transform=TRANSFORM,
            target_transform=target_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=4, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=4)
