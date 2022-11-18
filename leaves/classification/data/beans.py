from datasets import load_dataset
from torch.utils.data import Dataset


class BeansDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        super().__init__()

        self.dataset = load_dataset('beans', split='train')

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']

        label = 0 if label == 2 else 1

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
