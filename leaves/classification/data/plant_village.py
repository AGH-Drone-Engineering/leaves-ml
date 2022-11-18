import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset


class PlantVillage(IterableDataset):
    def __init__(self, split='train', transform=None, target_transform=None):
        self.ds, info = tfds.load(
            'plant_village',
            split=split,
            as_supervised=True,
            shuffle_files=False,
            with_info=True,
        )

        self.healthy = ['healthy' in n for n in info.features['label'].names]

        self.transform = transform
        self.target_transform = target_transform

    def __iter__(self):
        for image, label in tfds.as_numpy(self.ds):
            if self.transform:
                image = self.transform(image)

            label = 0 if self.healthy[label] else 1

            if self.target_transform:
                label = self.target_transform(label)

            yield image, label
