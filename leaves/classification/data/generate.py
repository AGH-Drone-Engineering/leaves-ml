import os
import shutil
import tensorflow_datasets as tfds
from torchvision.transforms import ToPILImage
from datasets import load_dataset
from tqdm import tqdm


def main():
    plants, info = tfds.load(
        'plant_village',
        split='train',
        as_supervised=True,
        shuffle_files=False,
        with_info=True,
    )

    healthy = ['healthy' in n for n in info.features['label'].names]

    image_transform = ToPILImage()

    beans = load_dataset('beans')

    shutil.rmtree('raw', ignore_errors=True)

    os.makedirs('raw/plant_village/healthy')
    os.makedirs('raw/plant_village/diseased')

    os.makedirs('raw/beans/healthy')
    os.makedirs('raw/beans/diseased')

    for i, (image, label) in enumerate(tqdm(tfds.as_numpy(plants))):
        image = image_transform(image)
        label = 'healthy' if healthy[label] else 'diseased'
        image.save(f'raw/plant_village/{label}/{i}.jpg')

    for i, item in enumerate(tqdm(beans['train'])):
        image = item['image']
        label = 'healthy' if item['labels'] == 2 else 'diseased'
        image.save(f'raw/beans/{label}/{i * 2}.jpg')

    for i, item in enumerate(tqdm(beans['test'])):
        image = item['image']
        label = 'healthy' if item['labels'] == 2 else 'diseased'
        image.save(f'raw/beans/{label}/{i * 2 + 1}.jpg')


if __name__ == '__main__':
    main()
