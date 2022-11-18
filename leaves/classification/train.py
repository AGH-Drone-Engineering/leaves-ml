import lightning as pl

from data.data_wrapper import DataWrapper
from models.model_wrapper import ModelWrapper

from data.beans import BeansDataset
from data.plant_village import PlantVillage

from models.simple_cnn import SimpleCNN


def train():
    data_module = DataWrapper(PlantVillage, BeansDataset, batch_size=32)
    model = SimpleCNN((3, 64, 64))
    model_module = ModelWrapper(model)

    trainer = pl.Trainer(max_epochs=1, val_check_interval=300)

    trainer.fit(model_module, data_module)

    trainer.test(model_module, data_module)


if __name__ == '__main__':
    train()
