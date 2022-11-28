import pytorch_lightning as pl

from my_model_module import MyModelModule
from my_data_module import MyDataModule


def train():
    model = MyModelModule()
    data = MyDataModule()
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        val_check_interval=0.25,
    )
    
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == '__main__':
    train()
