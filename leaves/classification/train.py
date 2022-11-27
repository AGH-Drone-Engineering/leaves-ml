import argparse
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from data.leaf_data import LeafData
from models.leaf_model import LeafModel


def train(config):
    data_module = LeafData(**config)
    model_module = LeafModel(**config)

    wandb_logger = WandbLogger(
        project='leaves',
        log_model='all',
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        precision=16,
        callbacks=[
            EarlyStopping(monitor="val_f1", mode="max", patience=4),
            ModelCheckpoint(monitor="val_f1", mode="max"),
        ],
        # min_epochs=1,
        max_epochs=-1,
        max_time='00:00:30:00',
        check_val_every_n_epoch=None,
        val_check_interval=config['val_steps'],
        log_every_n_steps=config['log_steps'],
        accumulate_grad_batches=config['gradient_accumulation'],
        gradient_clip_val=1,

    )

    trainer.fit(model=model_module, datamodule=data_module)
    trainer.test(model=model_module, datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gradient_accumulation', type=int)
    
    parser.add_argument('--log_steps', type=int)
    parser.add_argument('--val_steps', type=int)

    parser.add_argument('--model', type=str, choices=[
        'mobilenetv3s',
        'mobilenetv3l',
        'efficientnetv2s',
        'efficientnetv2m',
    ])

    parser.add_argument('--aug', type=str, choices=[
        'none',
        'imagenet',
        'cifar',
        'svhn',
    ])

    args = parser.parse_args()
    train(vars(args))

