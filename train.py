"""
Training script for WaveNeXt model
Author : Loïs Guerci

"""


from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


from models.wavenext import WaveNeXt
from torch.utils.data import DataLoader
from dataloader import WaveNeXtDataset

import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(hparams):
    torch.set_float32_matmul_precision('high')
    config = load_config(hparams.config_path)

    model = WaveNeXt(
        sample_rate=config['sample_rate'],
        dim=config['dim'],
        shift_dim=config['shift_dim'],
        n_mels=config['n_mels'],
        k=config['k']
    )

    dataset = WaveNeXtDataset(path_csv=config['train_csv'], sample_rate=config['sample_rate'], duration=config['duration'])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    checkpoint_callback = ModelCheckpoint(
        monitor='g_loss',
        dirpath='checkpoints',
        filename='wavenext-{epoch:02d}-{g_loss:.2f}',
        save_top_k=3,
        mode='min',
        every_n_epochs=1
    )

    logger = TensorBoardLogger(save_dir=config['log_dir'], name='wavenext')
    trainer = Trainer(accelerator=config['accelerator'], 
                      devices=config['devices'], 
                      max_epochs=config['num_epochs'], 
                      logger=logger,
                      strategy="ddp_find_unused_parameters_true",
                      callbacks=[ModelSummary(max_depth=2), checkpoint_callback])

    trainer.fit(model, train_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config file')
    hparams = parser.parse_args()

    main(hparams)