"""
Training script for WaveNeXt model
Author : Loïs Guerci

"""


from argparse import ArgumentParser
import os
from datetime import datetime, date

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


from models.wavenext import WaveNeXt
from torch.utils.data import DataLoader
from dataloader import WaveNeXtDataset
from audio_log import audio_log

import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(hparams):

    now = datetime.now() 
    formatted = now.strftime("%d-%m_at_%H_%M_%S")

    torch.set_float32_matmul_precision('high')
    config = load_config(hparams.config_path)

    model = WaveNeXt(
        sample_rate=config['sample_rate'],
        dim=config['dim'],
        shift_dim=config['shift_dim'],
        n_mels=config['n_mels'],
        k=config['k'],
        lr=config['learning_rate']
    )

    train_dataset = WaveNeXtDataset(path_csv=config['train_csv'], sample_rate=config['sample_rate'], duration=config['duration'])
    val_dataset = WaveNeXtDataset(path_csv=config['val_csv'], sample_rate=config['sample_rate'], duration=config['duration'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mel_loss',
        dirpath=f'checkpoints/{formatted}',
        filename='wavenext-{epoch:02d}-{val_mel_loss:.2f}',
        save_top_k=3,
        mode='min',
        every_n_epochs=1
    )

    logger = TensorBoardLogger(save_dir=config['log_dir'] + f'/{formatted}', name='wavenext')

    audio = audio_log(dataset=val_dataset, every_n_epochs=20, num_samples=4, sample_rate=config['sample_rate'])

    trainer = Trainer(accelerator=config['accelerator'], 
                      devices=config['devices'], 
                      max_epochs=config['num_epochs'], 
                      logger=logger,
                      callbacks=[ModelSummary(max_depth=2), checkpoint_callback, audio])
    
    resume_ckpt = '/home/lois/wavenext/checkpoints/04-05_at_13_25_19/wavenext-epoch=312-val_mel_loss=2.84.ckpt'

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt if config['resume'] else None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to config file')
    hparams = parser.parse_args()

    main(hparams)