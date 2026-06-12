"""
Training script for WaveNeXt model
Author : Loïs Guerci

"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
import torch

_load_kept = torch.load
def _trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _load_kept(*args, **kwargs)
torch.load = _trusted_load

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split

from models.wavenext import WaveNeXt
from models.wavenext_prior import WaveNeXtLatent
from torch.utils.data import DataLoader
from dataloader import WaveNeXtDataset
from audio_log import audio_log

import yaml



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(hparams):

    now = datetime.now() 
    formatted = now.strftime("%d-%m_at_%H_%M_%S")

    torch.set_float32_matmul_precision('high')
    config = load_config(hparams.config_path)

    model = WaveNeXtLatent(
        dim=config['dim'],
        sample_rate=config['sample_rate'],
        fft_dim=config['fft_dim'],
        shift_dim=config['shift_dim'],
        n_mels=config['n_mels'],
        k=config['k'],
        lr_g=config['learning_rate_g'],
        lr_d=config['learning_rate_d'],
        prior=config['prior']
    )

    dataset = WaveNeXtDataset(path_csv=config['dataset'], sample_rate=config['sample_rate'], samples=config['samples'])
    train_size = int(0.8 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size


    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
        )

    print(f"Number of train samples : {len(train_dataset)}")
    print(f"Number of validation samples : {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    if model.fd:
        print("Precomputing real stats for FD loss...")
        model.precompute_real_stats(train_loader)
        print("Real stats precomputed.")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_mel_loss',
        dirpath=f'checkpoints/{formatted}',
        filename='wavenext-{epoch:02d}-{val_mel_loss:.3f}',
        save_top_k=1,
        mode='min',
        every_n_epochs=1
    )

    logger = TensorBoardLogger(save_dir=config['log_dir'] + f'/{formatted}', name='wavenext')

    audio = audio_log(dataset=val_dataset, every_n_epochs=20, num_samples=4, sample_rate=config['sample_rate'], prior=config['prior'])

    trainer = Trainer(accelerator=config['accelerator'], 
                      devices=config['devices'], 
                      max_epochs=config['num_epochs'], 
                      logger=logger,
                      callbacks=[ModelSummary(max_depth=2), checkpoint_callback, audio])
    
    resume_ckpt = '/home/lois/wavenext/checkpoints/29-05_at_02_45_21/wavenext-epoch=03-val_mel_loss=2.246.ckpt'

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt if config['resume'] else None)

if __name__ == "__main__":
    # python train.py --config_path config_encodec_24.yaml
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config_48k.yaml', help='Path to config file')
    hparams = parser.parse_args()

    main(hparams)