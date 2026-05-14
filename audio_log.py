import os
import torch
import torchaudio
import pytorch_lightning as pl
from utils.mel import MelSpectra
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class audio_log(pl.Callback):
    def __init__(self, dataset, every_n_epochs=20, num_samples=4, sample_rate=24000):
        super().__init__()
        self.config = load_config('config_48k.yaml')
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dataset = dataset
        self.sample_rate = sample_rate

        self.mel_extractor = MelSpectra(sample_rate=self.config['sample_rate'], n_fft=self.config['fft_dim'], hop_length=self.config['shift_dim'], n_mels=self.config['n_mels'])

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            return
        
        path_dir = os.path.join(trainer.logger.log_dir, f'audio_epoch_{epoch}')
        os.makedirs(path_dir, exist_ok=True)

        pl_module.eval()
        with torch.no_grad():
            for i in range(self.num_samples):
                mel = self.mel_extractor(self.dataset[i]).to(pl_module.device)
                mel.squeeze_(1)
                fake = pl_module.generator(mel)
                fake = fake[..., :self.dataset[i].size(-1)]
                
                torchaudio.save(
                    os.path.join(path_dir, f'sample_{i}_fake.wav'),
                    fake.cpu(),
                    self.sample_rate
                )
                torchaudio.save(
                    os.path.join(path_dir, f'sample_{i}_real.wav'),
                    self.dataset[i].cpu(),
                    self.sample_rate
                )
        pl_module.train()