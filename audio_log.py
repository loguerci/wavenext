import os
import torch
import torchaudio
import pytorch_lightning as pl
from utils.mel import MelSpectra
import yaml
from encodec import EncodecModel
from encodec.utils import convert_audio

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class audio_log(pl.Callback):
    def __init__(self, dataset, every_n_epochs=20, num_samples=4, sample_rate=24000, prior="encodec"):
        super().__init__()
        self.config = load_config('config_48k.yaml')
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dataset = dataset
        self.prior = prior
        self.sample_rate = sample_rate
        
        if self.prior == "encodec":
            self.encoder = EncodecModel.encodec_model_24khz()
            self.encoder.set_target_bandwidth(6.0)
        if self.prior == "same":
            import sys
            sys.path.append('../stable-audio-3')
            from stable_audio_3 import AutoencoderModel
            self.encoder = AutoencoderModel.from_pretrained("same-s")

        self.mel_extractor = MelSpectra(sample_rate=self.config['sample_rate'], n_fft=self.config['fft_dim'], hop_length=self.config['shift_dim'], n_mels=self.config['n_mels'])

    def on_validation_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            return
        path_dir = os.path.join(trainer.logger.log_dir, f'audio_epoch_{epoch}')
        os.makedirs(path_dir, exist_ok=True)

        pl_module.eval()
        with torch.no_grad():
            for i in range(self.num_samples):
                encoded_frames = self.encoder.encode(self.dataset[i].unsqueeze(0))
                codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
                emb = self.encoder.quantizer.decode(codes.transpose(0, 1))
                fake = pl_module.decoder(emb.to(pl_module.device))
                
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