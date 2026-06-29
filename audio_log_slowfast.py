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

class audio_log_slowfast(pl.Callback):
    def __init__(self, dataset, every_n_epochs=20, num_samples=4, sample_rate=24000, prior="encodec"):
        super().__init__()
        self.config = load_config('config_slowfast_24k.yaml')
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dataset = dataset
        self.prior = prior
        self.sample_rate = sample_rate
        
        if self.prior == "encodec":
            self.encoder = EncodecModel.encodec_model_24khz()
            self.encoder.set_target_bandwidth(6.0)

        self.mel_extractor = MelSpectra(sample_rate=self.config['sample_rate'], 
                                        n_fft=self.config['fft_dim'], 
                                        hop_length=256, 
                                        n_mels=self.config['n_mels'])

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch % self.every_n_epochs != 0:
            return
        path_dir = os.path.join(trainer.logger.log_dir, f'audio_epoch_{epoch}')
        os.makedirs(path_dir, exist_ok=True)

        pl_module.eval()
        with torch.no_grad():
            for i in range(self.num_samples):
                x = self.dataset[i]
                split = int(x.size(1)*(2/3))
                x = x.unsqueeze(1)
                x_fast = x[:, :, split:]
                x_slow = x[:, :, :split]

                #print(x_fast.shape)
                #print(x_slow.shape)

                encoded_frames = self.encoder.encode(x_fast)
                codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
                fast_chunk = self.encoder.quantizer.decode(codes.transpose(0, 1))

                slow_chunk = self.mel_extractor(x_slow).squeeze(1)

                #print(fast_chunk.shape)
                #print(slow_chunk.shape)

                slow_chunk = slow_chunk.transpose(1,2)
                a, b = pl_module.slow_branch(slow_chunk.to(pl_module.device))
                fake = pl_module.decoder(fast_chunk.to(pl_module.device), (a.to(pl_module.device), b.to(pl_module.device)))
                #print(fake.shape)

                x_fast = x_fast.squeeze(1)
                
                torchaudio.save(
                    os.path.join(path_dir, f'sample_{i}_fake.wav'),
                    fake.cpu(),
                    self.sample_rate
                )
                torchaudio.save(
                    os.path.join(path_dir, f'sample_{i}_real.wav'),
                    x_fast.cpu(),
                    self.sample_rate
                )
        pl_module.train()