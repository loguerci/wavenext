import os
import torch
import torchaudio
import pytorch_lightning as pl
from utils.mel import MelSpectra


class audio_log(pl.Callback):
    def __init__(self, dataset, every_n_epochs=20, num_samples=4, sample_rate=24000):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self.dataset = dataset
        self.sample_rate = sample_rate

        self.mel_extractor = MelSpectra(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80)

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