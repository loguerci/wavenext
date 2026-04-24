"""
Lightning module for WaveNeXt overall architecture using ConvNeXt-based Generator and MPD MRD Discriminators
Author : Loïs Guerci

""" 

import torch
import torch.optim as optim
import yaml

from utils.mel import MelSpectra
from .generator import Generator
from .discriminator import MPD, MRD
from utils.loss import ReconstructionLoss, AdversarialLoss, FeatureMatchingLoss

#from dataloader import AudioDataset
import pytorch_lightning as pl

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class WaveNeXt(pl.LightningModule):
    def __init__(self, sample_rate=24000, fft_dim=1024, shift_dim=256, n_mels=80):
        super().__init__()

        self.sample_rate = sample_rate
        self.fft_dim = fft_dim
        self.shift_dim = shift_dim
        self.n_mels = n_mels

        # Model components
        self.generator = Generator(
            in_channels=self.n_mels,
            fft_dim=self.fft_dim,
            shift_dim=self.shift_dim,
            inter_channels=128,
            num_blocks=8)
        
        self.discriminator_mpd = MPD()
        self.discriminator_mrd = MRD()

        self.mel_extractor = MelSpectra(
            sample_rate=self.sample_rate,
            n_fft=self.fft_dim,
            hop_length=self.shift_dim,
            n_mels=self.n_mels
        )

        # Loss functions
        self.reconstruction_loss = ReconstructionLoss()
        self.adversarial_loss = AdversarialLoss()
        self.feature_matching_loss = FeatureMatchingLoss()

        # Weights for losses
        self.w_mrd = 0.1
        self.w_mel = 45.0

        self.automatic_optimization = False

    def training_step(self, batch):

        x = batch  # (B, 1, T)
        sequence_length = x.size(2)
  
        optimizer_g, optimizer_d = self.optimizers()

        mel = self.mel_extractor(x)
        mel.squeeze_(1)  # (B, n_mels, T)

        fake = self.generator(mel)  # (B, shift_dim * T)
        fake = fake.unsqueeze(1)  # (B, shift_dim * T) -> (B, 1, shift_dim * T)

        # Discriminator step
        optimizer_d.zero_grad()
        
        with torch.no_grad():
            fake = fake[:, :, :sequence_length]  # Ensure fake has the same length as real

        real_fmaps_mpd, real_out_mpd = self.discriminator_mpd(x)
        real_fmaps_mrd, real_out_mrd = self.discriminator_mrd(x)
        fake_fmaps_mpd, fake_out_mpd = self.discriminator_mpd(fake)
        fake_fmaps_mrd, fake_out_mrd = self.discriminator_mrd(fake)

        fake_fmaps_mpd = [[f.detach() for f in fmaps] for fmaps in fake_fmaps_mpd]
        fake_fmaps_mrd = [fmap.detach() for fmap in fake_fmaps_mrd]

        # Compute Losses
        d_loss_mpd = self.adversarial_loss.discriminator_loss(real_out_mpd, fake_out_mpd)
        d_loss_mrd = self.adversarial_loss.discriminator_loss(real_out_mrd, fake_out_mrd)
        total_d_loss = d_loss_mpd + self.w_mrd * d_loss_mrd

        self.log('d_loss', total_d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        total_d_loss.backward()        
        optimizer_d.step()

        # Generator step
        optimizer_g.zero_grad()

        fake = self.generator(mel)  # Regenerate fake for generator step
        fake = fake.unsqueeze(1)  
        fake = fake[:, :, :sequence_length]  

        # Compute Losses

        fake_fmaps_mpd, fake_out_mpd = self.discriminator_mpd(fake)
        fake_fmaps_mrd, fake_out_mrd = self.discriminator_mrd(fake)

        real_fmaps_mpd, _ = self.discriminator_mpd(x)  
        real_fmaps_mrd, _ = self.discriminator_mrd(x)

        g_loss_mpd = self.adversarial_loss.generator_loss(fake_out_mpd)
        g_loss_mrd = self.adversarial_loss.generator_loss(fake_out_mrd)
        g_loss_adv = g_loss_mpd + self.w_mrd * g_loss_mrd
        mel_fake = self.mel_extractor(fake) 
        mel_fake = mel_fake.squeeze(1)  # (B, n_mels, T)

        g_loss_recon = self.reconstruction_loss(mel_fake, mel)

        g_loss_fm_mpd = self.feature_matching_loss(fake_fmaps_mpd, real_fmaps_mpd)
        g_loss_fm_mrd = self.feature_matching_loss(fake_fmaps_mrd, real_fmaps_mrd)
        g_loss_fm = g_loss_fm_mpd + self.w_mrd * g_loss_fm_mrd
        total_g_loss = g_loss_adv + self.w_mel * g_loss_recon + g_loss_fm

        self.log('g_loss', total_g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        total_g_loss.backward()
        optimizer_g.step()
     

    def validation_step(self, batch):
        pass

    def configure_optimizers(self):

        config = load_config('config.yaml')

        optimizer_g = optim.AdamW(self.generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
        optimizer_d = optim.AdamW(list(self.discriminator_mpd.parameters())
                                   + list(self.discriminator_mrd.parameters()), lr=1e-4, betas=(0.9, 0.999))
        
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config['num_epochs'])
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config['num_epochs'])
        

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]