"""
Lightning module for WaveNeXt overall architecture using ConvNeXt-based Generator and MPD MRD Discriminators
Author : Loïs Guerci

""" 

from pyexpat import model

import torch
import torch.optim as optim
import yaml

from utils.mel import MelSpectra
from .decoder import CrossA_Decoder
from .discriminator import MPD, MRD
from utils.loss import ReconstructionLoss, AdversarialLoss, FeatureMatchingLoss
from encodec import EncodecModel
from .slow_branch import Transformer_SlowBranch
import sys

import pytorch_lightning as pl

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class WaveNeXtSlowFast(pl.LightningModule):
    def __init__(self, dim: int, sample_rate: int, fft_dim: int, shift_dim: int, n_mels: int, k: int, lr_g: float, lr_d: float, prior: str, slow_size: int, r_factor: int, delay: int):
        super().__init__()

        self.sample_rate = sample_rate
        self.fft_dim = fft_dim
        self.shift_dim = shift_dim
        self.n_mels = n_mels
        self.k = k
        self.dim = dim
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.prior = prior

        self.slow_size = slow_size
        self.r_factor = r_factor # reuse factor 
        self.delay = delay # delay steps of slow windows

        self.K_init = torch.nn.Parameter(torch.zeros(self.slow_size, self.dim))
        self.V_init = torch.nn.Parameter(torch.zeros(self.slow_size, self.dim))

        # Model components
        self.decoder = CrossA_Decoder(
            in_channels=self.n_mels,
            dim=self.dim,
            shift_dim=self.shift_dim,
            inter_channels=self.dim * self.k,
            num_blocks=8)
        
        #self.slow_branch = Naive_SlowBranch(latent_dim=128, film_dim=self.dim)
        self.slow_branch = Transformer_SlowBranch(latent_dim=slow_size, dim=self.dim)

        self.discriminator_mpd = MPD()
        self.discriminator_mrd = MRD()

        if self.prior == "encodec":
            self.encoder = EncodecModel.encodec_model_24khz()
            self.encoder.set_target_bandwidth(6.0)

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
        conds = [(self.K_init, self.V_init)] * N
        windows = []
        j = self.slow_size
  
        optimizer_g, optimizer_d = self.optimizers()

        if self.prior == "encodec":
            with torch.no_grad():
                encoded_frames = self.encoder.encode(x)
                codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
                emb = self.encoder.quantizer.decode(codes.transpose(0, 1)) # (B, D, N)

        N = emb.size(2)
        fast_chunks = list(emb.split(1, dim=2))

        # defining slow windows and the conditioned fast chunks for the current input
        while j + self.delay + self.r_factor - 1 < N:
            slow_window = (j-self.slow_size, j)
            cond_chunks = (j + self.delay, j + self.delay + self.r_factor)
            windows.append((slow_window, cond_chunks))
            j += self.r_factor

        # processing all slow chunks and storing conditioning outputs
        for slow_start, slow_end, cond_start, cond_end in windows:
            slow_chunk = emb[:, :, slow_start:slow_end].transpose(1, 2)
            cond = self.slow_branch(slow_chunk)
            for t in range(cond_start, cond_end):
                conds[t] = cond
        
        # processing all fast chunks with their corresponding conditioning
        fake_chunks = [self.decoder(fast_chunks[t], conds[t]) for t in range(N)]   
        fake = torch.cat(fake_chunks, dim=-1).unsqueeze(1)  # (B, 1, T)
        fake = fake[:, :, :sequence_length]

        # Discriminator step
        optimizer_d.zero_grad()

        fake = fake[:, :, :sequence_length]  # Ensure fake has the same length as real
        fake_detached = fake.detach()  # Detach fake from the generator graph for discriminator update

        real_fmaps_mpd, real_out_mpd = self.discriminator_mpd(x)
        real_fmaps_mrd, real_out_mrd = self.discriminator_mrd(x)
        fake_fmaps_mpd, fake_out_mpd = self.discriminator_mpd(fake_detached)
        fake_fmaps_mrd, fake_out_mrd = self.discriminator_mrd(fake_detached)

        # Compute Losses
        d_loss_mpd = self.adversarial_loss.discriminator_loss(real_out_mpd, fake_out_mpd)
        d_loss_mrd = self.adversarial_loss.discriminator_loss(real_out_mrd, fake_out_mrd)
        total_d_loss = d_loss_mpd + self.w_mrd * d_loss_mrd

        self.log('d_loss', total_d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        total_d_loss.backward()        
        optimizer_d.step()

        # Generator step
        optimizer_g.zero_grad()

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

        g_loss_recon = self.reconstruction_loss(mel_fake, self.mel_extractor(x).squeeze(1))

        g_loss_fm_mpd = self.feature_matching_loss(fake_fmaps_mpd, real_fmaps_mpd)
        g_loss_fm_mrd = self.feature_matching_loss(fake_fmaps_mrd, real_fmaps_mrd)
        g_loss_fm = g_loss_fm_mpd + self.w_mrd * g_loss_fm_mrd
        total_g_loss = g_loss_adv + self.w_mel * g_loss_recon + g_loss_fm

        self.log('g_loss', total_g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        total_g_loss.backward()
        optimizer_g.step()
     

    def validation_step(self, batch):
        x = batch
        conds = [(self.K_init, self.V_init)] * N
        windows = []
        j = self.slow_size

        with torch.no_grad():
            encoded_frames = self.encoder.encode(x)
            codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
            emb = self.encoder.quantizer.decode(codes.transpose(0, 1))

            N = emb.size(2)
            fast_chunks = list(emb.split(1, dim=2))

            # defining slow windows and the conditioned fast chunks for the current input
            while j + self.delay + self.r_factor - 1 < N:
                slow_window = (j-self.slow_size, j)
                cond_chunks = (j + self.delay, j + self.delay + self.r_factor)
                windows.append((slow_window, cond_chunks))
                j += self.r_factor

            # processing all slow chunks and storing conditioning outputs
            for slow_start, slow_end, cond_start, cond_end in windows:
                slow_chunk = emb[:, :, slow_start:slow_end].transpose(1, 2)
                cond = self.slow_branch(slow_chunk)
                for t in range(cond_start, cond_end):
                    conds[t] = cond
            
            # processing all fast chunks with their corresponding conditioning
            fake_chunks = [self.decoder(fast_chunks[t], conds[t]) for t in range(N)]   
            fake = torch.cat(fake_chunks, dim=-1).unsqueeze(1)  # (B, 1, T)
            fake = fake[:, :, :x.size(2)]

        # Just log some metrics, no backward
        fake_mel = self.mel_extractor(fake)
        fake_mel.squeeze_(1)
        mel_loss = self.reconstruction_loss(fake_mel, self.mel_extractor(x).squeeze(1))
        self.log('val_mel_loss', mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch):
        pass

    def configure_optimizers(self):

        config = load_config('config_48k.yaml')

        optimizer_g = optim.AdamW(list(self.decoder.parameters()) + list(self.slow_branch.parameters()) 
                                  + [self.K_init, self.V_init], lr=self.lr_g, betas=(0.9, 0.999))
        optimizer_d = optim.AdamW(list(self.discriminator_mpd.parameters())
                                   + list(self.discriminator_mrd.parameters()), lr=self.lr_d, betas=(0.9, 0.999))
        
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config['num_epochs'])
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config['num_epochs'])     

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]