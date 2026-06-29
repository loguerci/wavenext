"""
Lightning module for WaveNeXt overall architecture using ConvNeXt-based Generator and MPD MRD Discriminators
Author : Loïs Guerci

""" 

from pyexpat import model

import torch
import torch.optim as optim
import yaml

from utils.mel import MelSpectra
from .decoder import CrossA_Decoder, Cond_Decoder
from .generator import Generator
from .discriminator import MPD, MRD, MSSTFTDiscriminator
from utils.loss import ReconstructionLoss, AdversarialLoss, FeatureMatchingLoss
from encodec import EncodecModel
from .slow_branch import Transformer_SlowBranch, GRU_SlowBranch

import pytorch_lightning as pl

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class WaveNeXtSlowFast(pl.LightningModule):
    def __init__(self, dim: int, 
                 sample_rate: int, 
                 fft_dim: int, 
                 shift_dim: int, 
                 n_mels: int, 
                 k: int, 
                 lr_g: float, 
                 lr_d: float, 
                 prior: str, 
                 s_branch: str):
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

        # Model components
        if s_branch == 'transformer':
            self.decoder = CrossA_Decoder(
                in_channels=self.n_mels,
                dim=self.dim,
                shift_dim=self.shift_dim,
                inter_channels=self.dim * self.k,
                num_blocks=4)
            
            self.slow_branch = Transformer_SlowBranch(latent_dim=self.n_mels, dim=self.dim)
        elif s_branch == 'gru':
            self.decoder = Cond_Decoder(
                in_channels=self.n_mels,
                dim=self.dim,
                shift_dim=self.shift_dim,
                inter_channels=self.dim * self.k,
                num_blocks=8) 

            self.slow_branch = GRU_SlowBranch(latent_dim=self.n_mels, film_dim=self.dim)

        self.generator = Generator(
            in_channels=self.n_mels,
            dim=self.dim,
            shift_dim=256,
            inter_channels=self.dim * self.k,
            num_blocks=8)

        self.discriminator_mpd = MPD()
        self.discriminator_mrd = MRD()
        self.discriminator_msstft = MSSTFTDiscriminator()

        if self.prior == "encodec":
            self.encoder = EncodecModel.encodec_model_24khz()
            self.encoder.set_target_bandwidth(6.0)

        self.mel_extractor = MelSpectra(
            sample_rate=self.sample_rate,
            n_fft=self.fft_dim,
            hop_length=256,
            n_mels=self.n_mels
        )

        # Loss functions
        self.reconstruction_loss = ReconstructionLoss()
        self.adversarial_loss = AdversarialLoss()
        self.feature_matching_loss = FeatureMatchingLoss()

        # Weights for losses
        self.w_mrd = 0.1
        self.w_mel = 45.0
        self.w_msstft = 0.25

        self.automatic_optimization = False
        self.warmup_epochs = 20

    def training_step(self, batch):

        x = batch  # (B, 1, T)
        split = int(x.size(2)*(2/3))
        x_fast = x[:, :, split:]
        x_slow = x[:, :, :split]
  
        optimizer_g, optimizer_d = self.optimizers()

        if self.prior == "encodec":
            with torch.no_grad():
                encoded_frames = self.encoder.encode(x_fast)
                codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
                emb = self.encoder.quantizer.decode(codes.transpose(0, 1)) # (B, D, N)

        slow_chunk = self.mel_extractor(x_slow).squeeze(1)
        fast_chunk = emb

        cond = self.slow_branch(slow_chunk.transpose(1,2))
        fake = self.decoder(fast_chunk, cond)
        fake = fake.unsqueeze(1) 
        fake = fake[:, :, :x_fast.size(2)]

        fake_detached = fake.detach() 

        mel_fake = self.mel_extractor(fake).squeeze(1)
        g_loss_recon = self.reconstruction_loss(mel_fake, self.mel_extractor(x_fast).squeeze(1))

        # Discriminator step
        if self.current_epoch >= self.warmup_epochs:
            optimizer_d.zero_grad()

            real_fmaps_mpd, real_out_mpd = self.discriminator_mpd(x_fast)
            real_fmaps_mrd, real_out_mrd = self.discriminator_mrd(x_fast)
            real_fmaps_msstft, real_out_msstft = self.discriminator_msstft(x_fast)
            _, fake_out_mpd = self.discriminator_mpd(fake_detached)
            _, fake_out_mrd = self.discriminator_mrd(fake_detached)
            _, fake_out_msstft = self.discriminator_msstft(fake_detached)

            d_loss_mpd = self.adversarial_loss.discriminator_loss(real_out_mpd, fake_out_mpd)
            d_loss_mrd = self.adversarial_loss.discriminator_loss(real_out_mrd, fake_out_mrd)
            d_loss_msstft = self.adversarial_loss.discriminator_loss(real_out_msstft, fake_out_msstft)
            total_d_loss = d_loss_mpd + self.w_mrd * d_loss_mrd + self.w_msstft * d_loss_msstft

            self.log('d_loss', total_d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            total_d_loss.backward()
            optimizer_d.step()

        # Generator step
        optimizer_g.zero_grad()

        # Compute Losses

        if self.current_epoch >= self.warmup_epochs:
            fake_fmaps_mpd, fake_out_mpd = self.discriminator_mpd(fake)
            fake_fmaps_mrd, fake_out_mrd = self.discriminator_mrd(fake)
            fake_fmaps_msstft, fake_out_msstft = self.discriminator_msstft(fake)
            real_fmaps_mpd, _ = self.discriminator_mpd(x_fast)
            real_fmaps_mrd, _ = self.discriminator_mrd(x_fast)
            real_fmaps_msstft, _ = self.discriminator_msstft(x_fast)

            g_loss_mpd = self.adversarial_loss.generator_loss(fake_out_mpd)
            g_loss_mrd = self.adversarial_loss.generator_loss(fake_out_mrd)
            g_loss_msstft = self.adversarial_loss.generator_loss(fake_out_msstft)
            g_loss_adv = g_loss_mpd + self.w_mrd * g_loss_mrd + self.w_msstft * g_loss_msstft

            g_loss_fm_mpd = self.feature_matching_loss(fake_fmaps_mpd, real_fmaps_mpd)
            g_loss_fm_mrd = self.feature_matching_loss(fake_fmaps_mrd, real_fmaps_mrd)
            g_loss_fm_msstft = self.feature_matching_loss(fake_fmaps_msstft, real_fmaps_msstft)
            g_loss_fm = g_loss_fm_mpd + self.w_mrd * g_loss_fm_mrd + self.w_msstft * g_loss_fm_msstft

            total_g_loss = g_loss_adv + self.w_mel * g_loss_recon + g_loss_fm

        else:
            total_g_loss = self.w_mel * g_loss_recon

        self.log('g_loss', total_g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        total_g_loss.backward()
        optimizer_g.step()
     

    def validation_step(self, batch):
        x = batch
        split = int(x.size(2)*(2/3))
        x_fast = x[:, :, split:]
        x_slow = x[:, :, :split]

        with torch.no_grad():
            encoded_frames = self.encoder.encode(x_fast)
            codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
            emb = self.encoder.quantizer.decode(codes.transpose(0, 1)) # (B, D, N)

            fast_chunk = emb
            slow_chunk = self.mel_extractor(x_slow).squeeze(1)

            cond = self.slow_branch(slow_chunk.transpose(1,2))
            fake = self.decoder(fast_chunk, cond)
            fake = fake.unsqueeze(1) 
            fake = fake[:, :, :x_fast.size(2)]

        # Just log some metrics, no backward
        mel_fake = self.mel_extractor(fake).squeeze(1)
        mel_loss = self.reconstruction_loss(mel_fake, self.mel_extractor(x_fast).squeeze(1))
        self.log('val_mel_loss', mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch):
        pass

    def configure_optimizers(self):

        config = load_config('config_slowfast_24k.yaml')

        optimizer_g = optim.AdamW(list(self.decoder.parameters()) + list(self.slow_branch.parameters()), lr=self.lr_g, betas=(0.9, 0.999))
        optimizer_d = optim.AdamW(list(self.discriminator_mpd.parameters())
                                   + list(self.discriminator_mrd.parameters())
                                   + list(self.discriminator_msstft.parameters()), lr=self.lr_d, betas=(0.9, 0.999))
        
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config['num_epochs'])
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config['num_epochs'])     

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]