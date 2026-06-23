"""
Lightning module for WaveNeXt overall architecture using ConvNeXt-based Generator and MPD MRD Discriminators
Author : Loïs Guerci

""" 
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import sys

from utils.mel import MelSpectra
from utils.fadtk_emb import FADTKEmbedding
from .decoder import Decoder
from .discriminator import MPD, MRD, MSSTFTDiscriminator
from utils.loss import ReconstructionLoss, AdversarialLoss, FeatureMatchingLoss, FDLoss
from encodec import EncodecModel
from fadtk.model_loader import (
    CLAPLaionModel,
    CLAPModel,
    CdpamModel,
    DACModel
)

import pytorch_lightning as pl

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class WaveNeXtLatent(pl.LightningModule):
    def __init__(self, dim: int, sample_rate: int, fft_dim: int, shift_dim: int, n_mels: int, k: int, lr_g: float, lr_d: float, prior: str):
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
        self.decoder = Decoder(
            in_channels=self.n_mels,
            dim=self.dim,
            shift_dim=self.shift_dim,
            inter_channels=self.dim * self.k,
            num_blocks=8)
        
        self.discriminator_mpd = MPD()
        self.discriminator_mrd = MRD()
        self.discriminator_msstft = MSSTFTDiscriminator()

        if self.prior == "encodec":
            self.encoder = EncodecModel.encodec_model_24khz()
            self.encoder.set_target_bandwidth(6.0)
        elif self.prior == "same":
            sys.path.append('../stable-audio-3')
            from stable_audio_3 import AutoencoderModel
            self.encoder = AutoencoderModel.from_pretrained("same-s")

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
        self.w_msstft = 0.5

        self.automatic_optimization = False

        self.warmup_epochs = 20

        # for FD loss computation
        self.fd = False # Set to True to enable Fréchet Distance loss
        self.fd_weight = 1.0
        self.fd_norm_c = 0.01
        self.fd_loss_balance_beta = 0.99
        self.fd_loss_balance_eps = 1e-6
        self.fd_ema_decay = 0.999
        self.fd_warm_batches = 16

        self.fd_loss = FDLoss()
        self.feat_dim = 4*512

        if self.fd:
            self._build_fd_embedders(self.fd_ema_decay)

    def _build_fd_embedders(self, ema_decay: float):
        """Instantiate differentiable embedders and per-embedder buffers."""
        loaders = [
            CLAPModel("2023"),
            CLAPLaionModel("audio"),
            CdpamModel("acoustic"),
            DACModel(),
        ]
        self.embedders = nn.ModuleList([FADTKEmbedding(l) for l in loaders])
        self.embedder_feat_dims = [e.num_features for e in self.embedders]

        for i, feat_dim in enumerate(self.embedder_feat_dims):
            # target (real) stats — filled by precompute_real_stats
            self.register_buffer(f"mu_r_{i}",       torch.zeros(feat_dim))
            self.register_buffer(f"sig_r_{i}",      torch.zeros(feat_dim, feat_dim))
            self.register_buffer(f"sig_r_sqrt_{i}", torch.zeros(feat_dim, feat_dim))
            # EMA generated stats
            self.register_buffer(f"ema_mu_{i}",     torch.zeros(feat_dim))
            self.register_buffer(f"ema_M_{i}",      torch.zeros(feat_dim, feat_dim))
            self.register_buffer(f"ema_count_{i}",  torch.zeros((), dtype=torch.long))
            # loss scale for balancing
            self.register_buffer(f"loss_scale_{i}", torch.ones(()))

        self.ema_decay = ema_decay

    @torch.no_grad()
    def precompute_real_stats(self, real_dataloader, n_batches: int = 64, cache_path: str = "fd_stats.pt"):
        """
        Compute per-embedder (mu_r, sig_r, sig_r_sqrt) from real audio.
        Call this before trainer.fit().
        """
        if os.path.exists(cache_path):
            print(f"Loading cached FD stats from {cache_path}")
            stats = torch.load(cache_path, weights_only=True)
            for idx in range(len(self.embedders)):
                getattr(self, f"mu_r_{idx}").copy_(stats[f"mu_r_{idx}"])
                getattr(self, f"sig_r_{idx}").copy_(stats[f"sig_r_{idx}"])
                getattr(self, f"sig_r_sqrt_{idx}").copy_(stats[f"sig_r_sqrt_{idx}"])
            print("FD stats loaded from cache.")
            return
        
        print("Precomputing real audio FD statistics...")
        n_embedders = len(self.embedders)
        sums   = [None] * n_embedders
        sums_xx = [None] * n_embedders
        counts = [0]    * n_embedders

        self.eval()
        for i, batch in enumerate(real_dataloader):
            if i >= n_batches:
                break
            x = batch.to(self.device)   # (B, 1, T)
            audio = x.squeeze(1)        # (B, T)

            for idx, embedder in enumerate(self.embedders):
                feat = embedder(audio, self.sample_rate).float()   # (N, D)
                sums[idx]    = feat.sum(0)          if sums[idx]   is None else sums[idx]   + feat.sum(0)
                sums_xx[idx] = feat.T @ feat        if sums_xx[idx] is None else sums_xx[idx] + feat.T @ feat
                counts[idx] += feat.shape[0]

        for idx in range(n_embedders):
            n = counts[idx]
            mu     = sums[idx] / n
            second = sums_xx[idx] / n
            cov    = 0.5 * ((second - torch.outer(mu, mu)) + (second - torch.outer(mu, mu)).T)

            getattr(self, f"mu_r_{idx}").copy_(mu)
            getattr(self, f"sig_r_{idx}").copy_(cov)
            getattr(self, f"sig_r_sqrt_{idx}").copy_(FDLoss.compute_sqrt_cov(cov))

        self.train()

    def _fd_loss_all_embedders(self, fake_audio):
        """
        Compute normalized FD loss for each embedder separately,
        then return their weighted (balanced) sum — matching the
        per-representation approach from the reference.
        """
        total_fd_loss = fake_audio.new_tensor(0.0)
        logs = {}

        for idx, embedder in enumerate(self.embedders):
            feat = embedder(fake_audio, self.sample_rate).float()  # (N, D)

            # Multi-GPU: gather features across devices before EMA update
            if self.trainer.world_size > 1:
                feat = self.all_gather(feat).reshape(-1, feat.shape[-1])

            # --- EMA moment update (grad flows only through current batch) ---
            B = feat.shape[0]
            mu_b = feat.mean(dim=0)
            M_b  = feat.T @ feat / B

            ema_count = getattr(self, f"ema_count_{idx}")
            ema_mu    = getattr(self, f"ema_mu_{idx}")
            ema_M     = getattr(self, f"ema_M_{idx}")

            if ema_count.item() == 0:
                mu_g = mu_b
                M_g  = M_b
            else:
                b    = self.ema_decay
                mu_g = b * ema_mu.detach() + (1 - b) * mu_b
                M_g  = b * ema_M.detach()  + (1 - b) * M_b

            sig_g = M_g - mu_g.unsqueeze(1) @ mu_g.unsqueeze(0)

            # Update EMA buffers (no grad)
            with torch.no_grad():
                ema_mu.copy_(mu_g.detach())
                ema_M.copy_(M_g.detach())
                ema_count.add_(1)

            # --- Fréchet distance ---
            mu_r       = getattr(self, f"mu_r_{idx}").to(fake_audio.device)
            sig_r      = getattr(self, f"sig_r_{idx}").to(fake_audio.device)
            sig_r_sqrt = getattr(self, f"sig_r_sqrt_{idx}").to(fake_audio.device)

            fd_raw = self.fd_loss(mu_g, sig_g, mu_r, sig_r, sig_r_sqrt)

            # --- Normalize: loss / (stopgrad(loss) + c) ---
            fd_norm = fd_raw / (fd_raw.detach() + self.fd_norm_c)

            # --- Loss magnitude balancing (EMA of |loss|) ---
            loss_scale = getattr(self, f"loss_scale_{idx}")
            with torch.no_grad():
                magnitude = fd_norm.detach().abs().clamp_min(self.fd_loss_balance_eps)
                loss_scale.mul_(self.fd_loss_balance_beta).add_(
                    magnitude * (1.0 - self.fd_loss_balance_beta)
                )

            fd_balanced = fd_norm / loss_scale.detach().clamp_min(self.fd_loss_balance_eps)
            total_fd_loss = total_fd_loss + fd_balanced

            logs[f"fd_raw/{embedder.name}"]      = fd_raw.detach()
            logs[f"fd_norm/{embedder.name}"]     = fd_norm.detach()
            logs[f"fd_balanced/{embedder.name}"] = fd_balanced.detach()

        # Average across embedders then apply global weight
        total_fd_loss = self.fd_weight * total_fd_loss / len(self.embedders)
        logs["fd_total"] = total_fd_loss.detach()
        return total_fd_loss, logs
    
    def on_fit_start(self):
        if not self.fd:
            return

        if not hasattr(self, '_train_dl') or self._train_dl is None:
            raise RuntimeError(
                "Set model._train_dl = train_loader before trainer.fit()"
            )

        all_zeros = all(
            getattr(self, f"mu_r_{i}").abs().sum().item() == 0
            for i in range(len(self.embedders))
        )
        if all_zeros:
            self.precompute_real_stats(self._train_dl)

        already_warm = any(
            getattr(self, f"ema_count_{i}").item() > 0
            for i in range(len(self.embedders))
        )
        if already_warm:
            return

        print("Warm-starting FD EMA from generated audio...")
        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(self._train_dl):
                if i >= self.fd_warm_batches:
                    break
                x = batch.to(self.device)
                encoded_frames = self.encoder.encode(x)
                codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
                emb = self.encoder.quantizer.decode(codes.transpose(0, 1))
                fake = self.decoder(emb).unsqueeze(1)[:, :, :x.size(2)]
                audio = fake.squeeze(1)

                for idx, embedder in enumerate(self.embedders):
                    feat = embedder(audio, self.sample_rate).float()
                    B = feat.shape[0]
                    mu_b = feat.mean(0)
                    M_b  = feat.T @ feat / B

                    ema_count = getattr(self, f"ema_count_{idx}")
                    ema_mu    = getattr(self, f"ema_mu_{idx}")
                    ema_M     = getattr(self, f"ema_M_{idx}")

                    if ema_count.item() == 0:
                        ema_mu.copy_(mu_b)
                        ema_M.copy_(M_b)
                    else:
                        ema_mu.mul_(self.ema_decay).add_(mu_b * (1 - self.ema_decay))
                        ema_M.mul_(self.ema_decay).add_(M_b  * (1 - self.ema_decay))
                    ema_count.add_(1)

        self.train()
        print("FD EMA warm-start complete.")

#--------------------------------------------------------------#

    def training_step(self, batch):

        x = batch  # (B, 1, T)
        sequence_length = x.size(2)

        optimizer_g, optimizer_d = self.optimizers()

        if self.prior == "encodec":
            with torch.no_grad():
                encoded_frames = self.encoder.encode(x)
                codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
                emb = self.encoder.quantizer.decode(codes.transpose(0, 1))

        fake = self.decoder(emb)  # (B, shift_dim * T)
        fake = fake.unsqueeze(1)  # (B, shift_dim * T) -> (B, 1, shift_dim * T)
        fake = fake[:, :, :sequence_length]
        fake_detached = fake.detach()

        mel_fake = self.mel_extractor(fake).squeeze(1)
        g_loss_recon = self.reconstruction_loss(mel_fake, self.mel_extractor(x).squeeze(1))

        if self.current_epoch >= self.warmup_epochs:
            optimizer_d.zero_grad()

            real_fmaps_mpd, real_out_mpd = self.discriminator_mpd(x)
            real_fmaps_mrd, real_out_mrd = self.discriminator_mrd(x)
            real_fmaps_msstft, real_out_msstft = self.discriminator_msstft(x)
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

        if self.current_epoch >= self.warmup_epochs:
            fake_fmaps_mpd, fake_out_mpd = self.discriminator_mpd(fake)
            fake_fmaps_mrd, fake_out_mrd = self.discriminator_mrd(fake)
            fake_fmaps_msstft, fake_out_msstft = self.discriminator_msstft(fake)
            real_fmaps_mpd, _ = self.discriminator_mpd(x)
            real_fmaps_mrd, _ = self.discriminator_mrd(x)
            real_fmaps_msstft, _ = self.discriminator_msstft(x)

            g_loss_mpd = self.adversarial_loss.generator_loss(fake_out_mpd)
            g_loss_mrd = self.adversarial_loss.generator_loss(fake_out_mrd)
            g_loss_msstft = self.adversarial_loss.generator_loss(fake_out_msstft)
            g_loss_adv = g_loss_mpd + self.w_mrd * g_loss_mrd + self.w_msstft * g_loss_msstft

            g_loss_fm_mpd = self.feature_matching_loss(fake_fmaps_mpd, real_fmaps_mpd)
            g_loss_fm_mrd = self.feature_matching_loss(fake_fmaps_mrd, real_fmaps_mrd)
            g_loss_fm_msstft = self.feature_matching_loss(fake_fmaps_msstft, real_fmaps_msstft)
            g_loss_fm = g_loss_fm_mpd + self.w_mrd * g_loss_fm_mrd + self.w_msstft * g_loss_fm_msstft

            total_g_loss = g_loss_adv + self.w_mel * g_loss_recon + g_loss_fm

            if self.fd:
                fd_loss, fd_logs = self._fd_loss_all_embedders(fake.squeeze(1))
                total_g_loss = total_g_loss + fd_loss
                self.log_dict(fd_logs, on_step=True, on_epoch=True, sync_dist=True)

        else:
            total_g_loss = self.w_mel * g_loss_recon

        self.log('g_loss', total_g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        total_g_loss.backward()
        optimizer_g.step()
     

    def validation_step(self, batch):
        x = batch

        with torch.no_grad():
            encoded_frames = self.encoder.encode(x)
            codes = torch.cat([f[0] for f in encoded_frames], dim=-1)
            emb = self.encoder.quantizer.decode(codes.transpose(0, 1))
            fake = self.decoder(emb)

        # Just log some metrics, no backward
        fake_mel = self.mel_extractor(fake)
        fake_mel.squeeze_(1)
        mel_loss = self.reconstruction_loss(fake_mel, self.mel_extractor(x).squeeze(1))
        self.log('val_mel_loss', mel_loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch):
        pass

    def configure_optimizers(self):

        config = load_config('config_prior_24k.yaml')

        optimizer_g = optim.AdamW(self.decoder.parameters(), lr=self.lr_g, betas=(0.9, 0.999))
        optimizer_d = optim.AdamW(list(self.discriminator_mpd.parameters())
                                   + list(self.discriminator_mrd.parameters())
                                   + list(self.discriminator_msstft.parameters()), lr=self.lr_d, betas=(0.9, 0.999))
        
        scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=config['num_epochs'])
        scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=config['num_epochs'])     

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
