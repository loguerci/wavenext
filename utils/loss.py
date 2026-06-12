"""
Loss functions for WaveNeXt model training and validation :
- Reconstruction loss : L1 loss between the generated audio and the target audio.
- Adversarial loss : Hinge loss for the  kth sub-discriminators and generator.
- Feature matching loss : L1 loss between the feature maps of the real and fake audio from the discriminators.

Author : Loïs Guerci

"""
import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    """
    L1 loss between the mel spectrogram of the generated audio and the target audio.
    arguments:
    - pred : (B, C, T) tensor of mel spectrogram.
    - target : (B, C, T) tensor of target mel spectrogram.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = nn.functional.l1_loss(pred, target)
        return loss
    

class AdversarialLoss(nn.Module):
    """
    Hinge loss for the kth sub-discriminators and generator.
    arguments:
    - real_out, fake_out : 
        (B, 1, C, p) => C : T//p (compressed),  p : period [MPD]
        (B, 1, F, T) => F : frequency bins (compressed),  T : time bins [MRD]
    """
    def __init__(self):
        super().__init__()

    def discriminator_loss(self, real_out, fake_out):
        loss = 0.0
        for r, f in zip(real_out, fake_out):
            one_arr = torch.ones_like(r)
            loss += torch.mean(torch.relu(one_arr - r))
            loss += torch.mean(torch.relu(one_arr + f))
        return loss

    def generator_loss(self, fake_out):
        loss = 0.0
        for f in fake_out:
            one_arr = torch.ones_like(f)
            loss += torch.mean(torch.relu(one_arr - f))
        return loss
    
class FeatureMatchingLoss(nn.Module):
    """
    L1 loss between the feature maps of the real and fake audio from the generator.
    arguments:  
    - real_fmaps, fake_fmaps : 
        (B, L, C, p) => L : Layers dimension, C : T//p (compressed),  p : period [MPD]
        (B, L, F, T) => L : Layers dimension, F : frequency bins (compressed),  T : time bins [MRD]

    """
    def __init__(self):
        super().__init__()

    def forward(self, real_fmaps, fake_fmaps):
        loss = 0.0
        for d_real, d_fake in zip(real_fmaps, fake_fmaps):
            for l_real, l_fake in zip(d_real, d_fake):
                loss += nn.functional.l1_loss(l_real, l_fake)
        return loss
    


class FDLoss(nn.Module):
    def __init__(self, weight: float = 1.0, eps: float = 0.01):
        super().__init__()
        self.weight = weight
        self.eps = eps

    @staticmethod
    def compute_sqrt_cov(cov: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
        """Precompute sig_r_sqrt once offline. Pass the result as a buffer."""
        dim = cov.shape[0]
        cov = 0.5 * (cov + cov.T)
        eye = torch.eye(dim, device=cov.device, dtype=cov.dtype)
        for _ in range(6):
            try:
                eigvals, eigvecs = torch.linalg.eigh(cov + eye * jitter)
                eigvals = eigvals.clamp_min(jitter).sqrt()
                return (eigvecs * eigvals.unsqueeze(0)) @ eigvecs.T
            except torch.linalg.LinAlgError:
                jitter *= 10.0
        # fallback
        u, s, vh = torch.linalg.svd(cov + eye * jitter)
        return (u * s.clamp_min(jitter).sqrt().unsqueeze(0)) @ vh

    def forward(self, mu_g, sig_g, mu_r, sig_r, sig_r_sqrt):
        """
        mu_g, sig_g   : generated distribution (grad flows through these)
        mu_r, sig_r   : real distribution stats (precomputed buffers, no grad)
        sig_r_sqrt    : precomputed matrix sqrt of sig_r (buffer, no grad)
        """
        diff_mu = torch.sum((mu_g - mu_r) ** 2)

        # Symmetrize for numerical safety
        dim = sig_g.shape[0]
        sig_g = 0.5 * (sig_g + sig_g.T)
        sig_g = sig_g + torch.eye(dim, device=sig_g.device, dtype=sig_g.dtype) * self.eps

        # Tr((sig_r @ sig_g)^{1/2}) via eigvalsh of sig_r_sqrt @ sig_g @ sig_r_sqrt
        M = sig_r_sqrt @ sig_g @ sig_r_sqrt
        M = 0.5 * (M + M.T)
        trace_sqrt = torch.linalg.eigvalsh(M).clamp_min(0).sqrt().sum()

        trace_term = torch.trace(sig_g) + torch.trace(sig_r) - 2 * trace_sqrt

        loss = diff_mu + trace_term
        loss = loss.clamp(min=0)  # FD is non-negative
        loss = loss / (loss.detach() + self.eps)
        return loss * self.weight


if "__main__" == __name__: 
    real_fmaps = [torch.randn(2, 4, 16, 64), torch.randn(2, 4, 32, 128)]
    fake_fmaps = [torch.randn(2, 4, 16, 64), torch.randn(2, 4, 32, 128)]
    fm_loss = FeatureMatchingLoss()
    loss = fm_loss(real_fmaps, fake_fmaps)
    print(f"Feature Matching Loss: {loss.item()}")
