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


if "__main__" == __name__: 
    real_fmaps = [torch.randn(2, 4, 16, 64), torch.randn(2, 4, 32, 128)]
    fake_fmaps = [torch.randn(2, 4, 16, 64), torch.randn(2, 4, 32, 128)]
    fm_loss = FeatureMatchingLoss()
    loss = fm_loss(real_fmaps, fake_fmaps)
    print(f"Feature Matching Loss: {loss.item()}")
