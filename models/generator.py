"""
WaveNeXt architecture : Convnext-based Generator 
Author : Loïs Guerci

""" 

import torch
import torch.nn as nn
import cached_conv as cc
from .blocks import ConvNeXt, cached_ConvNeXt
from utils.mel import MelSpectra

class Generator(nn.Module):
    def __init__(self, in_channels: int, fft_dim:int, shift_dim: int, inter_channels: int, num_blocks: int):
        super(Generator, self).__init__()
        self.conv = nn.Conv1d(in_channels, fft_dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(fft_dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXt(fft_dim, inter_channels) for _ in range(num_blocks)])
        self.linear1 = nn.Linear(fft_dim, fft_dim)
        self.linear2 = nn.Linear(fft_dim, shift_dim, bias=False) 
        # (B, shift_dim, T) -> (B, 1 , shift_dim * T)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, fft_dim)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, fft_dim, T)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, fft_dim)
        x = self.linear1(x)
        x = self.linear2(x) # (B, T, shift_dim)
        x = x.view(x.size(0), -1) # (B, shift_dim * T)

        return x
    
if "__main__" == __name__: # run python -m wavenext.models.generator 

    x = torch.randn(2, 1, 48000*5)
    mel_extractor = MelSpectra()
    mel = mel_extractor(x)
    mel.squeeze_(1) 
    print(f"Mel-spectrogram shape: {mel.shape}")
    model = Generator(in_channels=80, fft_dim=1024, shift_dim=256, inter_channels=128, num_blocks=4)
    out = model(mel)
    out = out.unsqueeze(1) # (B, 1, shift_dim * T)
    print(f"Output shape: {out.shape}")


