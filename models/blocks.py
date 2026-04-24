"""
WaveNext Blocks for Generator
Author : Loïs Guerci

""" 

import torch
import torch.nn as nn
import cached_conv as cc


class ConvNeXt(nn.Module):
    # Input shape : (B, C, T)
    def __init__(self, in_channels, inter_channels):
        super(ConvNeXt, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pointwise1 = nn.Linear(in_channels, inter_channels)
        self.pointwise2 = nn.Linear(inter_channels, in_channels)

        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = x + res
        return x
    
class cached_ConvNeXt(nn.Module):
    # Input shape : (B, C, T)
    def __init__(self, in_channels, inter_channels):
        super(cached_ConvNeXt, self).__init__()
        self.depthwise = cc.CachedConv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.pointwise1 = nn.Linear(in_channels, inter_channels)
        self.pointwise2 = nn.Linear(inter_channels, in_channels)

    def forward(self, x):
        res = x
        x = self.depthwise(x)
        x = x.transpose(1, 2)  # (B, T, C)        
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = x + res
        return x
    
if "__main__" == __name__:
    x = torch.randn(2, 64, 128)
    model = ConvNeXt(64, 128)
    out = model(x)
    print(out.shape)