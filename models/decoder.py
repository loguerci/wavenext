import torch.nn as nn
from .blocks import ConvNeXtcausal

class Decoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, shift_dim: int, inter_channels: int, num_blocks: int):
        super(Decoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtcausal(dim, inter_channels) for _ in range(num_blocks)])
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, shift_dim, bias=False) 
        # (B, shift_dim, T) -> (B, 1 , shift_dim * T)
    
    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        x = x.transpose(1, 2)  # (B, T, dim)
        print(x.shape)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, dim, T)
        print(x.shape)
        for block in self.blocks:
            x = block(x)
        print(x.shape)
        x = x.transpose(1, 2)  # (B, T, dim)
        print(x.shape)
        x = self.linear1(x)
        print(x.shape)
        x = self.linear2(x) # (B, T, shift_dim)
        print(x.shape)
        x = x.view(x.size(0), -1) # (B, shift_dim * T)

        return x