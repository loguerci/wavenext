import torch.nn as nn
from .blocks import ConvNeXtcausal
from .slow_branch import CrossAttention

class Decoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, shift_dim: int, inter_channels: int, num_blocks: int):
        super(Decoder, self).__init__()
        self.pad_input = nn.ConstantPad1d((6, 0), 0)
        self.conv = nn.Conv1d(in_channels, dim, kernel_size=7, padding=0)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtcausal(dim, inter_channels) for _ in range(num_blocks)])
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, shift_dim, bias=False) 
        # (B, shift_dim, T) -> (B, 1 , shift_dim * T)
    
    def forward(self, x):
        x = self.pad_input(x)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, dim, T)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.linear1(x)
        x = self.linear2(x) # (B, T, shift_dim)
        x = x.view(x.size(0), -1) # (B, shift_dim * T)

        return x


class Cond_Decoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, shift_dim: int, inter_channels: int, num_blocks: int):
        super(Cond_Decoder, self).__init__()
        self.pad_input = nn.ConstantPad1d((6, 0), 0)
        self.conv = nn.Conv1d(in_channels, dim, kernel_size=7, padding=0)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtcausal(dim, inter_channels) for _ in range(num_blocks)])
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, shift_dim, bias=False) 
        # (B, shift_dim, T) -> (B, 1 , shift_dim * T)
    
    def forward(self, x, cond):
        gamma, beta = cond
        x = self.pad_input(x)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, dim, T)
        x = x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

        for block in self.blocks:
            x = block(x)
            x = x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.linear1(x)
        x = self.linear2(x) # (B, T, shift_dim)
        x = x.view(x.size(0), -1) # (B, shift_dim * T)

        return x
    

class CrossA_Decoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, shift_dim: int, inter_channels: int, num_blocks: int):
        super(CrossA_Decoder, self).__init__()
        self.pad_input = nn.ConstantPad1d((6, 0), 0)
        self.conv = nn.Conv1d(in_channels, dim, kernel_size=7, padding=0)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.blocks = nn.ModuleList([ConvNeXtcausal(dim, inter_channels) for _ in range(num_blocks)])
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, shift_dim, bias=False) 
        self.cross_a = CrossAttention(dim)
        # (B, shift_dim, T) -> (B, 1 , shift_dim * T)
    
    def forward(self, x, cond):
        K, V = cond
        x = self.pad_input(x)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, dim, T)
        x = self.cross_a(x, K, V)  

        for block in self.blocks:
            x = block(x)
            x = self.cross_a(x, K, V)  

        x = x.transpose(1, 2)  # (B, T, dim)
        x = self.linear1(x)
        x = self.linear2(x) # (B, T, shift_dim)
        x = x.view(x.size(0), -1) # (B, shift_dim * T)

        return x