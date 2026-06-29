import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import ConvNeXt
from torch.nn.utils.parametrizations import weight_norm

class GRU_SlowBranch(nn.Module):
    def __init__(self, latent_dim, film_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.gru = nn.GRU(
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Linear(hidden_dim * 2, film_dim * 2)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, latents):
        out, _ = self.gru(latents)  # (B, T, 2H)
        weights = torch.softmax(self.attn(out), dim=1)
        pooled = (out * weights).sum(dim=1)

        gamma, beta = self.proj(pooled).chunk(2, dim=-1)
        return gamma, beta
    

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, dim)
    
    def forward(self, x):
        # x : (B, T, dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Transformer_SlowBranch(nn.Module):
    def __init__(self, latent_dim: int, dim: int, num_heads: int = 4,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.proj = nn.Linear(latent_dim, dim)
        self.pos_encoding = PositionalEncoding(dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
    
    def forward(self, latents):
        # latents : (B, slow_size, latent_dim)
        x = self.proj(latents)       # (B, slow_size, dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)      # self-attention bidirectionnelle sur tout le buffer
        x = self.norm(x)
        
        K = self.to_k(x)             # (B, slow_size, dim)
        V = self.to_v(x)             # (B, slow_size, dim)
        return K, V


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def split_heads(self, t: torch.Tensor, B: int, seq_len: int) -> torch.Tensor:
        return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, seq_len, head_dim)
    
    def forward(self, x: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # x : (B, dim, T_fast) 
        # K : (B, slow_size, dim) 
        # V : (B, slow_size, dim)
        B, _, T_fast = x.shape
        slow_size = K.size(1)
        
        # Pre-norm
        x_res = x
        x = self.norm(x.transpose(1, 2))    # (B, T_fast, dim)
        
        Q = self.to_q(x)                    # (B, T_fast, dim)
        
        # Multi-head split
        Q = self.split_heads(Q, B, T_fast)      # (B, heads, T_fast, head_dim)
        K = self.split_heads(K, B, slow_size)   # (B, heads, slow_size, head_dim)
        V = self.split_heads(V, B, slow_size)   # (B, heads, slow_size, head_dim)
        
        # Cross-attention fast → slow
        # Pas de masque causal : toutes les frames slow sont dans le passé par construction
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, T_fast, slow_size)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)         # (B, heads, T_fast, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T_fast, -1)  # (B, T_fast, dim)
        out = self.out_proj(out)            # (B, T_fast, dim)
        
        # Residual
        return x_res + out.transpose(1, 2) # (B, dim, T_fast)