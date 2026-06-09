import torch
import torch.nn as nn

class Naive_SlowBranch(nn.Module):
    def __init__(self, latent_dim, film_dim):
        super().__init__() 
        self.gru = nn.GRU(latent_dim, 128, num_layers=4, 
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(256, film_dim * 2)  # *2 pour α et β
    
    def forward(self, latents):  # latents: (B, 16, latent_dim)
        out, _ = self.gru(latents)
        pooled = out.mean(dim=1)       # (B, 256)
        film_params = self.proj(pooled) # (B, film_dim*2)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma, beta

class Transformer_SlowBranch(nn.Module):
    def __init__(self, latent_dim, dim, num_heads=4):
        super().__init__()

        self.proj = nn.Linear(latent_dim, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, num_heads, batch_first=True),
            num_layers=2
        )
        self.to_kv = nn.Linear(dim, dim * 2)    

    def forward(self, latents):
        x = self.proj(latents)    
        x = self.transformer(x)        
        K, V = self.to_kv(x).chunk(2, dim=-1)
        return K, V 


class CrossAttentionConditioning(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
    
    def forward(self, x, K, V):
        x = x.transpose(1, 2)            # (B, T, dim)
        Q = self.to_q(x)                 # (B, T, dim)
        
        attn = torch.softmax(Q @ K.transpose(1, 2) / self.dim**0.5, dim=-1) # (B, T, slow_size)
        out = attn @ V                   # (B, T, dim)
        
        x = x + self.out_proj(out)
        return x.transpose(1, 2)   