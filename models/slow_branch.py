import torch.nn as nn

class SlowBranch(nn.Module):
    def __init__(self, latent_dim, film_dim):
        self.gru = nn.GRU(latent_dim, 128, num_layers=4, 
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(256, film_dim * 2)  # *2 pour α et β
    
    def forward(self, latents):  # latents: (B, 16, latent_dim)
        out, _ = self.gru(latents)
        pooled = out.mean(dim=1)       # (B, 256)
        film_params = self.proj(pooled) # (B, film_dim*2)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma, beta