"""
Just a dummy autoencoder to compare with true baselines on evaluation metrics  

Author: Loïs Guerci
"""

import torch
import torch.nn as nn  

class DumpCodec(nn.Module):
    def __init__(self, dim=24000):
        super().__init__()
        # needs to process 24kHz audio and output 24kHz audio, but otherwise does nothing

        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)
                nn.init.normal_(m.bias, mean=0, std=1.0)

    def forward(self, x):
        z = self.encoder(x)  # (B, 1, 256)
        out = self.decoder(z)  # (B, 1, 24000)
        return out.clamp(-1,1)
    
if "__main__" == __name__:

    x = torch.randn(1, 1, 24000)


    model = DumpCodec(dim=24000)

    with torch.no_grad():
        y = model(x)
    
    print(x.shape, y.shape)