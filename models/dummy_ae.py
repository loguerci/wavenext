"""
Just a dummy autoencoder to compare with true baselines on evaluation metrics  

Author: Loïs Guerci
"""

import torch
import torch.nn as nn  


class DumpCodec(nn.Module):
    def __init__(self):
        super().__init__()
        # needs to process 24kHz audio and output 24kHz audio, but otherwise does nothing

        self.encoder = nn.Sequential(
            nn.Linear(24000, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 24000),
        )


    def forward(self, x):
        z = self.encoder(x)  # (B, 1, 256)
        out = self.decoder(z)  # (B, 1, 24000)
        return out
    
if "__main__" == __name__:
    x = torch.randn(1, 1, 24000)


    model = DumpCodec()

    with torch.no_grad():
        y = model(x)
    
    print(x.shape, y.shape)