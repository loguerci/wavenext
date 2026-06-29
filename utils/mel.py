"""
Mel-spectrogram extractor for GAN training
Author : Loïs Guerci

""" 

import torch
import torch.nn as nn
import torchaudio


class MelSpectra(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, n_mels: int):
        super(MelSpectra, self).__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, x):
        mel = self.mel_spectrogram(x)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel
        
    

if "__main__" == __name__:
    x = torch.randn(1, 1, 24000*2)
    mel_extractor = MelSpectra(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=128)
    mel = mel_extractor(x)
    mel.squeeze_(1) # (B, n_mels, T)
    print(mel.shape)