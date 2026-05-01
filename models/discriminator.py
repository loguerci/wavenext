"""
Discriminator architecture for WaveNeXt : MPD (from HiFi-GAN) and MRD (from UnivNet)
Author : Loïs Guerci

""" 

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Spectrogram



def period_reshape(x, period):
    b, c, t = x.shape
    if t % period != 0:
        pad_len = period - (t % period)
        x = nn.functional.pad(x, (0, pad_len), "reflect")
        t = t + pad_len
    x = x.view(b, c, t // period, period)  # (B, C, T//period, period)
    return x

class MPD(nn.Module):
    """
    Multi-Period waveform Discriminator (MPD) from HiFi-GAN
    arguments:
    - periods = [2, 3, 5, 7, 11] : list of periods to reshape the input audio for each sub-discriminator.
    - discriminators : list of sub-discriminators, each taking input of shape (B, 1, T//period, period).
    
    """
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(MPD, self).__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([OnePeriod(period) for period in periods])
    
    def forward(self, x):
        fmaps = []
        outputs = []
        for period in self.periods:
            one_period = self.discriminators[self.periods.index(period)]
            one_period.to(x.device)
            fmap, out = one_period(period_reshape(x, period))
            fmaps.append(fmap)
            outputs.append(out)
        return fmaps, outputs


class OnePeriod(nn.Module):
    """
    Sub-discriminator for a specific period.
    arguments:
    - period : the period for reshaping the input audio.
    """

    def __init__(self, period):
        super(OnePeriod, self).__init__()
        self.period = period
        self.conv = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), padding=(2, 0)))
        ])
        self.conv2 = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        #print(f"Input shape to OnePeriod (period={self.period}): {x.shape}")
        fmap = []
        for conv in self.conv:
            x = conv(x)
            x = self.leaky_relu(x)
            fmap.append(x)
        x = self.conv2(x)
        return fmap, x


class MRD(nn.Module):
    """ 
    Multi-Resolution spectrogram Discriminator (MRD) adapted from UnivNet : https://github.com/rishikksh20/UnivNet-pytorch/blob/master/discriminator.py
    arguments:
    - fft_sizes = [2048, 1024, 512] : list of FFT sizes for the spectrogram input to each sub-discriminator
    - hop_lenghts = [240, 120, 50] : list of hop lengths for the spectrogram input to each sub-discriminator
    - win_lengths = [1200, 600, 240] : list of window lengths for the spectrogram input to each sub-discriminator
    - discriminators : list of sub-discriminators, each taking input of shape (B, 1, F, T).
    """
    def __init__(self, 
                 fft_sizes=[2048, 1024, 512],
                 hop_lenghts=[240, 120, 50],
                 win_lengths=[1200, 600, 240]):
        super(MRD, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_lenghts = hop_lenghts
        self.win_lengths = win_lengths
        
        self.discriminators = nn.ModuleList([
            OneResolution(fft_sizes[i], hop_lenghts[i], win_lengths[i]) for i in range(len(fft_sizes))
        ])

    def forward(self, x):
        fmap = []
        outputs = []
        for i in range(len(self.fft_sizes)):
            one_resolution = self.discriminators[i]
            one_resolution.to(x.device)
            fmaps, out = one_resolution(x)
            fmap.extend(fmaps)
            outputs.append(out)
        return fmap, outputs


class OneResolution(nn.Module):
    def __init__(self, fft_size, hop_length, win_length):
        super(OneResolution, self).__init__()
        self.fft_size = fft_size
        self.spectrogram = Spectrogram(n_fft=fft_size, hop_length=hop_length, win_length=win_length, window_fn=torch.hann_window)

        self.conv = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 1), padding=(1, 1)))
        ])
        self.conv2 = weight_norm(nn.Conv2d(32, 1, 3, 1, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        with torch.no_grad():
            x = self.spectrogram(x)  # (B, 1, F, T)
        fmap = []
        for conv in self.conv:
            x = conv(x)
            x = self.leaky_relu(x)
            fmap.append(x)
        x = self.conv2(x)
        return fmap, x
    
if "__main__" == __name__:
    x = torch.randn(1, 1, 24000*1)

    model = MPD()
    fmaps, out = model(x)

    # Print output shapes
    print("MPD output shapes:")
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)

    # Print feature map shapes
    for i in range(len(fmaps)):
        print(f"FMAP {i} shapes:")
        for j, fmap in enumerate(fmaps[i]):
            print(f"  Layer {j}: {fmap.shape}")
    

    model = MRD()
    fmaps, out = model(x)

    # Print output shapes
    print("MRD output shapes:")
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)

    # Print feature map shapes
    print("MRD FMAP shapes:")
    for i in range(len(fmaps)):
        print(f"  Layer {i}: {fmaps[i].shape}")