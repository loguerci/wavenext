"""
Discriminator architecture for WaveNeXt : MPD (from HiFi-GAN) and MRD (from UnivNet)
Author : Loïs Guerci

""" 

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
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
        for i, (period, disc) in enumerate(zip(self.periods, self.discriminators)):
            fmap, out = disc(period_reshape(x, period))
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
            fmaps, out = one_resolution(x)
            fmap.append(fmaps)
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
            weight_norm(nn.Conv2d(32, 32, (3, 3), (1, 1), padding=(1, 1)))
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

class MSSTFTDiscriminator(nn.Module):
    """
    Multi-Scale STFT Discriminator from EnCodec.
    Operates on complex STFT: processes real and imaginary parts jointly.
    """
    def __init__(self,
                 fft_sizes=[2048, 1024, 512, 256, 128],
                 hop_lengths=[512, 256, 128, 64, 32],
                 win_lengths=[2048, 1024, 512, 256, 128]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            OneScaleSTFT(n, h, w)
            for n, h, w in zip(fft_sizes, hop_lengths, win_lengths)
        ])

    def forward(self, x):
        fmaps, outputs = [], []
        for disc in self.discriminators:
            fmap, out = disc(x)
            fmaps.append(fmap)
            outputs.append(out)
        return fmaps, outputs


class OneScaleSTFT(nn.Module):
    def __init__(self, fft_size, hop_length, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.conv = nn.ModuleList([
            weight_norm(nn.Conv2d(2, 32, (3, 9), padding=(1, 4))), 
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(32, 1, 3, 1, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (B, 1, T)
        x = x.squeeze(1)  # (B, T)
        with torch.no_grad():
            window = torch.hann_window(self.win_length, device=x.device)
            stft = torch.stft(x, self.fft_size, self.hop_length,
                              self.win_length, window,
                              return_complex=True)  # (B, F, T_frames)
            # Stack real and imaginary as channels
            x = torch.stack([stft.real, stft.imag], dim=1)  # (B, 2, F, T_frames)
        fmap = []
        for conv in self.conv:
            x = self.leaky_relu(conv(x))
            fmap.append(x)
        return fmap, self.conv_post(x)

    
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
        print(f"  Scale {i}:")
        for j, fmap in enumerate(fmaps[i]):
            print(f"    Layer {j}: {fmap.shape}")


    model = MSSTFTDiscriminator()
    fmaps, out = model(x)   

    # Print output shapes
    print("MSSTFT Discriminator output shapes:")
    for i in range(len(out)):
        print(f"  Scale {i}: {out[i].shape}")
    # Print feature map shapes
    print("MSSTFT Discriminator FMAP shapes:")
    for i in range(len(fmaps)):
        print(f"  Scale {i}:")
        for j, fmap in enumerate(fmaps[i]):
            print(f"    Layer {j}: {fmap.shape}")