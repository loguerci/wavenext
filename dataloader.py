"""
Dataloader for the WaveNext model :
- MockDataset : A simple dataset that generates random noise as audio samples for testing purposes.
- AudioDataset : A dataset that loads audio files from a specified directory

Author : Loïs Guerci

"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import soundfile as sf
from utils.mel import MelSpectra
import os


class MockDataset(Dataset):
    """
    A simple dataset that generates random noise as audio samples for testing purposes.
    """
    def __init__(self, num_samples=1000, sample_length=24000*5):
        self.num_samples = num_samples
        self.sample_length = sample_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        audio = torch.randn(1, self.sample_length)  # (1, T)
        return audio

class WaveNeXtDataset(Dataset):
    def __init__(self, path_csv, sample_rate=24000, duration=1, limit=None):
        self.sample_rate = sample_rate
        self.segment_length = sample_rate * duration
        self.segments = []  # list of (file_path, start_sample)
        self.limit = limit

        with open(path_csv, 'r') as f:
            for line in f:
                path = line.strip()
                if path.endswith('.mp3'):
                    wav_path = path.replace('.mp3', '.wav')
                    if not os.path.exists(wav_path):
                        print(f"Converting {path} to wav...")
                        audio, sr = sf.read(path)
                        sf.write(wav_path, audio, sr)
                    path = wav_path
                info = sf.info(path)
                file_sr = info.samplerate
                num_samples = int(info.frames * sample_rate / file_sr)
                n_segments = num_samples // self.segment_length
                for i in range(n_segments):
                    self.segments.append((path, file_sr, i * self.segment_length))
                if limit is not None:
                    self.segments = self.segments[:self.limit]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        path, file_sr, start = self.segments[idx]
        # Load only the needed frames
        start_orig = int(start * file_sr / self.sample_rate)
        frames_orig = int(self.segment_length * file_sr / self.sample_rate)
        audio, sr = torchaudio.load(path, frame_offset=start_orig, num_frames=frames_orig)

        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)

        # Ensure exact length
        if audio.size(1) < self.segment_length:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(1)))
        else:
            audio = audio[:, :self.segment_length]

        return audio



if "__main__" == __name__:

    print("Process LibriTTS dataset")
    dataset = WaveNeXtDataset(path_csv="data/libritts_dataset.csv", sample_rate=24000, duration=1)
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")

    mel_extractor = MelSpectra() 
    mel = mel_extractor(batch)
    mel.squeeze_(1) # (B, n_mels, T)
    print(f"Mel shape: {mel.shape}")

    print("Process Bach violin dataset")
    dataset =  WaveNeXtDataset(path_csv="data/bach_violin_dataset.csv", sample_rate=44100, duration=1, limit=None)
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}") 

    mel = mel_extractor(batch)
    mel.squeeze_(1) # (B, n_mels, T)
    print(f"Mel shape: {mel.shape}")