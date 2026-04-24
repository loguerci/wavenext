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
from utils.mel import MelSpectra

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
        # Generate random noise as audio sample
        audio = torch.randn(1, self.sample_length)  # (1, T)
        return audio

class WaveNeXtDataset(Dataset):
    def __init__(self, path_csv, sample_rate=24000, duration=5):
        self.file_paths = []
        with open(path_csv, 'r') as f:
            for line in f:
                self.file_paths.append(line.strip())
        self.sample_rate = sample_rate
        self.duration = duration

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        audio, sr = torchaudio.load(file_path)

        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)  # Convert to mono
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio = resampler(audio)

        if audio.size(1) > self.sample_rate * self.duration:
            audio = audio[:, :self.sample_rate * self.duration]  # Truncate to desired duration

        elif audio.size(1) < self.sample_rate * self.duration:
            pad_length = self.sample_rate * self.duration - audio.size(1)
            audio = torch.nn.functional.pad(audio, (0, pad_length))  # Pad with zeros
            
        return audio



if "__main__" == __name__:

    #dataset = MockDataset(num_samples=10, sample_length=24000*5)
    dataset = WaveNeXtDataset(path_csv="libritts_dataset.csv", sample_rate=24000, duration=10)
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")

    mel_extractor = MelSpectra() 
    mel = mel_extractor(batch)
    mel.squeeze_(1) # (B, n_mels, T)
    print(f"Mel shape: {mel.shape}")