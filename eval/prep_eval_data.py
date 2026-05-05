import os 
from pathlib import Path
import torch
from models.wavenext import WaveNeXt
from dataloader import WaveNeXtDataset
from torch.utils.data import DataLoader
from utils.mel import MelSpectra

gen_out_path = 'gen_out.scp'
ref_out_path = 'ref_out.scp'

model = WaveNeXt()
mel_extractor = MelSpectra(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)
model.load_state_dict(torch.load('/home/lois/wavenext/checkpoints/04-05_at_13_25_19/wavenext-epoch=151-val_mel_loss=3.33.ckpt')['state_dict'])
model.eval()
dataset = WaveNeXtDataset(path_csv='data/test.csv', sample_rate=24000, duration=1)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, audio in enumerate(dataloader):
    mel_input = mel_extractor(audio)
    mel_input = mel_input.squeeze_(1)
    gen_audio = model(mel_input)
    gen_audio = gen_audio[:,:,:audio.size(2)]
    gen_audio = gen_audio.squeeze().cpu().numpy()
    ref_audio = audio.squeeze().cpu().numpy()

    with open(gen_out_path, 'a') as f_gen, open(ref_out_path, 'a') as f_ref:
        f_gen.write(f'utt_{i} {gen_audio}\n')
        f_ref.write(f'utt_{i} {ref_audio}\n')

