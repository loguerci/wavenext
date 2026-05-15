import torch
import yaml
from models.wavenext import WaveNeXt

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('/home/lois/wavenext/config_24k.yaml')

model = WaveNeXt(dim=config['dim'], 
                 sample_rate=config['sample_rate'], 
                 fft_dim=config['fft_dim'], 
                 shift_dim=config['shift_dim'], 
                 n_mels=config['n_mels'], 
                 k=config['k'], 
                 lr=config['learning_rate']).to('cuda')

model.load_state_dict(torch.load("/home/lois/wavenext/checkpoints/07-05_at_03_56_57/wavenext-epoch=453-val_mel_loss=1.89.ckpt")['state_dict'])
model.eval()

export_model= torch.jit.script(model.generator)
torch.jit.save(export_model, "my_wavenext_24k.pt")
