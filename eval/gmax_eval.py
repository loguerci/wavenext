"""
Evaluation script using Gmax repository by Philippe Esling for several baselines including : 
- Reimplementation of WaveNeXt,
- RAVE,
- Morpho V1,
- Morpho V2,
- Dummy Codec (non trained),
- Noise (white noise)

Made to evaluate timbre transfer performance with metrics categorized in 3 groups :
- Audio quality : Audiobox Aesthetics, MMMOS, DeePAQScorer
- Target-domain match : Kernel Audio Distance, FAD
- Source-content preservation : F0 stabilithy, Raw Pitch Accuracy, Raw Chroma Accuracy, Cents RMSE, Voicing Recall, Structure Metric

Author: Loïs Guerci
"""

import math
import os
import sys
import tqdm

sys.path.insert(0, '/home/lois/models_benchmark/')
sys.path.insert(0, '/home/lois/wavenext/')

import torchaudio
import torch
import numpy as np

from testing import *
from quality import AudioboxAestheticsMetric, MMMOSMetric, DeePAQScorer
from audio import FADMetric, MelSpectrogramMetric  
from features import F0StabilityMetric, RawPitchAccuracyMetric, RawChromaAccuracyMetric, CentsRMSEMetric, VoicingRecallMetric, StructureMetric
from distribution import KernelAudioDistance, InstrumentSubsetFADMetric

from models.wavenext import WaveNeXt
from utils.mel import MelSpectra
from torch.utils.data import DataLoader
from dataloader import WaveNeXtDataset
import yaml
from audiobox_aesthetics.infer import AesPredictor, make_inference_batch
import tempfile
import soundfile as sf

# Initialize backends 
_aes_predictor = AesPredictor(checkpoint_pth=None, precision='bf16', sample_rate=24000)
_aes_predictor.setup_model()

def aesthetics_backend(audio, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
    
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    torchaudio.save(tmp_path, audio_tensor, sr)
    
    batch = [{'path': tmp_path}]
    with torch.no_grad():
        scores = _aes_predictor.forward(batch)
    
    os.remove(tmp_path)
    return float(np.mean([scores[0][k] for k in ['CE', 'CU', 'PC', 'PQ']]))

def mmmos_backend(audio, sr):
    # placeholder until real MMMOS is available
    rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    overall = float(np.clip(rms * 50, 1.0, 5.0))
    return {'overall': overall}

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class GMAXEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # -----------------------------------------------------------------
        # Audio quality metrics :

        self.audiobox_aesthetics_metric = AudioboxAestheticsMetric(backend=aesthetics_backend, sample_rate=24000, target_sr=24000)
        self.mmos = MMMOSMetric(axis='overall', predictor=mmmos_backend, sample_rate=24000, target_sr=24000)
        #self.deepaq_scorer = DeePAQScorer(...)

        # Target-domain match metrics :

        self.kernel_audio_distance = KernelAudioDistance(model='clap-2023', pre_compute=False, sample_rate=24000)
        self.instrument_subset_fad = InstrumentSubsetFADMetric(label_key='instrument', label_value='piano', embedding_model='clap-2023', min_samples=4)

        # Source-content preservation metrics :
        self.mel_spectrogram_metric = MelSpectrogramMetric()
        self.f0_stability_metric = F0StabilityMetric(sample_rate=24000)
        self.raw_pitch_accuracy_metric = RawPitchAccuracyMetric(sample_rate=24000)
        self.raw_chroma_accuracy_metric = RawChromaAccuracyMetric(sample_rate=24000)
        self.cents_rms_error_metric = CentsRMSEMetric(sample_rate=24000)
        # -----------------------------------------------------------------

        self.mel_extractor = MelSpectra(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80).to(self.device)

        # Models :

        self.model = WaveNeXt().to(self.device)
        self.model.load_state_dict(torch.load("/home/lois/wavenext/checkpoints/07-05_at_03_56_57/wavenext-epoch=453-val_mel_loss=1.89.ckpt")['state_dict'])
        self.model.eval()

    def evaluate(self, deg, ref):
        results = {}

        results['audiobox_aesthetics'] = self.audiobox_aesthetics_metric(deg.detach().cpu())
        results['mmos'] = self.mmos(deg.detach().cpu())

        results['mel_spectrogram_distance'] = self.mel_spectrogram_metric(self.mel_extractor(deg), self.mel_extractor(ref))
        results['f0_stability'] = self.f0_stability_metric(deg, ref)
        return {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in results.items()}
    

if __name__ == "__main__":
    
    def average_results(results_list):
        keys = results_list[0].keys()
        return {k: np.mean([r[k] for r in results_list]) for k in keys}

    config = load_config('/home/lois/wavenext/config.yaml')

    test_dataset = WaveNeXtDataset(path_csv=config['test_csv'], sample_rate=config['sample_rate'], duration=config['duration'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    
    evaluator = GMAXEvaluator()

    total_results = []
    total_noise_results = []

    print("Evaluating WaveNeXt :")

    pbar = tqdm.tqdm(test_loader)

    import numpy as np
    test_audio = np.random.randn(24000).astype(np.float32)
    print("Testing aesthetics_backend directly:")
    result = aesthetics_backend(test_audio, 24000)
    print("Backend result:", result)

    with torch.no_grad():
        for batch in pbar:
            reference_audio = batch.to(evaluator.device)
            reference_audio =reference_audio.squeeze(1)
            mel = evaluator.mel_extractor(reference_audio.to(evaluator.device))
            mel.squeeze_(1)
            generated_audio = evaluator.model.generator(mel)
            generated_audio = generated_audio[..., :reference_audio.size(-1)]

            #print("Generated audio shape:", generated_audio.shape)
            #print("Reference audio shape:", reference_audio.shape)

            noise = torch.randn_like(generated_audio) * 0.01 
            result = evaluator.evaluate(generated_audio, reference_audio)
            result_noise = evaluator.evaluate(noise, reference_audio)
            
            if not math.isnan(result['f0_stability']):
                total_results.append(result)
            if not math.isnan(result_noise['f0_stability']):
                total_noise_results.append(result_noise)
        
    print("results:", average_results(total_results))
    print("noise_results:", average_results(total_noise_results))