"""
Evaluation script using Gmax repository by Philippe Esling for several baselines including : 
- Reimplementation of WaveNeXt,
- RAVE,
- Morpho V1,
- Morpho V2,
- Dummy Codec (non trained),
- Noise (white noise)

Made to evaluate timbre transfer performance with metrics categorized in 3 groups :
- Audio quality : Audiobox Aesthetics, MMMOS, DeePAQScorer, ViSQOL, Multi-Scale STFT Distance, Zimtohrli
- Target-domain match : Kernel Audio Distance, FAD, Density and Coverage
- Source-content preservation : F0 stability, Raw Pitch Accuracy, Raw Chroma Accuracy, Cents RMSE, Voicing Recall, Structure Metric

Author: Loïs Guerci
"""

import math
import os
import sys
from librosa import to_mono
import tqdm
import yaml
import tempfile
from rich import print
from rich.console import Console
from rich.table import Table

sys.path.insert(0, '/home/lois/models_benchmark/')
sys.path.insert(0, '/home/lois/wavenext/')

import torchaudio
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import WaveNeXtDataset
from utils.mel import MelSpectra

from testing import *
from quality import AudioboxAestheticsMetric, MMMOSMetric, DeePAQScorer
from audio import FADMetric, MultiScaleSTFTMetric, ZimtohrliMetric, ViSQOLMetric
from features import F0StabilityMetric, RawPitchAccuracyMetric, RawChromaAccuracyMetric, CentsRMSEMetric, ChromaConsistencyMetric, VoicingRecallMetric, StructureMetric
from distribution import KernelAudioDistance, InstrumentSubsetFADMetric, DensityAndCoverage
from audiobox_aesthetics.infer import AesPredictor
from audio.utils import _to_numpy

# Models :
from models.wavenext import WaveNeXt


sample_rate = 24000

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def average_results(results_list):
    keys = results_list[0].keys()
    return {k: np.mean([r[k] for r in results_list]) for k in keys}

# Initialize backends 
_aes_predictor = AesPredictor(checkpoint_pth=None, precision='bf16', sample_rate=sample_rate)
_aes_predictor.device = 'cuda' if torch.cuda.is_available() else 'cpu'
_aes_predictor.setup_model()

def aesthetics_backend(audio, sr):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp_path = f.name
    
    #audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    torchaudio.save(tmp_path, audio.cpu(), sr)
    
    batch = [{'path': tmp_path}]
    with torch.no_grad():
        scores = _aes_predictor.forward(batch)
    
    os.remove(tmp_path)
    return scores[0]


class GMAXEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # -----------------------------------------------------------------
        # Audio quality metrics :

        # - reference free :
        self.audiobox_aesthetics_metric = AudioboxAestheticsMetric(backend=aesthetics_backend, sample_rate=sample_rate, target_sr=sample_rate)
        self.mmos = MMMOSMetric(axis='overall', predictor=None, sample_rate=sample_rate, target_sr=sample_rate)
        self.zimtohrli_metric = ZimtohrliMetric(sample_rate=sample_rate, target_sr=sample_rate)

        # - reference based :
        self.multiscale_stft_metric = MultiScaleSTFTMetric(scales=[512, 256, 128], overlap=0.5)
        self.visqol_metric = ViSQOLMetric(sample_rate=sample_rate)

        # Target-domain match metrics :

        self.kernel_audio_distance = KernelAudioDistance(model='clap-2023', pre_compute=False, sample_rate=sample_rate)
        self.instrument_subset_fad = InstrumentSubsetFADMetric(label_key='instrument', label_value='piano', embedding_model='clap-2023', min_samples=4)
        self.density_and_coverage = DensityAndCoverage(nearest_k=1, compute_pr=True)

        # Source-content preservation metrics :
        self.f0_stability_metric = F0StabilityMetric(sample_rate=sample_rate)
        self.raw_pitch_accuracy_metric = RawPitchAccuracyMetric(sample_rate=sample_rate)
        self.raw_chroma_accuracy_metric = RawChromaAccuracyMetric(sample_rate=sample_rate)
        self.cents_rms_error_metric = CentsRMSEMetric(sample_rate=sample_rate)
        self.chroma_consistency_metric = ChromaConsistencyMetric(sample_rate=sample_rate)
        self.voicing_recall_metric = VoicingRecallMetric(sample_rate=sample_rate)
        # -----------------------------------------------------------------

        self.mel_extractor = MelSpectra(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80).to(self.device)

        # Models :

        self.model = WaveNeXt().to(self.device)
        self.model.load_state_dict(torch.load("/home/lois/wavenext/checkpoints/07-05_at_03_56_57/wavenext-epoch=453-val_mel_loss=1.89.ckpt")['state_dict'])
        self.model.eval()

    def evaluate(self, deg, ref):
        results = {}

        #results['audiobox_aesthetics'] = self.audiobox_aesthetics_metric(deg.detach().cpu())
        aes_scores = aesthetics_backend(deg, sample_rate)
        results['audiobox_CE'] = aes_scores['CE']
        results['audiobox_CU'] = aes_scores['CU']
        results['audiobox_PC'] = aes_scores['PC']
        results['audiobox_PQ'] = aes_scores['PQ']
        results['multiscale_stft'] = self.multiscale_stft_metric(deg, ref)
        results['visqol'] = self.visqol_metric(deg, ref)

        results['kernel_audio_distance'] = self.kernel_audio_distance(deg, ref)

        results['f0_stability'] = self.f0_stability_metric(deg, ref)
        results['cents_rms_error'] = self.cents_rms_error_metric(deg, ref)
        results['raw_pitch_accuracy'] = self.raw_pitch_accuracy_metric(deg, ref)
        results['raw_chroma_accuracy'] = self.raw_chroma_accuracy_metric(deg, ref)
        results['chroma_consistency'] = self.chroma_consistency_metric(deg, ref)
        results['voicing_recall'] = self.voicing_recall_metric(deg, ref)

        return {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in results.items()}
    

if __name__ == "__main__":

    config = load_config('/home/lois/wavenext/config.yaml')

    test_dataset = WaveNeXtDataset(path_csv=config['test_csv'], sample_rate=config['sample_rate'], duration=config['duration'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    
    evaluator = GMAXEvaluator()

    total_results = []
    total_noise_results = []
    total_identity_results = []
    gen = []
    ref = []
    ran = []

    print("Evaluating WaveNeXt :")

    pbar = tqdm.tqdm(test_loader)

    with torch.no_grad():
        for batch in pbar:
            reference_audio = batch.to(evaluator.device)
            reference_audio =reference_audio.squeeze(1)
            mel = evaluator.mel_extractor(reference_audio.to(evaluator.device))
            mel.squeeze_(1)
            generated_audio = evaluator.model.generator(mel)
            generated_audio = generated_audio[..., :reference_audio.size(-1)]

            noise = torch.rand_like(generated_audio)

            result = evaluator.evaluate(generated_audio, reference_audio)
            result_noise = evaluator.evaluate(noise, reference_audio)
            result_identity = evaluator.evaluate(reference_audio, reference_audio)

            if not math.isnan(result['f0_stability']):
                total_results.append(result)
            if not math.isnan(result_noise['f0_stability']):
                total_noise_results.append(result_noise)
            if not math.isnan(result_identity['f0_stability']):
                total_identity_results.append(result_identity)

            gen.append(generated_audio)
            ref.append(reference_audio)
            ran.append(noise)
            
    generated_audio = torch.cat(gen, dim=0)
    reference_audio = torch.cat(ref, dim=0)
    
    table = Table(title="Evaluation Results")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("WaveNeXt", justify="right", style="magenta")
    table.add_column("Random", justify="right", style="red")
    table.add_column("Identity", justify="right", style="green")    

    for key in total_results[0].keys():
        table.add_row(key, f"{average_results(total_results)[key]:.4f}", f"{average_results(total_noise_results)[key]:.4f}", f"{average_results(total_identity_results)[key]:.4f}")


    dataset_results = evaluator.density_and_coverage(generated_audio, reference_audio)
    dataset_noise_results = evaluator.density_and_coverage(noise, reference_audio)
    dataset_identity_results = evaluator.density_and_coverage(reference_audio, reference_audio)
    table.add_row("Density and Coverage"
                  , f"Density: {dataset_results[0]:.4f}, Coverage: {dataset_results[1]:.4f}, Precision: {dataset_results[2]:.4f}, Recall: {dataset_results[3]:.4f}"
                  , f"Density: {dataset_noise_results[0]:.4f}, Coverage: {dataset_noise_results[1]:.4f}, Precision: {dataset_noise_results[2]:.4f}, Recall: {dataset_noise_results[3]:.4f}"
                  , f"Density: {dataset_identity_results[0]:.4f}, Coverage: {dataset_identity_results[1]:.4f}, Precision: {dataset_identity_results[2]:.4f}, Recall: {dataset_identity_results[3]:.4f}"
    )
    console = Console()
    console.print(table)
