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

The test currently performed on the reimplementation of WaveNeXt (trained on the libriTTS dataset) and a non trained auto encoder.

Author: Loïs Guerci
"""

import sys
sys.path.insert(0, '/home/lois/models_benchmark/')
sys.path.insert(0, '/home/lois/wavenext/')

import math
import tqdm
import yaml
from rich import print
from rich.console import Console
from rich.table import Table

import torch
import numpy as np
from torch.utils.data import DataLoader
from dataloader import WaveNeXtDataset
from utils.mel import MelSpectra

from testing import *
from quality import AudioboxAestheticsMetric, MMMOSMetric, DeePAQScorer
from audio import FADMetric, MultiScaleSTFTMetric, ZimtohrliMetric, ViSQOLMetric
from features import F0StabilityMetric, RawPitchAccuracyMetric, RawChromaAccuracyMetric, CentsRMSEMetric, ChromaConsistencyMetric, VoicingRecallMetric, StructureMetric
from distribution import KernelAudioDistance, DensityAndCoverage
from audiobox_aesthetics.infer import AesPredictor
import zimtohrli as zim

# Models :
from models.wavenext import WaveNeXt
from models.dummy_ae import DumpCodec

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('/home/lois/wavenext/config_24k.yaml')

def average_results(results_list):
    keys = results_list[0].keys()
    return {k: np.mean([r[k] for r in results_list]) for k in keys}


class GMAXEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # -----------------------------------------------------------------
        # Audio quality metrics :

        # - reference free :
        #need to pip install audiobox-aesthetics and import the checkpoint file : https://huggingface.co/facebook/audiobox-aesthetics/blob/main/checkpoint.pt
        self.audiobox = AesPredictor(checkpoint_pth='audiobox_checkpoint.pt', precision='bf16', sample_rate=config['sample_rate'], data_col='wav')
        self.audiobox.device = 'cuda'
        self.audiobox.setup_model()

        # - reference based :
        self.multiscale_stft_metric = MultiScaleSTFTMetric(scales=[512, 256, 128], overlap=0.5)
        self.visqol_metric = ViSQOLMetric(sample_rate=config['sample_rate'])

        # Target-domain match metrics :

        self.kernel_audio_distance = KernelAudioDistance(model='clap-2023', pre_compute=False, sample_rate=config['sample_rate'])
        self.fad = FADMetric(sample_rate=config['sample_rate'], duration_seconds=1.0, embedding_model='panns-wavegram-logmel')
        self.density_and_coverage = DensityAndCoverage(nearest_k=1, compute_pr=True)

        # Source-content preservation metrics :
        self.f0_stability_metric = F0StabilityMetric(sample_rate=config['sample_rate'])
        self.raw_pitch_accuracy_metric = RawPitchAccuracyMetric(sample_rate=config['sample_rate'])
        self.raw_chroma_accuracy_metric = RawChromaAccuracyMetric(sample_rate=config['sample_rate'])
        self.cents_rms_error_metric = CentsRMSEMetric(sample_rate=config['sample_rate'])
        self.chroma_consistency_metric = ChromaConsistencyMetric(sample_rate=config['sample_rate'])
        self.voicing_recall_metric = VoicingRecallMetric(sample_rate=config['sample_rate'])
        # -----------------------------------------------------------------

        self.mel_extractor = MelSpectra(sample_rate=config['sample_rate'], n_fft=1024, hop_length=256, n_mels=80).to(self.device)

        # Models :
        self.model = WaveNeXt().to(self.device)
        self.model.load_state_dict(torch.load("/home/lois/wavenext/checkpoints/07-05_at_03_56_57/wavenext-epoch=453-val_mel_loss=1.89.ckpt")['state_dict'])
        self.model.eval()

        self.dumb = DumpCodec().to(self.device)
        self.dumb.eval()

    def aesthetics_backend(self, audio, sr):

        batch = [{'wav': audio, 'sample_rate': sr}]
        with torch.no_grad():
            scores = self.audiobox.forward(batch)

        return scores[0]

    def evaluate(self, deg, ref):
        results = {}

        results['audiobox : content enjoyment'] = self.aesthetics_backend(deg, config['sample_rate'])['CE']
        results['audiobox : content usefulness'] = self.aesthetics_backend(deg, config['sample_rate'])['CU']
        results['audiobox : production complexity'] = self.aesthetics_backend(deg, config['sample_rate'])['PC']
        results['audiobox : production quality'] = self.aesthetics_backend(deg, config['sample_rate'])['PQ']

        results['multiscale_stft'] = self.multiscale_stft_metric(deg, ref)
        results['visqol'] = self.visqol_metric(deg, ref)

        results['kernel_audio_distance'] = self.kernel_audio_distance(deg, ref)
        results['fad'] = self.fad(deg, ref)

        results['f0_stability'] = self.f0_stability_metric(deg, ref)
        results['cents_rms_error'] = self.cents_rms_error_metric(deg, ref)
        results['raw_pitch_accuracy'] = self.raw_pitch_accuracy_metric(deg, ref)
        results['raw_chroma_accuracy'] = self.raw_chroma_accuracy_metric(deg, ref)
        results['chroma_consistency'] = self.chroma_consistency_metric(deg, ref)
        results['voicing_recall'] = self.voicing_recall_metric(deg, ref)

        return {k: v.item() if isinstance(v, torch.Tensor) else float(v) for k, v in results.items()}
    

if __name__ == "__main__":


    # Loading dataset 
    data = "/home/lois/wavenext/data/libritts_test.csv"
    test_dataset = WaveNeXtDataset(path_csv=data, sample_rate=config['sample_rate'], duration=config['duration'], limit =100)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])
    
    # Init evaluation benchmark
    evaluator = GMAXEvaluator()

    total_results = []
    total_noise_results = []
    total_dumb_results = []
    gen = []
    ref = []
    ran = []
    dumb = []

    print("Evaluating WaveNeXt...")

    pbar = tqdm.tqdm(test_loader)

    with torch.no_grad():
        for batch in pbar:
            reference_audio = batch.to(evaluator.device)
            reference_audio = reference_audio.squeeze(1)
            mel = evaluator.mel_extractor(reference_audio.to(evaluator.device))
            mel.squeeze_(1)
            generated_audio = evaluator.model.generator(mel)
            generated_audio = generated_audio[..., :reference_audio.size(-1)]
            dumb_gen = evaluator.dumb(reference_audio)
            dumb_gen.squeeze_(1)

            noise = torch.rand_like(generated_audio)

            result = evaluator.evaluate(generated_audio, reference_audio)
            result_dumb= evaluator.evaluate(dumb_gen, reference_audio)
            result_noise = evaluator.evaluate(noise, reference_audio)

            if not math.isnan(result['f0_stability']):
                total_results.append(result)
            if not math.isnan(result_noise['f0_stability']):
                total_noise_results.append(result_noise)
            if not math.isnan(result_dumb['f0_stability']):
                total_dumb_results.append(result_dumb)

            gen.append(generated_audio)
            ref.append(reference_audio)
            ran.append(noise)
            dumb.append(dumb_gen)
            
    generated_audio = torch.cat(gen, dim=0)
    reference_audio = torch.cat(ref, dim=0)
    noise = torch.cat(ran, dim=0)
    dumb_audio = torch.cat(dumb, dim=0)
    
    table = Table(title="Evaluation Results")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("WaveNeXt", justify="right", style="magenta")
    table.add_column("Random", justify="right", style="red")  
    table.add_column("Dumb model", justify="right", style="yellow")  

    for key in total_results[0].keys():
        table.add_row(key, f"{average_results(total_results)[key]:.4f}", f"{average_results(total_noise_results)[key]:.4f}", f"{average_results(total_dumb_results)[key]:.4f}")


    dataset_results = evaluator.density_and_coverage(generated_audio, reference_audio)
    dataset_noise_results = evaluator.density_and_coverage(noise, reference_audio)
    dataset_db_results = evaluator.density_and_coverage(dumb_audio, reference_audio)

    table.add_row("Density", f"{dataset_results[0]:.4f}", f"{dataset_noise_results[0]:.4f}", f"{dataset_db_results[0]:.4f}")
    table.add_row("Coverage", f"{dataset_results[1]:.4f}", f"{dataset_noise_results[1]:.4f}", f"{dataset_db_results[1]:.4f}")
    table.add_row("Precision", f"{dataset_results[2]:.4f}", f"{dataset_noise_results[2]:.4f}", f"{dataset_db_results[2]:.4f}")
    table.add_row("Recall", f"{dataset_results[3]:.4f}", f"{dataset_noise_results[3]:.4f}", f"{dataset_db_results[3]:.4f}")
    console = Console()
    console.print(table)


