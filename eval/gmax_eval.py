import sys

sys.path.insert(0, '/home/lois/models_benchmark/')
sys.path.insert(0, '/home/lois/wavenext/')

import torchaudio
import torch

from testing import *
from quality import DeePAQScorer, MMMOSMetric, AudioboxAestheticsMetric
from audio import FADMetric, MelSpectrogramMetric
from distribution import KernelAudioDistance
from features import F0StabilityMetric, RawPitchAccuracyMetric, VoicingRecallMetric, StructureMetric

from models.wavenext import WaveNeXt
from utils.mel import MelSpectra

class GMAXEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.mel_spectrogram_metric = MelSpectrogramMetric()
        self.f0_stability_metric = F0StabilityMetric()
        self.raw_pitch_accuracy_metric = RawPitchAccuracyMetric(sample_rate=24000)

        self.mel_extractor = MelSpectra(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)

        self.model = WaveNeXt().to(self.device)
        self.model.load_state_dict(torch.load("/home/lois/wavenext/checkpoints/06-05_at_15_04_31/wavenext-epoch=410-val_mel_loss=2.58.ckpt")['state_dict'])

    def evaluate(self, generated_audio, reference_audio):
        results = {}
        results['mel_spectrogram_distance'] = self.mel_spectrogram_metric(self.mel_extractor(generated_audio), self.mel_extractor(reference_audio))
        results['f0_stability'] = self.f0_stability_metric(generated_audio, reference_audio)
        results['raw_pitch_accuracy'] = self.raw_pitch_accuracy_metric(generated_audio, reference_audio)
       
        return results
    

if __name__ == "__main__":
    evaluator = GMAXEvaluator()

    reference_audio = "/home/lois/wavenext/logs/06-05_at_15_04_31/wavenext/version_0/audio_epoch_460/sample_0_real.wav"
    reference_audio, _ = torchaudio.load(reference_audio)

    with torch.no_grad():
        mel = evaluator.mel_extractor(reference_audio.to(evaluator.device))
        mel.squeeze_(1)
        generated_audio = evaluator.model.generator(mel)
        generated_audio = generated_audio[..., :reference_audio.size(-1)]

    noise = torch.randn_like(generated_audio)
    
    results = evaluator.evaluate(generated_audio, reference_audio)
    print("results:", results)

    random_noise_results = evaluator.evaluate(noise, reference_audio)
    print("random_noise_results:", random_noise_results)

    identical_results = evaluator.evaluate(reference_audio, reference_audio)
    print("identical_results:", identical_results)
