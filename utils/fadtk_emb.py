import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class FADTKEmbedding(nn.Module):
    """
    Thin differentiable wrapper around fadtk embedding models.
    Calls the underlying frozen Torch model directly so gradients
    flow to the generated audio, unlike fadtk's public scoring API
    which detaches embeddings for evaluation.
    """

    _KIND_MAP = {
        "clap-2023":        "msclap",
        "clap-laion-audio": "laion_clap",
        "clap-laion-music": "laion_clap",
        "dac-44kHz":        "dac",
        "cdpam-acoustic":   "cdpam",
        "cdpam-content":    "cdpam",
    }

    def __init__(self, loader):
        super().__init__()
        loader.load_model()
        self.loader = loader
        self.sr = loader.sr
        self.num_features = loader.num_features
        self.name = loader.name
        self.kind = self._infer_kind(loader.name)
        self.model = self._unwrap(loader)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # RNN-based models need train() during backward for CuDNN
        self.needs_train_for_backward = any(
            isinstance(m, nn.RNNBase) for m in self.model.modules()
        )

    def _infer_kind(self, name):
        for prefix, kind in self._KIND_MAP.items():
            if name.startswith(prefix):
                return kind
        return None

    def _unwrap(self, loader):
        if self.kind == "msclap":
            return loader.model.clap
        if self.kind == "cdpam":
            return loader.model.model
        return loader.model

    def _prepare(self, audio, sample_rate, channels):
        """Resample and channel-match audio to model requirements."""
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)          # (B, T) -> (B, 1, T)
        if audio.shape[1] != channels:
            if channels == 1:
                audio = audio.mean(dim=1, keepdim=True)
            elif audio.shape[1] == 1:
                audio = audio.repeat(1, channels, 1)
            else:
                audio = audio[:, :channels]
        if sample_rate != self.sr:
            B, C, L = audio.shape
            audio = torchaudio.functional.resample(
                audio.reshape(B * C, L), sample_rate, self.sr
            ).reshape(B, C, -1)
        return audio.clamp(-1.0, 1.0)

    def forward(self, audio, sample_rate):
        device = audio.device
        self.model.to(device)
        if self.loader is not None:
            self.loader.device = device

        if torch.is_grad_enabled() and audio.requires_grad and self.needs_train_for_backward:
            self.model.train()
        else:
            self.model.eval()

        if self.kind == "msclap":
            audio = self._prepare(audio, sample_rate, 1)[:, 0]
            chunk = 7 * self.sr
            frames = []
            for off in range(0, audio.shape[-1], chunk):
                c = audio[:, off:off + chunk]
                if c.shape[-1] < chunk:
                    c = F.pad(c, (0, chunk - c.shape[-1]))
                frames.append(self.model.audio_encoder(c)[0])
            return torch.cat(frames, dim=0)

        if self.kind == "laion_clap":
            audio = self._prepare(audio, sample_rate, 1)[:, 0]
            chunk = 10 * self.sr
            frames = []
            for off in range(0, audio.shape[-1], chunk):
                c = audio[:, off:off + chunk]
                if c.shape[-1] < chunk:
                    c = F.pad(c, (0, chunk - c.shape[-1]))
                frames.append(
                    self.loader.model.get_audio_embedding_from_data(c, use_tensor=True)
                )
            return torch.cat(frames, dim=0)

        if self.kind == "dac":
            audio = self._prepare(audio, sample_rate, 1)
            chunk = int(5 * self.sr)
            frames = []
            for off in range(0, audio.shape[-1], chunk):
                c = audio[..., off:off + chunk]
                if c.shape[-1] < chunk:
                    c = F.pad(c, (0, chunk - c.shape[-1]))
                emb = self.model.encoder(c)
                frames.append(emb.transpose(1, 2).reshape(-1, emb.shape[1]))
            return torch.cat(frames, dim=0)

        if self.kind == "cdpam":
            audio = self._prepare(audio, sample_rate, 1)[:, 0]
            chunk = self.sr
            frames = []
            for off in range(0, audio.shape[-1], chunk):
                c = audio[:, off:off + chunk]
                if c.shape[-1] < chunk:
                    c = F.pad(c, (0, chunk - c.shape[-1]))
                _, acoustic, content = self.model.base_encoder.forward(c.unsqueeze(1))
                emb = acoustic if self.loader.mode == "acoustic" else content
                frames.append(F.normalize(emb, dim=1))
            return torch.cat(frames, dim=0)

        raise ValueError(f"Unsupported embedding kind: {self.kind}")