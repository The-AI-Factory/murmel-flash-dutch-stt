"""Log-mel frontend inside the model: waveform (B, T) at 16 kHz -> normalised features (B, F, n_mels)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _hz_to_mel(f: np.ndarray) -> np.ndarray:  # Slaney scale
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, math.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular filters, area-normalised (Slaney)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)
    hz_pts = _mel_to_hz(np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2))
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    fb = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        fb[i] = np.maximum(0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    fb *= (2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels]))[:, None]
    return fb.astype(np.float32)


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80, f_min=0.0, f_max=8000.0):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length, self.n_mels = n_fft, win_length, hop_length, n_mels
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("mel", torch.from_numpy(mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)), persistent=False)

    def forward(self, wave: torch.Tensor, wave_lengths: torch.Tensor):
        with torch.autocast(device_type=wave.device.type, enabled=False):
            wave = wave.float() * (torch.arange(wave.shape[1], device=wave.device)[None] < wave_lengths[:, None])   # zero beyond length
            spec = torch.stft(wave, self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window,
                              center=True, pad_mode="constant", return_complex=True)          # (B, n_fft/2+1, F)
            mel = torch.matmul(self.mel, spec.real.pow(2) + spec.imag.pow(2))                 # (B, n_mels, F)
            feats = torch.log(mel + 1e-6).transpose(1, 2)                                   # (B, F, n_mels)
            lengths = (wave_lengths // self.hop_length + 1).clamp(max=feats.shape[1])
            # per-utterance mean/variance normalisation over the valid frames
            m = (torch.arange(feats.shape[1], device=feats.device)[None, :] < lengths[:, None])[..., None].float()
            n = m.sum(dim=1).clamp(min=1.0)
            mean = (feats * m).sum(dim=1, keepdim=True) / n[:, None]
            var = ((feats - mean) ** 2 * m).sum(dim=1, keepdim=True) / n[:, None]
            feats = (feats - mean) / (var.sqrt() + 1e-5) * m
        return feats, lengths
