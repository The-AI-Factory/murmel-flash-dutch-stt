"""HybridASR: log-mel frontend + FastConformer encoder + CTC head + attention decoder (inference)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .decoders import CTCHead, TransformerDecoder
from .encoder import ConformerEncoder
from .frontend import LogMelFrontend


class HybridASR(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        f, e, d = cfg.frontend, cfg.encoder, cfg.decoder
        self.frontend = LogMelFrontend(f.sample_rate, f.n_fft, f.win_length, f.hop_length, f.n_mels, f.f_min, f.f_max)
        self.encoder = ConformerEncoder(f.n_mels, e.d_model, e.n_layers, e.n_heads, e.d_ff, e.conv_kernel,
                                        e.subsampling_channels, e.dropout, e.rope_base, e.xscale)
        self.ctc = CTCHead(e.d_model, cfg.vocab_size, e.dropout)
        self.decoder = TransformerDecoder(cfg.vocab_size, d.d_model, d.n_layers, d.n_heads, d.d_ff, d.dropout, d.max_len)

    def encode(self, wave: torch.Tensor, wave_lengths: torch.Tensor):
        """wave (B, T) float32 16 kHz, lengths (B,) -> encoded (B, T', d_model), enc_lengths (B,)."""
        feats, feat_lengths = self.frontend(wave, wave_lengths)
        return self.encoder(feats, feat_lengths)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
