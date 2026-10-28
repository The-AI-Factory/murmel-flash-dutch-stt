"""FastConformer encoder: 8x depthwise-separable convolutional subsampling + Conformer blocks with rotary attention.

    features (B, F, n_mels), lengths (B,)  ->  encoded (B, T, d_model), enc_lengths (B,)
    T = sub(sub(sub(F))) with sub(n) = (n - 1) // 2 + 1, i.e. 12.5 frames per second of audio.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def subsampled_length(n, times: int = 3):
    for _ in range(times):
        n = (n - 1) // 2 + 1
    return n


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B, max_len) bool, True where a frame is valid."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


class ConvSubsampling8x(nn.Module):
    def __init__(self, n_mels: int, channels: int, d_model: int):
        super().__init__()
        C = channels
        self.conv1 = nn.Conv2d(1, C, kernel_size=3, stride=2, padding=1)
        self.dw2 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C)
        self.pw2 = nn.Conv2d(C, C, kernel_size=1)
        self.dw3 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, groups=C)
        self.pw3 = nn.Conv2d(C, C, kernel_size=1)
        self.out = nn.Linear(C * subsampled_length(n_mels), d_model)

    @staticmethod
    def _mask_time(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # zero frames beyond each utterance so the next stride-2 conv never sees padding (padding-invariant encoder)
        return x * make_pad_mask(lengths, x.shape[2])[:, None, :, None].to(x.dtype)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self._mask_time(x.unsqueeze(1), lengths)                                   # (B, 1, F, n_mels)
        x = F.relu(self.conv1(x)); lengths = subsampled_length(lengths, 1); x = self._mask_time(x, lengths)
        x = F.relu(self.pw2(self.dw2(x))); lengths = subsampled_length(lengths, 1); x = self._mask_time(x, lengths)
        x = F.relu(self.pw3(self.dw3(x))); lengths = subsampled_length(lengths, 1); x = self._mask_time(x, lengths)
        B, C, T, M = x.shape
        return self.out(x.transpose(1, 2).reshape(B, T, C * M)), lengths


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)
        self._cos = self._sin = None
        self._cache_len = 0

    def _tables(self, T: int, device, dtype):
        if self._cos is None or self._cache_len < T or self._cos.device != device:
            freqs = torch.outer(torch.arange(T, device=device).float(), self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos, self._sin, self._cache_len = emb.cos(), emb.sin(), T
        return self._cos[:T].to(dtype), self._sin[:T].to(dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        cos, sin = self._tables(q.shape[2], q.device, q.dtype)
        cos, sin = cos[None, None], sin[None, None]
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RotaryEmbedding):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """LN -> pointwise -> GLU -> depthwise -> LN -> SiLU -> pointwise, padding-safe."""

    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw1(self.ln(x).transpose(1, 2)), dim=1)
        y = self.dw(y * valid[:, None, :].to(y.dtype))
        y = F.silu(self.ln2(y.transpose(1, 2)).transpose(1, 2))
        return self.drop(self.pw2(y).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, conv_kernel, dropout, rope):
        super().__init__()
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.ln_att = nn.LayerNorm(d_model)
        self.att = MultiHeadSelfAttention(d_model, n_heads, rope)
        self.drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.att(self.ln_att(x), valid))
        x = x + self.conv(x, valid)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class ConformerEncoder(nn.Module):
    def __init__(self, n_mels: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, conv_kernel: int,
                 subsampling_channels: int, dropout: float, rope_base: float = 10000.0, xscale: bool = False):
        super().__init__()
        self.subsampling = ConvSubsampling8x(n_mels, subsampling_channels, d_model)
        self.in_drop = nn.Dropout(dropout)
        rope = RotaryEmbedding(d_model // n_heads, rope_base)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, n_heads, d_ff, conv_kernel, dropout, rope) for _ in range(n_layers)])
        self.xscale = math.sqrt(d_model) if xscale else 1.0

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x, lengths = self.subsampling(feats, feat_lengths)
        x = self.in_drop(x * self.xscale)
        valid = make_pad_mask(lengths, x.shape[1])
        x = x * valid[..., None].to(x.dtype)
        for block in self.blocks:
            x = block(x, valid)
        return x * valid[..., None].to(x.dtype), lengths
