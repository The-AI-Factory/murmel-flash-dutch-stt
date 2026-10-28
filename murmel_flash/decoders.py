"""The two heads on top of the encoder: the CTC head (one linear layer) and the Transformer attention decoder.

The decoder has two entry points: ``forward`` scores a whole token sequence at once (used to score prefixes), and
``step`` produces the next-token logits for a batch of hypotheses incrementally, with cached keys/values -- that is
what beam search uses.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, enc: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> log-probabilities (B, T, V), fp32."""
        return F.log_softmax(self.proj(self.drop(enc)).float(), dim=-1)


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).float()[:, None]
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def _heads(self, x, n):
        B, T, _ = x.shape
        return x.view(B, T, n, self.h, self.dh).permute(2, 0, 3, 1, 4)

    def forward(self, y: torch.Tensor, mem: torch.Tensor, mem_valid: torch.Tensor) -> torch.Tensor:
        B, S, D = y.shape
        q, k, v = self._heads(self.self_qkv(self.ln1(y)), 3)
        causal = torch.ones(S, S, dtype=torch.bool, device=y.device).tril()[None, None]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
        y = y + self.drop(self.self_out(a.transpose(1, 2).reshape(B, S, D)))
        q = self._heads(self.cross_q(self.ln2(y)), 1)[0]
        k, v = self._heads(self.cross_kv(mem), 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mem_valid[:, None, None, :])
        y = y + self.drop(self.cross_out(a.transpose(1, 2).reshape(B, S, D)))
        return y + self.drop(self.ff(self.ln3(y)))

    def memory_kv(self, mem: torch.Tensor):
        """Cross-attention keys/values of the encoder memory: computed once per utterance."""
        return self._heads(self.cross_kv(mem), 2)

    def step(self, y: torch.Tensor, cache: dict, mem_k: torch.Tensor, mem_v: torch.Tensor, mem_valid: torch.Tensor) -> torch.Tensor:
        """One new position y (N, 1, D). cache['k'], cache['v'] hold the prefix keys/values (N, H, t, dh) and are
        extended in place; mem_k / mem_v (N, H, T, dh) are the utterance memory keys/values."""
        B, S, D = y.shape
        q, k, v = self._heads(self.self_qkv(self.ln1(y)), 3)
        if cache.get("k") is not None:
            k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
        a = F.scaled_dot_product_attention(q, k, v)
        y = y + self.self_out(a.transpose(1, 2).reshape(B, S, D))
        q = self._heads(self.cross_q(self.ln2(y)), 1)[0]
        a = F.scaled_dot_product_attention(q, mem_k, mem_v, attn_mask=mem_valid[:, None, None, :])
        y = y + self.cross_out(a.transpose(1, 2).reshape(B, S, D))
        return y + self.ff(self.ln3(y))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.register_buffer("pos", sinusoidal_positions(max_len, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight                     # tied input/output embeddings

    def forward(self, tokens: torch.Tensor, mem: torch.Tensor, mem_valid: torch.Tensor) -> torch.Tensor:
        """tokens (B, S) decoder input ids -> logits (B, S, V)."""
        y = self.drop(self.embed(tokens) * math.sqrt(self.d_model) + self.pos[: tokens.shape[1]][None])
        for layer in self.layers:
            y = layer(y, mem, mem_valid)
        return self.out(self.ln(y))

    # ---- incremental decoding ----------------------------------------------------------------------------
    def memory_cache(self, mem: torch.Tensor):
        return [layer.memory_kv(mem) for layer in self.layers]

    def new_cache(self):
        return [{} for _ in self.layers]

    @staticmethod
    def reorder_cache(caches, parents: torch.Tensor) -> None:
        """Keep the cached prefixes of the surviving hypotheses (beam search reorders every step)."""
        for c in caches:
            if c.get("k") is not None:
                c["k"], c["v"] = c["k"][parents], c["v"][parents]

    def step(self, last_tokens: torch.Tensor, pos: int, caches, mem_cache, utt: torch.Tensor, mem_valid: torch.Tensor) -> torch.Tensor:
        """Next-token logits (N, V) for N hypotheses whose last token is last_tokens (N,) at position pos;
        utt (N,) maps each hypothesis to its utterance in mem_cache / mem_valid (B, T)."""
        y = self.embed(last_tokens)[:, None] * math.sqrt(self.d_model) + self.pos[pos][None, None]
        mv = mem_valid[utt]
        for layer, cache, (mk, mvv) in zip(self.layers, caches, mem_cache):
            y = layer.step(y, cache, mk[utt], mvv[utt], mv)
        return self.out(self.ln(y[:, -1]))
