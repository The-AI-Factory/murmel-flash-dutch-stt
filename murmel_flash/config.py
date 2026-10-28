"""Model configuration: three small dataclasses that map 1:1 onto the module constructors.

`model_config_from_yaml` reads the ``model:`` section of a checkpoint's ``config.yaml`` and ignores keys it does not
know (training-only settings such as SpecAugment), so config files written by the trainer load unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FrontendConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400          # 25 ms
    hop_length: int = 160          # 10 ms
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8000.0


@dataclass
class EncoderConfig:
    d_model: int = 512
    n_layers: int = 17
    n_heads: int = 8
    d_ff: int = 2048
    conv_kernel: int = 9
    subsampling_channels: int = 256
    dropout: float = 0.1
    rope_base: float = 10000.0
    xscale: bool = False           # multiply the subsampling output by sqrt(d_model)


@dataclass
class DecoderConfig:
    n_layers: int = 4
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 512             # positional table size = longest token sequence


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


def _build(cls, raw: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    kwargs = {}
    for k, v in (raw or {}).items():
        if k not in known:
            continue                                    # training-only or newer key: ignored
        t = known[k].type
        if isinstance(v, dict) and t not in ("int", "float", "bool", "str"):
            v = _build({"FrontendConfig": FrontendConfig, "EncoderConfig": EncoderConfig, "DecoderConfig": DecoderConfig}[t], v)
        kwargs[k] = v
    return cls(**kwargs)


def model_config_from_yaml(path: str | Path) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _build(ModelConfig, raw.get("model", raw))
    if cfg.decoder.d_model != cfg.encoder.d_model:
        raise ValueError("decoder.d_model must equal encoder.d_model")
    return cfg
