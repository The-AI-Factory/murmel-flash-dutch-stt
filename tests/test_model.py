"""Shape and decoding tests on a tiny random model (no weights, no network)."""
import numpy as np
import pytest
import torch

from murmel_flash.config import DecoderConfig, EncoderConfig, ModelConfig
from murmel_flash.decode import attention_beam, ctc_confidence, ctc_greedy, has_loop
from murmel_flash.model import HybridASR
from murmel_flash.timing import align_times, ctc_token_times
from murmel_flash.tokenizer import BOS, EOS, PAD


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, encoder=EncoderConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, subsampling_channels=8),
                      decoder=DecoderConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, max_len=64))
    return HybridASR(cfg).eval()


def test_encode_shapes_and_padding_invariance(model):
    wave = torch.randn(2, 16000)
    enc, el = model.encode(wave, torch.tensor([16000, 8000]))
    assert enc.shape == (2, 101 // 8 + 1, 32) and el.tolist() == [13, 7]
    enc1, el1 = model.encode(wave[1:, :8000], torch.tensor([8000]))         # the short clip alone
    assert torch.allclose(enc[1, :7], enc1[0, :7], atol=1e-4)               # padding changes nothing


def test_incremental_decoder_matches_full_forward(model):
    enc, el = model.encode(torch.randn(1, 16000), torch.tensor([16000]))
    toks = torch.tensor([[BOS, 5, 9, 12, 7]])
    valid = torch.ones(1, enc.shape[1], dtype=torch.bool)
    full = model.decoder(toks, enc, valid)[0]
    caches, mc = model.decoder.new_cache(), model.decoder.memory_cache(enc)
    inc = torch.stack([model.decoder.step(toks[:, t], t, caches, mc, torch.zeros(1, dtype=torch.long), valid)[0] for t in range(5)])
    assert torch.allclose(full, inc, atol=1e-4)


def test_beam_search_batched_equals_single(model):
    enc, el = model.encode(torch.randn(3, 12000), torch.tensor([12000, 9000, 6000]))
    joint = attention_beam(model, enc, el, beam=3, max_len=12)
    single = [attention_beam(model, enc[i : i + 1, : int(el[i])], el[i : i + 1], beam=3, max_len=12)[0] for i in range(3)]
    assert joint == single
    assert all(EOS not in s and PAD not in s and len(s) <= 12 for s in joint)


def test_ctc_helpers():
    lp = torch.full((1, 6, 5), -9.0)
    lp[0, :, 0] = -0.01
    for t, tok in ((2, 3), (3, 3), (5, 2)):                                   # frames 2-3 emit token 3, frame 5 token 2
        lp[0, t, 0], lp[0, t, tok] = -5.0, -0.1
    assert ctc_greedy(lp, torch.tensor([6])) == [[3, 2]]
    assert 0.0 < ctc_confidence(lp, 6) < 1.0
    assert [t[0] for t in ctc_token_times(lp, 6)] == [3, 2]
    assert has_loop([1, 2, 1, 2, 1, 2, 1, 2]) and not has_loop([1, 2, 3, 4])


def test_align_times_interpolates_unmatched_tokens():
    ctc = [(4, 0.0, 0.1), (5, 0.1, 0.2), (7, 0.3, 0.4)]
    timed = align_times([4, 6, 7], ctc)
    assert [t[0] for t in timed] == [4, 6, 7] and timed[0][1] == 0.0 and timed[2][2] == 0.4
    assert timed[0][2] <= timed[1][1] <= timed[1][2] <= timed[2][1]


def test_speech_regions_and_windows():
    from murmel_flash.inference import SR, speech_regions, split_windows
    rng = np.random.default_rng(0)
    wave = np.zeros(60 * SR, dtype=np.float32)
    for s in (2, 12, 30, 45):                                                # four 5-s bursts
        wave[s * SR : (s + 5) * SR] = rng.standard_normal(5 * SR).astype(np.float32) * 0.1
    regions = speech_regions(wave)
    assert len(regions) == 4 and all(abs((b - a) / SR - 5) < 0.3 for a, b in regions)
    wins = split_windows(wave, max_s=25.0)
    assert 2 <= len(wins) <= 4 and all((b - a) / SR <= 25.4 for a, b in wins)


def test_speech_regions_on_quiet_recordings():
    """A soft speaker or low-gain microphone (speech near -70 dBFS) is found like normal speech; noise, hum and silence
    stay speech-free."""
    from murmel_flash.inference import SR, speech_regions
    rng = np.random.default_rng(0)
    wave = np.zeros(60 * SR, dtype=np.float32)
    for s in (2, 12, 30, 45):
        wave[s * SR : (s + 5) * SR] = rng.standard_normal(5 * SR).astype(np.float32) * 0.1
    for gain_db in (0, -40, -50):
        x = wave * 10 ** (gain_db / 20) + rng.standard_normal(len(wave)).astype(np.float32) * 10 ** (-100 / 20)
        regions = speech_regions(x)
        assert len(regions) == 4 and all(abs((b - a) / SR - 5) < 0.3 for a, b in regions), gain_db
    noisy = wave + rng.standard_normal(len(wave)).astype(np.float32) * 10 ** (-30 / 20)          # 10 dB above the noise
    for gain_db in (-30, -50):                                                                  # quiet = the same turned up
        assert speech_regions(noisy * 10 ** (gain_db / 20)) == speech_regions(noisy), gain_db
    t = np.arange(30 * SR) / SR
    for quiet in (np.zeros(30 * SR), rng.standard_normal(30 * SR) * 10 ** (-60 / 20), 10 ** (-50 / 20) * np.sin(2 * np.pi * 50 * t)):
        assert speech_regions(quiet.astype(np.float32)) == []


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_resample_keeps_speech_band_and_length(sr):
    from murmel_flash.inference import SR, resample
    t = np.arange(int(sr * 2.5)) / sr
    y = resample((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    assert y.dtype == np.float32 and len(y) == int(np.ceil(len(t) * SR / sr))
    ty = np.arange(len(y)) / SR; mid = slice(SR // 4, -SR // 4)
    assert np.abs(y[mid] - 0.3 * np.sin(2 * np.pi * 440 * ty[mid])).max() < 0.01
