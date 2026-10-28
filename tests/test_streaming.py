"""Streaming session mechanics on the tiny random model, and equivalence with offline decoding on real weights."""
import os

import numpy as np
import pytest
import torch
from huggingface_hub import get_token

from murmel_flash.config import DecoderConfig, EncoderConfig, ModelConfig
from murmel_flash.inference import SR, Transcriber
from murmel_flash.model import HybridASR
from murmel_flash.streaming import StreamingSession


class _TinyTok:
    """Vocabulary of 64 fake pieces; every id decodes to a word so the gates let random output through."""
    _special = {0, 1, 2, 3}
    lang_ids = {"nl": 4, "en": 5}

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids if i not in self._special)

    def lang_of(self, ids):
        return "nl"

    def piece(self, i):
        return f"▁w{i}"


def _tiny_transcriber():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, encoder=EncoderConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, subsampling_channels=8),
                      decoder=DecoderConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, max_len=64))
    tr = Transcriber.__new__(Transcriber)
    tr.cfg, tr.tok, tr.model, tr.device, tr.precision, tr._boosters = cfg, _TinyTok(), HybridASR(cfg).eval(), torch.device("cpu"), "fp32", {}
    return tr


def test_session_emits_finals_at_pauses_and_on_finish():
    tr = _tiny_transcriber()
    rng = np.random.default_rng(0)
    wave = np.zeros(20 * SR, dtype=np.float32)
    for s in (1, 8, 15):                                                    # three 3-s bursts, 4-s pauses between
        wave[s * SR : (s + 3) * SR] = rng.standard_normal(3 * SR).astype(np.float32) * 0.1
    session = StreamingSession(tr, mode="ctc", min_conf=0.0, partial_every_s=0.5, end_pause_s=0.8)
    events = []
    for i in range(0, len(wave), SR // 4):                                   # 250 ms chunks
        events += session.feed(wave[i : i + SR // 4])
    events += session.finish()
    finals = [e["segment"] for e in events if e["type"] == "final"]
    assert len(finals) == 3, [e["type"] for e in events]
    for seg, s in zip(finals, (1, 8, 15)):
        assert abs(seg["start"] - s) < 0.4 and abs(seg["end"] - (s + 3)) < 0.4 and seg["text"]
        assert seg["words"] and seg["words"][0]["start"] >= seg["start"] - 0.01
    assert any(e["type"] == "partial" for e in events)
    assert session.text == " ".join(seg["text"] for seg in finals)


REPO = os.environ.get("MURMEL_TEST_MODEL", "theaifactorybv/murmel-flash-m-v5")


@pytest.mark.skipif(not (os.path.isdir(REPO) or get_token()), reason="no released weights available")
def test_single_feed_equals_offline():
    """The whole clip fed at once and finished must give exactly the offline segments (join step aside)."""
    import shutil
    import subprocess
    import tempfile

    from murmel_flash import load_audio
    if not shutil.which("say"):
        pytest.skip("needs macOS TTS for a synthetic clip")
    path = os.path.join(tempfile.mkdtemp(), "t.wav")       # 22.05 kHz: also exercises resampling
    subprocess.run(["say", "-v", "Xander", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", path, "Dit is een korte test. En dit is nog een zin, met een komma erin."], check=True)
    tr = Transcriber.from_pretrained(REPO, device="cpu"); wave = load_audio(path)
    offline = tr.transcribe(wave, join_gap_s=0)["segments"]
    session = tr.stream(); events = session.feed(wave) + session.finish()
    finals = [e["segment"] for e in events if e["type"] == "final"]
    assert [s["text"] for s in finals] == [s["text"] for s in offline]
