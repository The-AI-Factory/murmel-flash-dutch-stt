"""Transcribe audio of any length with a murmel-flash model.

    from murmel_flash import Transcriber, load_audio
    tr = Transcriber.from_pretrained()          # murmel-flash-m-v5
    result = tr.transcribe(load_audio("interview.mp3"), hotwords=["Bab el-Mandeb"])

Pipeline: pause-based windows (<= 25 s) -> encoder + CTC head -> gates (no words / music) -> attention beam search
over all windows jointly -> CTC referee against hallucinations -> word timestamps -> sentence continuity across
windows. See README.md, "How inference works".
"""
from __future__ import annotations

import contextlib
import io
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch

from .config import model_config_from_yaml
from .decode import attention_beam, ctc_greedy, has_loop, non_speech_window
from .model import HybridASR
from .timing import align_times, ctc_token_times, words_from_tokens
from .tokenizer import Tokenizer

SR = 16000

# ---------------------------------------------------------------------------------------------------- audio


def _ffmpeg_decode(data: bytes, suffix: str) -> np.ndarray:
    # decode from a file, not a pipe: MP4/MOV keep their index at the end and yield nothing from a non-seekable pipe
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False) as f:
        f.write(data); tmp = f.name
    try:
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", tmp, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                           capture_output=True)
    finally:
        os.unlink(tmp)
    if p.returncode != 0 or len(p.stdout) < 4:
        raise RuntimeError(f"could not decode audio: {p.stderr.decode()[:200] or 'no audio stream'} (is ffmpeg installed?)")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def resample(wave: np.ndarray, sr: int) -> np.ndarray:
    """float32 mono at sr -> 16 kHz: windowed-sinc polyphase resampling (Hann window, 6 zero crossings, 0.99 rolloff),
    in chunks so hour-long files stay within memory. Keeps files soundfile can read independent of ffmpeg."""
    if sr == SR:
        return wave.astype(np.float32)
    g = math.gcd(sr, SR); orig, new = sr // g, SR // g
    zeros, base = 6, min(orig, new) * 0.99
    width = math.ceil(zeros * orig / base)
    t = torch.arange(0, -new, -1, dtype=torch.float64)[:, None] / new + torch.arange(-width, width + orig, dtype=torch.float64)[None] / orig
    t = (t * base).clamp(-zeros, zeros)
    window = torch.cos(t * math.pi / zeros / 2) ** 2
    t = t * math.pi
    kernels = (torch.where(t == 0, torch.ones_like(t), t.sin() / t) * window * base / orig).float()[:, None]   # (new, 1, K)
    x = torch.nn.functional.pad(torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None, None], (width, width + orig))
    n_frames, k, step = (x.shape[-1] - kernels.shape[-1]) // orig + 1, kernels.shape[-1], 20000   # frame f reads x[f*orig : f*orig+k]
    out = [torch.nn.functional.conv1d(x[..., f * orig : (min(f + step, n_frames) - 1) * orig + k], kernels, stride=orig)[0].T.reshape(-1)
           for f in range(0, n_frames, step)]
    return torch.cat(out)[: math.ceil(new * len(wave) / orig)].numpy()


def load_audio(src: str | Path | bytes, name: str = "") -> np.ndarray:
    """Any file soundfile or ffmpeg understands (wav, flac, aiff, ogg, mp3, m4a, mp4, webm, ...) -> float32 mono 16 kHz.
    Files soundfile reads are decoded and resampled without ffmpeg."""
    import soundfile as sf
    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    suffix = Path(name).suffix or (Path(src).suffix if isinstance(src, (str, Path)) else "")
    try:
        wave, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        return _ffmpeg_decode(data, suffix)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if not len(wave):                                   # header without a frame count (e.g. streamed AIFF): let ffmpeg try
        return _ffmpeg_decode(data, suffix)
    return resample(wave, sr)


# ---------------------------------------------------------------------------------------------------- windows


def speech_regions(wave: np.ndarray, frame_s: float = 0.02, margin_db: float = 12.0, min_speech_s: float = 0.25,
                   min_gap_s: float = 0.35) -> list[tuple]:
    """Energy-based voice activity: frames louder than the noise floor + margin; gaps shorter than min_gap merged.
    Returns (start, end) in samples."""
    hop = int(frame_s * SR); n = len(wave) // hop
    if n == 0:
        return []
    e = np.sqrt((wave[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-8
    db = 20 * np.log10(e)
    # quiet recordings (soft speakers, low-gain microphones) are treated like the same recording turned up: lift the
    # loudest stretches to normal speech level before the absolute limits below apply. Only when something stands out
    # from the floor (steady noise and hum have ~no dynamic range and stay as they are); never attenuates, at most +50 dB
    lo, hi = np.percentile(db, [15, 99.5])
    if hi - lo >= 10.0:
        db += float(np.clip(-12.0 - hi, 0.0, 50.0))
    thr = max(min(np.percentile(db, 15) + margin_db, -35.0), -55.0)   # never above -35 dBFS (music beds), never below -55
    act = db > thr
    regions, i = [], 0
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]:
                j += 1
            regions.append([i * hop, j * hop]); i = j
        else:
            i += 1
    merged: list[list] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_gap_s * SR:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_speech_s * SR]


def split_windows(wave: np.ndarray, max_s: float = 25.0, pad_s: float = 0.15) -> list[tuple]:
    """Windows of <= max_s that end at pauses; a full window is cut at the longest pause in its second half.
    Stretches without speech are not decoded at all. Returns (start_sample, end_sample) pairs."""
    regions = speech_regions(wave)
    if not regions:
        return []
    out, cur = [], [regions[0]]
    for a, b in regions[1:]:
        if b - cur[0][0] <= max_s * SR and a - cur[-1][1] < 2.0 * SR:
            cur.append((a, b)); continue
        if a - cur[-1][1] < 2.0 * SR and len(cur) > 1:
            span = cur[-1][1] - cur[0][0]
            gaps = [(cur[i][0] - cur[i - 1][1], i) for i in range(1, len(cur)) if cur[i][0] - cur[0][0] >= 0.5 * span]
            if gaps:
                g, i = max(gaps)
                if g >= 0.5 * SR or g >= 1.5 * (a - cur[-1][1]):
                    out.append((cur[0][0], cur[i - 1][1])); cur = cur[i:] + [(a, b)]; continue
        out.append((cur[0][0], cur[-1][1])); cur = [(a, b)]
    out.append((cur[0][0], cur[-1][1]))
    final = []
    for a, b in out:
        while b - a > max_s * SR:                                    # one region longer than max_s: hard cuts
            final.append((a, a + int(max_s * SR))); a += int(max_s * SR)
        final.append((a, b))
    pad = int(pad_s * SR)
    return [(max(0, a - pad), min(len(wave), b + pad)) for a, b in final]


_STOP = {"de", "het", "een", "en", "maar", "dat", "die", "dan", "dus", "ook", "of", "om", "op", "in", "is", "er", "we", "ik", "je", "ze",
         "the", "and", "but", "that", "then", "so", "also", "or", "to", "it", "i", "you", "they", "a", "an"}


def join_windows(segments: list[dict], join_gap_s: float, is_sentence_end: Callable[[float], bool]) -> None:
    """Sentence continuity across windows, in place. Every window ends with a period (clip-level training) and the
    next starts with a capital; when the pause is short and the CTC head sees no sentence end at the boundary, the
    period becomes a comma and the next word is lowercased -- unless it looks like a proper noun."""
    if len(segments) < 2:
        return
    mid_caps = set()
    for seg in segments:
        toks = seg["text"].split()
        for i in range(1, len(toks)):
            w = toks[i].strip(",;:\"'()")
            if w[:1].isupper() and not toks[i - 1].endswith((".", "?", "!", ":")):
                mid_caps.add(w)
    for prev, nxt in zip(segments, segments[1:]):
        pt, nt = prev["text"].rstrip(), nxt["text"].lstrip()
        if nxt["start"] - prev["end"] > join_gap_s or not pt.endswith(".") or pt.endswith("...") or not nt or not nt[0].isupper():
            continue
        if is_sentence_end((prev["end"] + nxt["start"]) / 2):
            continue
        first = nt.split()[0].strip(",;:\"'()")
        if first in mid_caps or len(first) == 1 or first.isupper() or first == "Ik":
            continue
        if not (first.lower() in _STOP or first[1:].islower()):
            continue
        prev["text"] = pt[:-1] + ","
        nxt["text"] = nt[0].lower() + nt[1:]
        if nxt.get("words") and nxt["words"][0]["word"][:1] == nt[0]:
            nxt["words"][0]["word"] = nt[0].lower() + nxt["words"][0]["word"][1:]


# ---------------------------------------------------------------------------------------------------- model


# The model this package ships with. NOTE for the release: the final public repo will be murmel-flash-m-v5; that name is
# currently taken by the training run, so the release-layout repo is used until the repos are renamed.
DEFAULT_MODEL = "theaifactorybv/murmel-flash-m-v5"


class Transcriber:
    """A loaded model + tokenizer. Thread-safe for concurrent transcribe() calls."""

    def __init__(self, model_dir: str | Path, device: str | None = None):
        d = Path(model_dir)
        self.cfg = model_config_from_yaml(d / "config.yaml")
        tok_path = next(p for p in (d / "tokenizer.model", d / "spm.model") if p.exists())
        self.tok = Tokenizer(tok_path)
        self.model = HybridASR(self.cfg)
        self.model.load_state_dict(_load_weights(d))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.precision = "auto"                  # "fp32" disables mixed precision on CUDA
        self._boosters: dict[tuple, object] = {}
        # v4 tokenizers carry style tags: always decode <verbatim>, the style the output is documented and evaluated in
        # (fillers and repetitions kept, as spoken). v3 tokenizers have no style tags and decode unchanged.
        vid = self.tok.style_ids.get("verbatim")
        self.force_tokens = {1: vid} if vid is not None else None

    @classmethod
    def from_pretrained(cls, repo_or_dir: str = DEFAULT_MODEL, device: str | None = None,
                        token: str | None = None, revision: str | None = None) -> Transcriber:
        """Load from a local directory or a Hugging Face model repo (downloaded once, then cached)."""
        return cls(_resolve(repo_or_dir, token, revision), device)

    # ---- helpers
    def _autocast(self):
        if self.device.type == "cuda" and self.precision != "fp32":
            ampere = torch.cuda.get_device_capability(self.device)[0] >= 8
            return torch.autocast("cuda", dtype=torch.bfloat16 if ampere else torch.float16)
        return contextlib.nullcontext()

    def booster(self, hotwords: list[str] | None, weight: float):
        from .biasing import WordBooster
        key = (tuple(hotwords or ()), weight)
        if not key[0] or weight <= 0:
            return None
        if key not in self._boosters:
            self._boosters = {key: WordBooster(hotwords, self.tok, weight)}      # keep the latest list only
        return self._boosters[key]

    @torch.no_grad()
    def encode_windows(self, wave: np.ndarray, wins: list[tuple], batch_s: float) -> Iterator[tuple]:
        """Encoder + CTC head over windows, several per call on a GPU (padded to the longest; the encoder masks
        padding). Yields (start, end, enc (1,T,d) fp32, log_probs (1,T,V), enc_len (1,)) per window, in order."""
        i = 0
        while i < len(wins):
            batch = [wins[i]]; longest = wins[i][1] - wins[i][0]; i += 1
            while i < len(wins) and batch_s > 0:
                L = max(longest, wins[i][1] - wins[i][0])
                if (len(batch) + 1) * L > batch_s * SR:
                    break
                batch.append(wins[i]); longest = L; i += 1
            x = torch.zeros(len(batch), longest)
            for j, (a, b) in enumerate(batch):
                x[j, : b - a] = torch.from_numpy(wave[a:b])
            xl = torch.tensor([b - a for a, b in batch])
            with self._autocast():
                enc, el = self.model.encode(x.to(self.device), xl.to(self.device))
            enc = enc.float(); lp = self.model.ctc(enc)
            for j, (a, b) in enumerate(batch):
                L = int(el[j])
                yield a, b, enc[j : j + 1, :L], lp[j : j + 1, :L], el[j : j + 1]

    @torch.no_grad()
    def _sentence_end_at(self, wave: np.ndarray, t: float, span_s: float = 6.0, tol_s: float = 0.7) -> bool:
        """Does the CTC head put sentence punctuation within tol_s of time t, given span_s of audio on both sides?"""
        a, b = max(0, int((t - span_s) * SR)), min(len(wave), int((t + span_s) * SR))
        x = torch.from_numpy(wave[a:b])[None].to(self.device); xl = torch.tensor([b - a], device=self.device)
        with self._autocast():
            enc, el = self.model.encode(x, xl)
        lp = self.model.ctc(enc.float())
        for tid, _, e0 in ctc_token_times(lp, int(el[0])):
            if self.tok.piece(tid).rstrip("▁").endswith((".", "?", "!")) and abs(a / SR + e0 - t) <= tol_s:
                return True
        return False

    @torch.no_grad()
    def decode_windows(self, wave: np.ndarray, wins: list[tuple], mode: str = "beam", beam: int = 4, hotwords: list[str] | None = None,
                       hotword_weight: float = 1.0, min_conf: float = 0.72, batch_s: float | None = None,
                       dense_tps: float = 2.0) -> tuple:
        """Decode the given (start, end) sample windows of wave -> (segments, n_fallback, n_skipped). Used by transcribe()
        and by the streaming session; see transcribe() for the parameters."""
        if batch_s is None:
            batch_s = 300.0 if self.device.type == "cuda" else 0.0
        pending, n_skipped = [], 0
        for a, b, enc, lp, el in self.encode_windows(wave, wins, batch_s):
            ids_ctc = ctc_greedy(lp, el)[0]
            if len(ids_ctc) <= 1 or (len(ids_ctc) - 1) / ((b - a) / SR) < 0.25:      # no words, or a stray word
                continue
            if non_speech_window(lp, int(el[0]), (b - a) / SR, min_conf, dense_tps, len(ids_ctc)):   # music, jingles
                n_skipped += 1
                continue
            cap = min(self.cfg.decoder.max_len - 2, int(len(ids_ctc) * 1.6) + 8)
            pending.append({"a": a, "b": b, "enc": enc, "lp": lp, "el": el, "ctc": ids_ctc, "cap": cap, "att": None})
        if mode == "beam" and pending:
            group = 32 if self.device.type == "cuda" else 1
            booster = self.booster(hotwords, hotword_weight)
            for g0 in range(0, len(pending), group):
                grp = pending[g0 : g0 + group]
                Tm = max(int(w["el"][0]) for w in grp)
                encb = torch.zeros(len(grp), Tm, grp[0]["enc"].shape[-1], device=self.device)
                for j, w in enumerate(grp):
                    encb[j, : int(w["el"][0])] = w["enc"][0]
                elb = torch.tensor([int(w["el"][0]) for w in grp], device=self.device)
                for w, ids in zip(grp, attention_beam(self.model, encb, elb, beam=beam, max_len=[w["cap"] for w in grp], booster=booster,
                                                      force_tokens=self.force_tokens)):
                    w["att"] = ids
        segments, n_fallback = [], 0
        for w in pending:
            ids = w["ctc"]
            if w["att"] is not None:                                                  # CTC referee
                if len(w["att"]) <= len(w["ctc"]) * 1.5 + 4 and not has_loop(w["att"]):
                    ids = w["att"]
                else:
                    n_fallback += 1
            text = self.tok.decode(ids)
            if text:
                words = words_from_tokens(align_times(list(ids), ctc_token_times(w["lp"], int(w["el"][0]))), self.tok, w["a"] / SR)
                segments.append({"start": w["a"] / SR, "end": w["b"] / SR, "text": text, "language": self.tok.lang_of(ids) or "?", "words": words})
        return segments, n_fallback, n_skipped

    def stream(self, **kw):
        """A streaming session: feed audio chunks, get partial and final segments. See murmel_flash.streaming."""
        from .streaming import StreamingSession
        return StreamingSession(self, **kw)

    # ---- main entry point
    @torch.no_grad()
    def transcribe(self, wave: np.ndarray, mode: str = "beam", beam: int = 4, hotwords: list[str] | None = None,
                   hotword_weight: float = 1.0, min_conf: float = 0.72, join_gap_s: float = 1.0,
                   batch_s: float | None = None, dense_tps: float = 2.0) -> dict:
        """wave: float32 mono 16 kHz (see load_audio).
        mode: "beam" (attention decoder, best) or "ctc" (CTC only: ~3x faster, ~2 WER points worse).
        hotwords / hotword_weight: domain word list (names, jargon) boosted in beam search; 1.0 is the measured optimum.
        min_conf / dense_tps: a window is skipped as music or jingle when its CTC confidence is below min_conf AND CTC
            hears fewer than dense_tps tokens per second; unsure but dense windows (phone lines, old recordings) are
            kept. min_conf 0 turns the gate off.
        join_gap_s: windows meeting at a shorter pause are joined into one sentence when the CTC head agrees (0 = off).
        batch_s: padded seconds of audio per encoder call (default 300 on GPU, one window at a time on CPU).

        Returns text, segments (per window: start, end, text, language, words), words, language, duration, seconds,
        rtf, windows, ctc_fallbacks, skipped_windows."""
        t0 = time.time()
        segments, n_fallback, n_skipped = self.decode_windows(wave, split_windows(wave), mode, beam, hotwords, hotword_weight, min_conf, batch_s,
                                                                dense_tps)
        if join_gap_s > 0:
            join_windows(segments, join_gap_s, lambda t: self._sentence_end_at(wave, t))
        dur, elapsed = len(wave) / SR, time.time() - t0
        langs = [s["language"] for s in segments]
        return {"text": " ".join(s["text"] for s in segments), "segments": segments, "words": [w for s in segments for w in s["words"]],
                "language": max(set(langs), key=langs.count) if langs else "?", "duration": dur, "seconds": elapsed,
                "rtf": elapsed / max(dur, 1e-6), "windows": len(segments), "ctc_fallbacks": n_fallback, "skipped_windows": n_skipped}


# ---------------------------------------------------------------------------------------------------- loading


def _resolve(repo_or_dir: str, token: str | None, revision: str | None) -> Path:
    if Path(repo_or_dir).is_dir():
        return Path(repo_or_dir)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    try:
        return Path(snapshot_download(repo_or_dir, repo_type="model", token=token or os.environ.get("HF_TOKEN"), revision=revision,
                                      allow_patterns=["*.yaml", "*.json", "*.model", "*.safetensors", "best.pt"]))
    except GatedRepoError as e:
        raise RuntimeError(
            f"{repo_or_dir} is gated: request access and accept the licence at https://huggingface.co/{repo_or_dir}, "
            "wait for approval, then run `hf auth login` (or set HF_TOKEN)") from e


def _load_weights(d: Path) -> dict[str, torch.Tensor]:
    """model.safetensors (release format) or best.pt (trainer checkpoint; training-only tensors dropped)."""
    if (d / "model.safetensors").exists():
        from safetensors.torch import load_file
        sd = {k: v.float() for k, v in load_file(str(d / "model.safetensors")).items()}   # always fp32 in memory
        sd.setdefault("decoder.out.weight", sd["decoder.embed.weight"])       # tied embedding (safetensors stores it once)
        # v4 checkpoints also carry the trained context module; this package biases word lists with the search-time
        # trie only (biasing.py), so those tensors are not loaded
        return {k: v for k, v in sd.items() if "ctx" not in k}
    sd = torch.load(d / "best.pt", map_location="cpu")
    return {k: v for k, v in sd.items() if "ctx" not in k}          # context-module tensors of v3 trainer checkpoints
