"""Live transcription of audio that arrives in chunks (microphone, a call, a broadcast).

The encoder is a full-context model, so this is the standard "live captions" scheme for such models rather than a
low-latency streaming decoder: while a stretch of speech is open, a cheap CTC partial of it is emitted every
``partial_every_s``; as soon as it ends (a pause of ``end_pause_s``, or ``max_window_s`` of speech, cut at the best
pause) the stretch is decoded properly -- beam search, word list, timestamps, all the offline guards -- and
committed as a final segment that will not change any more.

    session = tr.stream(hotwords=["Bab el-Mandeb"])
    for chunk in microphone():                 # float32 mono 16 kHz, any chunk size
        for ev in session.feed(chunk):
            if ev["type"] == "partial": print("\\r" + ev["text"], end="")
            else:                       print("\\r" + ev["segment"]["text"])
    for ev in session.finish(): ...

Events: {"type": "partial", "start": s, "text": str} (best guess for the open stretch, replaced by the next event)
        {"type": "final", "segment": {start, end, text, language, words}} (times in seconds from the stream start).
Latency: partials lag by partial_every_s + encoder time of the open stretch; finals by end_pause_s + decode time.
"""
from __future__ import annotations

import numpy as np
import torch

from .decode import ctc_greedy, non_speech_window
from .inference import SR, speech_regions, split_windows


class StreamingSession:
    def __init__(self, tr, mode: str = "beam", beam: int = 4, hotwords: list[str] | None = None, hotword_weight: float = 1.0,
                 min_conf: float = 0.72, partial_every_s: float = 1.0, end_pause_s: float = 0.8, max_window_s: float = 25.0):
        self.tr = tr
        self.opts = dict(mode=mode, beam=beam, hotwords=hotwords, hotword_weight=hotword_weight, min_conf=min_conf)
        self.partial_every, self.end_pause, self.max_window = partial_every_s, end_pause_s, max_window_s
        self.buf = np.zeros(0, dtype=np.float32)     # audio not yet committed
        self.offset = 0                              # samples of the stream before buf[0]
        self._since_partial = 0
        self._last_partial = ""
        self.segments: list[dict] = []               # all committed segments

    # ---- public
    def feed(self, chunk: np.ndarray) -> list[dict]:
        """Add audio (float32 mono 16 kHz); returns the events it produced."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.buf = np.concatenate([self.buf, chunk]); self._since_partial += len(chunk)
        events: list[dict] = []
        regions = speech_regions(self.buf)
        if not regions:                                              # no speech yet: keep only a short tail
            if len(self.buf) > 2 * SR:
                self._drop(len(self.buf) - SR)
            return events
        first_start, last_end = regions[0][0], regions[-1][1]
        trailing_silence = (len(self.buf) - last_end) / SR
        open_len = (len(self.buf) - first_start) / SR
        if trailing_silence >= self.end_pause:                       # the stretch ended: finalise everything up to it
            events += self._finalise(split_windows(self.buf[: last_end + int(0.15 * SR)]), keep_from=last_end)
        elif open_len >= self.max_window:                            # too long: commit all windows but the open last one
            wins = split_windows(self.buf)
            if len(wins) >= 2:
                cut = wins[-2][1]
                events += self._finalise(wins[:-1], keep_from=cut)
            else:                                                    # one region longer than max_window: hard cut
                events += self._finalise(wins, keep_from=wins[0][1])
        elif self._since_partial >= self.partial_every * SR:         # otherwise a cheap partial of the open stretch
            self._since_partial = 0
            ev = self._partial(first_start)
            if ev:
                events.append(ev)
        return events

    def finish(self) -> list[dict]:
        """Flush: decode whatever is still open. Call once at the end of the stream."""
        events = self._finalise(split_windows(self.buf), keep_from=len(self.buf)) if len(self.buf) else []
        self.buf = np.zeros(0, dtype=np.float32)
        return events

    @property
    def text(self) -> str:
        return " ".join(s["text"] for s in self.segments)

    # ---- internals
    def _drop(self, n: int) -> None:
        self.buf = self.buf[n:]; self.offset += n

    @torch.no_grad()
    def _partial(self, start: int) -> dict | None:
        tr = self.tr
        x = torch.from_numpy(self.buf[start:])[None].to(tr.device); xl = torch.tensor([len(self.buf) - start], device=tr.device)
        with tr._autocast():
            enc, el = tr.model.encode(x, xl)
        lp = tr.model.ctc(enc.float())
        ids = ctc_greedy(lp, el)[0]
        min_conf = self.opts["min_conf"]
        if len(ids) <= 1 or non_speech_window(lp, int(el[0]), (len(self.buf) - start) / SR, min_conf, n_ctc=len(ids)):   # same gates as the finals
            return None
        text = tr.tok.decode(ids)
        if not text or text == self._last_partial:
            return None
        self._last_partial = text
        return {"type": "partial", "start": (self.offset + start) / SR, "text": text}

    def _finalise(self, wins: list[tuple], keep_from: int) -> list[dict]:
        events: list[dict] = []
        if wins:
            segs, _, _ = self.tr.decode_windows(self.buf, wins, **self.opts)
            for seg in segs:
                seg["start"] = round(seg["start"] + self.offset / SR, 2); seg["end"] = round(seg["end"] + self.offset / SR, 2)
                for w in seg["words"]:
                    w["start"] = round(w["start"] + self.offset / SR, 2); w["end"] = round(w["end"] + self.offset / SR, 2)
                self.segments.append(seg); events.append({"type": "final", "segment": seg})
        self._drop(min(keep_from, len(self.buf))); self._since_partial = 0; self._last_partial = ""
        return events
