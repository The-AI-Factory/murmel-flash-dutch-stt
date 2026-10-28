"""Word-level timestamps: the CTC head emits tokens on 80 ms frames; the final token sequence (from the attention
decoder) is aligned to the CTC sequence and inherits its times."""
from __future__ import annotations

from difflib import SequenceMatcher

import torch

from .decode import FPS
from .tokenizer import PAD

Tok = tuple[int, float, float]     # (id, start_s, end_s)


def ctc_token_times(log_probs: torch.Tensor, enc_len: int) -> list[Tok]:
    """Greedy CTC path -> (token id, start, end) per emitted token, window-relative seconds."""
    toks: list[list] = []
    prev = None
    for t, i in enumerate(log_probs[0, :enc_len].argmax(-1).tolist()):
        if i != PAD and i != prev:
            toks.append([i, t, t])
        elif i != PAD:
            toks[-1][2] = t
        prev = i
    return [(i, f0 / FPS, (f1 + 1) / FPS) for i, f0, f1 in toks]


def align_times(final_ids: list[int], ctc: list[Tok]) -> list[Tok]:
    """Time every token of final_ids by aligning it to the CTC tokens; unmatched tokens are interpolated."""
    ctc_ids = [c[0] for c in ctc]
    times: list = [None] * len(final_ids)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, final_ids, ctc_ids, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                times[i1 + k] = (ctc[j1 + k][1], ctc[j1 + k][2])
        elif tag in ("replace", "insert") and j2 > j1 and i2 > i1:
            s, e, n = ctc[j1][1], ctc[j2 - 1][2], i2 - i1
            for k in range(n):
                times[i1 + k] = (s + (e - s) * k / n, s + (e - s) * (k + 1) / n)
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        end, n = (ctc[-1][2] if ctc else 0.0), max(len(final_ids), 1)
        return [(i, end * k / n, end * (k + 1) / n) for k, i in enumerate(final_ids)]
    for i in range(len(times)):
        if times[i] is None:
            prev = max([k for k in known if k < i], default=None); nxt = min([k for k in known if k > i], default=None)
            s = times[prev][1] if prev is not None else times[nxt][0]
            e = times[nxt][0] if nxt is not None else times[prev][1]
            times[i] = (s, max(s, e))
    return [(i, t[0], t[1]) for i, t in zip(final_ids, times)]


def words_from_tokens(timed: list[Tok], tok, offset_s: float = 0.0) -> list[dict]:
    """Timed token ids -> [{"word", "start", "end"}] (a SentencePiece piece starting with '▁' starts a word)."""
    words: list[dict] = []
    for i, s, e in timed:
        if i in tok._special:
            continue
        piece = tok.piece(i)
        if piece.startswith("▁") or not words:
            words.append({"word": piece.lstrip("▁"), "start": round(offset_s + s, 2), "end": round(offset_s + e, 2)})
        else:
            words[-1]["word"] += piece; words[-1]["end"] = round(offset_s + e, 2)
    return [w for w in words if w["word"]]
