"""Decoding on an encoded batch: CTC greedy (cheap) and attention beam search (best, incremental, batched)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .tokenizer import BOS, EOS, PAD

FPS = 12.5   # encoder frames per second (10 ms hop x 8 subsampling)


def ctc_greedy(log_probs: torch.Tensor, enc_lengths: torch.Tensor) -> list[list[int]]:
    """Collapse repeats, drop blanks. log_probs (B, T, V)."""
    best = log_probs.argmax(dim=-1).cpu()
    out = []
    for b in range(best.shape[0]):
        ids, prev = [], None
        for t in best[b, : int(enc_lengths[b])].tolist():
            if t != prev and t != PAD:
                ids.append(t)
            prev = t
        out.append(ids)
    return out


def ctc_confidence(log_probs: torch.Tensor, enc_len: int) -> float:
    """Mean posterior of the argmax on the non-blank frames of one window (log_probs (1, T, V)).
    Speech scores ~0.85 (rarely below 0.74); music, singing and jingles 0.53-0.71 -- the no-speech gate."""
    mx, am = log_probs[0, :enc_len].float().exp().max(-1)
    nb = am != PAD
    return float(mx[nb].mean()) if bool(nb.any()) else 0.0


def non_speech_window(log_probs: torch.Tensor, enc_len: int, win_s: float, min_conf: float, dense_tps: float = 2.0,
                      n_ctc: int | None = None) -> bool:
    """The music / jingle gate for one window: True when the CTC confidence is below min_conf AND CTC hears fewer than
    dense_tps tokens per second. Confidence alone cannot tell a jingle from degraded speech (both 0.5-0.7); the token
    rate can: jingles and music beds give <= 1.3 tokens/s at low confidence, phone lines, old tapes and heavily
    compressed video 2.5-4, so those are kept even when the model is unsure. min_conf 0 disables the gate, dense_tps 0
    the rescue. n_ctc: the CTC greedy token count if the caller already has it."""
    if min_conf <= 0 or ctc_confidence(log_probs, enc_len) >= min_conf:
        return False
    if dense_tps <= 0:
        return True
    if n_ctc is None:
        n_ctc = len(ctc_greedy(log_probs, torch.tensor([enc_len]))[0])
    return n_ctc / max(win_s, 1e-6) < dense_tps


def _loops(seq: Sequence[int], n: int = 3, times: int = 3) -> bool:
    """True if the last n-gram already occurred >= times-1 times before (a decoder stuck in a loop)."""
    if len(seq) < n * times:
        return False
    tail = tuple(seq[-n:])
    return sum(1 for i in range(len(seq) - n) if tuple(seq[i : i + n]) == tail) >= times - 1


def has_loop(ids: Sequence[int], max_repeat: int = 4) -> bool:
    """True when some 1- to 3-gram repeats max_repeat times back to back."""
    for n in (1, 2, 3):
        run = 1
        for i in range(n, len(ids), n):
            if list(ids[i : i + n]) == list(ids[i - n : i]):
                run += 1
                if run >= max_repeat:
                    return True
            else:
                run = 1
    return False


@torch.no_grad()
def attention_beam(model, enc: torch.Tensor, enc_lengths: torch.Tensor, beam: int = 4, max_len: int | Sequence[int] = 256,
                   length_penalty: float = 1.0, force_lang_id: int | None = None, booster=None,
                   force_tokens: dict[int, int] | None = None) -> list[list[int]]:
    """Beam search over a batch of utterances at once. Incremental: memory keys/values once per utterance, prefix
    keys/values cached per hypothesis, one decoder call per step for all live hypotheses. Length-normalised scores.
    max_len: int, or one cap per utterance. booster: optional WordBooster (domain word list).
    force_tokens: {decoder step: token id} emitted at that step for every utterance -- v4 models get their style tag
    forced at step 1 (step 0, the language, stays free). force_lang_id is the same for step 0."""
    forced = dict(force_tokens or {})
    if force_lang_id is not None:
        forced[0] = force_lang_id
    dec = model.decoder
    B = enc.shape[0]
    caps = [int(max_len)] * B if isinstance(max_len, (int, float)) else [int(c) for c in max_len]
    Tmax = int(enc_lengths.max())
    mem = enc[:, :Tmax]
    valid = torch.arange(Tmax, device=enc.device)[None] < enc_lengths.to(enc.device)[:, None]
    mem_cache, caches = dec.memory_cache(mem), dec.new_cache()
    init = booster.initial if booster else None
    hyps = [[([BOS], 0.0, init, -1)] for _ in range(B)]          # (tokens, score, booster state, parent slot)
    finished: list[list] = [[] for _ in range(B)]
    done = [False] * B
    for step in range(max(caps)):
        active = [(b, h) for b in range(B) if not done[b] for h in hyps[b]]
        if not active:
            break
        if step > 0:
            dec.reorder_cache(caches, torch.tensor([h[3] for _, h in active], dtype=torch.long, device=enc.device))
        last = torch.tensor([h[0][-1] for _, h in active], dtype=torch.long, device=enc.device)
        utt = torch.tensor([b for b, _ in active], dtype=torch.long, device=enc.device)
        logp = F.log_softmax(dec.step(last, step, caches, mem_cache, utt, valid).float(), dim=-1)
        if step in forced:
            mask = torch.full_like(logp, float("-inf")); mask[:, forced[step]] = 0.0
            logp = logp + mask
        topv, topi = logp.topk(beam, dim=-1)
        topv, topi = topv.tolist(), topi.tolist()
        cands: dict[int, list] = {}
        for k, (b, (seq, score, bst, _)) in enumerate(active):
            cand = cands.setdefault(b, [])
            for v, i in zip(topv[k], topi[k]):
                if i != EOS and _loops(seq + [i]):
                    continue
                if booster is None:
                    cand.append((seq + [i], score + v, None, k))
                elif i != EOS:
                    nbst, delta = booster.step(bst, i)
                    cand.append((seq + [i], score + v + delta, nbst, k))
                else:                                          # EOS cancels an unfinished partial match
                    cand.append((seq + [i], score + v - bst[1], bst, k))
        for b, cand in cands.items():
            cand.sort(key=lambda x: x[1], reverse=True)
            live = []
            for seq, score, bst, par in cand:
                if seq[-1] == EOS:
                    finished[b].append((seq, score / (len(seq) - 1) ** length_penalty))
                else:
                    live.append((seq, score, bst, par))
                if len(live) >= beam:
                    break
            # every candidate can be a loop (repetitive music, silence): keep the previous beam then, so the utterance
            # still ends with a hypothesis instead of an empty beam (max() over nothing crashed decoding)
            if live:
                hyps[b] = live
            fin = finished[b]
            if not live or step + 1 >= caps[b] or (fin and len(fin) >= beam and max(f[1] for f in fin) >
                                                    max(h[1] for h in live) / len(live[0][0]) ** length_penalty):
                done[b] = True
    results = []
    for b in range(B):
        fin = finished[b] or [(seq + [EOS], score / len(seq) ** length_penalty) for seq, score, _, _ in hyps[b]]
        if not fin:
            results.append([]); continue
        best = max(fin, key=lambda x: x[1])[0]
        results.append([t for t in best[1:] if t not in (EOS, PAD)])
    return results
