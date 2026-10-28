"""Domain word lists for beam search: a token trie built from the list gives hypotheses that follow a listed word a
per-token bonus, taken back if the word is abandoned before it completes. No retraining; a different list per call."""
from __future__ import annotations

from collections.abc import Iterable

State = tuple[dict | None, float]      # (trie node of the current partial match, uncommitted bonus)


class WordBooster:
    def __init__(self, words: Iterable[str], tok, weight: float = 1.0, max_bonus_per_word: float = 8.0, min_pieces: int = 2):
        """Terms the tokenizer already writes as ONE piece are common words; boosting them only creates false alarms,
        so by default only multi-piece terms count (min_pieces)."""
        self.weight, self.cap, self.min_pieces = float(weight), float(max_bonus_per_word), int(min_pieces)
        self.root: dict = {"kids": {}, "end": False}
        seen = set()
        for w in words:
            w = " ".join(w.split())
            if not w:
                continue
            for variant in {w, w.lower(), w[:1].upper() + w[1:]}:
                ids = tuple(tok.sp.encode(variant, out_type=int))
                if len(ids) < self.min_pieces or ids in seen:
                    continue
                seen.add(ids)
                node = self.root
                for i in ids:
                    node = node["kids"].setdefault(i, {"kids": {}, "end": False})
                node["end"] = True

    @property
    def initial(self) -> State:
        return (None, 0.0)

    def step(self, state: State, token: int) -> tuple[State, float]:
        """Advance the match by one emitted token -> (new state, score delta)."""
        node, pending = state
        delta = 0.0
        if node is not None and token in node["kids"]:
            nxt = node["kids"][token]
            b = min(self.weight, self.cap - pending) if pending < self.cap else 0.0
            delta += b; pending += b
            if nxt["end"]:
                pending = 0.0                                   # word complete: bonus committed
            return ((nxt if nxt["kids"] else None), pending), delta
        if node is not None:                                    # partial match broken: cancel its bonus
            delta -= pending
        if token in self.root["kids"]:                          # can this token start a listed word?
            nxt = self.root["kids"][token]
            b = min(self.weight, self.cap)
            return ((nxt if nxt["kids"] else None), 0.0 if nxt["end"] else b), delta + b
        return (None, 0.0), delta
