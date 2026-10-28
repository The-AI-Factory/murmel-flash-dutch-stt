"""SentencePiece tokenizer with the reserved ids the model relies on.

    0 = <pad> (also the CTC blank)   1 = <unk>   2 = <s> (decoder start)   3 = </s> (decoder end)
    4.. = language tags (<nl>, <en>), then the learned pieces; v4 models add two style tags (<verbatim>, <clean>).

A v3 label sequence is ``[lang_id] + pieces``; a v4 one is ``[lang_id, style_id] + pieces``. The first token the
decoder emits is the language, so language identification is part of decoding; the style is forced at inference.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import sentencepiece as spm

PAD, UNK, BOS, EOS = 0, 1, 2, 3
LANGS = ["nl", "en"]
STYLES = ["verbatim", "clean"]


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        assert self.sp.pad_id() == PAD and self.sp.unk_id() == UNK and self.sp.bos_id() == BOS and self.sp.eos_id() == EOS
        self.lang_ids = {lang: self.sp.piece_to_id(f"<{lang}>") for lang in LANGS}
        # style tags exist only in v4 tokenizers (piece_to_id returns <unk> for a missing piece)
        self.style_ids = {st: i for st in STYLES if (i := self.sp.piece_to_id(f"<{st}>")) != UNK}
        self._special = {PAD, UNK, BOS, EOS, *self.lang_ids.values(), *self.style_ids.values()}

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def lang_of(self, ids: Sequence[int]) -> str | None:
        """Language tag found first in an id sequence, if any."""
        inv = {v: k for k, v in self.lang_ids.items()}
        return next((inv[int(i)] for i in ids if int(i) in inv), None)

    def encode(self, text: str, lang: str) -> list[int]:
        return [self.lang_ids[lang]] + self.sp.encode(text, out_type=int)

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode([int(i) for i in ids if int(i) not in self._special])

    def piece(self, i: int) -> str:
        return self.sp.id_to_piece(int(i))
