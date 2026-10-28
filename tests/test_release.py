"""End-to-end on the released weights; skipped when they are not available locally or on the Hub."""
import os

import numpy as np
import pytest
from huggingface_hub import get_token

from murmel_flash import Transcriber, load_audio
from murmel_flash.inference import SR

REPO = os.environ.get("MURMEL_TEST_MODEL", "theaifactorybv/murmel-flash-m-v5")


@pytest.fixture(scope="module")
def tr():
    if not (os.path.isdir(REPO) or get_token()):            # HF_TOKEN or `hf auth login`
        pytest.skip("no released weights available")
    return Transcriber.from_pretrained(REPO, device="cpu")


def test_silence_gives_empty_result(tr):
    res = tr.transcribe(np.zeros(5 * SR, dtype=np.float32))
    assert res["text"] == "" and res["segments"] == []


def test_result_schema_on_synthetic_speech(tr):
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("say"):
        pytest.skip("needs macOS TTS for a quick synthetic clip")
    path = os.path.join(tempfile.mkdtemp(), "t.wav")       # 22.05 kHz: also exercises resampling
    subprocess.run(["say", "-v", "Xander", "--file-format=WAVE", "--data-format=LEI16@22050", "-o", path, "Dit is een korte test van het model."], check=True)
    res = tr.transcribe(load_audio(path))
    assert res["language"] == "nl" and res["words"] and {"start", "end", "text", "language", "words"} <= set(res["segments"][0])
    assert res["words"][0]["start"] < res["words"][-1]["end"]
