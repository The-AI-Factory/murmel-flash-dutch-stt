"""OpenAI-compatible server on the released weights; skipped when they are not available."""
import os

import pytest
from huggingface_hub import get_token

REPO = os.environ.get("MURMEL_TEST_MODEL", "theaifactorybv/murmel-flash-m-v5")


@pytest.fixture(scope="module")
def client():
    if not (os.path.isdir(REPO) or get_token()):            # HF_TOKEN or `hf auth login`
        pytest.skip("no released weights available")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from murmel_flash.server import build_app
    return TestClient(build_app(REPO, "cpu", "secret", beam=2))


def _wav_bytes():
    import io
    import struct

    import numpy as np
    sr = 16000; t = np.arange(sr * 2) / sr
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) * (t < 1.0)).astype(np.float32)   # 1 s tone, 1 s silence (no speech)
    data = struct.pack(f"<{len(pcm)}h", *(pcm * 32767).astype("int16"))
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", len(data))
    return io.BytesIO(hdr + data)


def test_auth_models_and_formats(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/v1/models").status_code == 401
    h = {"Authorization": "Bearer secret"}
    assert client.get("/v1/models", headers=h).json()["data"][0]["id"] == os.path.basename(REPO.rstrip("/"))
    for path in ("/v1/audio/transcriptions", "/audio/transcriptions"):
        r = client.post(path, headers=h, files={"file": ("tone.wav", _wav_bytes(), "audio/wav")}, data={"response_format": "json"})
        assert r.status_code == 200 and "text" in r.json()
    r = client.post("/v1/audio/transcriptions", headers=h, files={"file": ("tone.wav", _wav_bytes(), "audio/wav")}, data={"response_format": "verbose_json"})
    assert {"task", "language", "duration", "text", "segments", "words"} <= set(r.json())
    r = client.post("/v1/audio/transcriptions", headers=h, files={"file": ("tone.wav", _wav_bytes(), "audio/wav")}, data={"response_format": "vtt"})
    assert r.text.startswith("WEBVTT")
    r = client.post("/v1/audio/transcriptions", headers=h, files={"file": ("x.bin", b"not audio", "application/octet-stream")})
    assert r.status_code == 400
