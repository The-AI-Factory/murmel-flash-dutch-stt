"""OpenAI-compatible transcription server:  murmel-flash serve [--port 8765] [--model REPO] [--api-key KEY]

Any client that speaks the OpenAI audio API can use it (OpenWhispr's "Custom provider", scripts using the openai
package, curl). Endpoints:

    POST /v1/audio/transcriptions   multipart: file (required), model, language, prompt, response_format
                                    (json | text | verbose_json | srt | vtt), plus an extra: words (comma-separated
                                    domain word list)
    GET  /v1/models                 lists the loaded model
    GET  /health

The ``prompt`` field is treated as a word list when it looks like one (comma-separated, no sentences), so clients
without a dedicated field can still pass names and jargon.
"""

import os
import re
import threading
import time


def _srt_time(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _subtitles(segments, vtt: bool) -> str:
    sep, out = (".", ["WEBVTT", ""]) if vtt else (",", [])
    for i, seg in enumerate(segments, 1):
        if not vtt:
            out.append(str(i))
        out += [f"{_srt_time(seg['start'], sep)} --> {_srt_time(seg['end'], sep)}", seg["text"], ""]
    return "\n".join(out)


def _words_from_prompt(prompt: str | None) -> list[str] | None:
    if not prompt:
        return None
    parts = [p.strip() for p in re.split(r"[,\n;]", prompt) if p.strip()]
    if not parts or len(parts) > 200 or any(len(p.split()) > 4 for p in parts) or re.search(r"[.!?]\s", prompt):
        return None                                                 # a sentence, not a list
    return parts


def build_app(model: str, device: str | None, api_key: str | None, beam: int):
    from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    from .inference import Transcriber, load_audio

    tr = Transcriber.from_pretrained(model, device=device)
    lock = threading.Lock()            # one request at a time per process (use several workers for throughput)
    app = FastAPI(title="murmel-flash", version="4.0.0")
    model_id = model.rstrip("/").split("/")[-1]

    def auth(request: Request):
        if api_key and request.headers.get("authorization", "").removeprefix("Bearer ").strip() != api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_id, "device": str(tr.device)}

    @app.get("/v1/models")
    @app.get("/models")
    def models(_=Depends(auth)):
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "murmel"}]}

    @app.post("/v1/audio/transcriptions")
    @app.post("/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...), model: str | None = Form(None), language: str | None = Form(None),
                             prompt: str | None = Form(None), response_format: str = Form("json"), words: str | None = Form(None),
                             temperature: str | None = Form(None), _=Depends(auth)):
        data = await file.read()
        try:
            wave = load_audio(data, file.filename or "audio")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"could not decode audio: {e}") from e
        hot = [w.strip() for w in words.split(",") if w.strip()] if words else _words_from_prompt(prompt)
        t0 = time.time()
        with lock:
            res = tr.transcribe(wave, beam=beam, hotwords=hot)
        fmt = (response_format or "json").lower()
        if fmt == "text":
            return PlainTextResponse(res["text"])
        if fmt in ("srt", "vtt"):
            return PlainTextResponse(_subtitles(res["segments"], vtt=fmt == "vtt"), media_type="text/vtt" if fmt == "vtt" else "text/plain")
        if fmt == "verbose_json":
            segs = [{"id": i, "start": s["start"], "end": s["end"], "text": s["text"],
                     **({"language": s["language"]} if "language" in s else {})} for i, s in enumerate(res["segments"])]
            return JSONResponse({"task": "transcribe", "language": res.get("language", language or "nl"), "duration": round(len(wave) / 16000, 2),
                                 "text": res["text"], "segments": segs, "words": res.get("words", []),
                                 "processing_seconds": round(time.time() - t0, 2)})
        return JSONResponse({"text": res["text"]})

    return app


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="murmel-flash serve", description="OpenAI-compatible transcription server")
    ap.add_argument("--model", default=None, help="Hub repo or local directory (default: the packaged model)")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default=None); ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--api-key", default=os.environ.get("MURMEL_API_KEY"), help="require this bearer token (default: none)")
    args = ap.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('the server needs extra packages:  pip install "murmel-flash[serve]"') from None
    from .inference import DEFAULT_MODEL
    app = build_app(args.model or DEFAULT_MODEL, args.device, args.api_key, args.beam)
    print(f"murmel-flash serve: http://{args.host}:{args.port}/v1  (OpenWhispr: custom provider base URL = that address)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
