"""Command line:  murmel-flash FILE [FILE ...] [--model REPO] [--words w1,w2] [--json DIR]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["serve"]:                         # murmel-flash serve [...]: the OpenAI-compatible server
        from .server import main as serve
        return serve(argv[1:])
    ap = argparse.ArgumentParser(prog="murmel-flash", description="Dutch + English speech recognition.  Also: murmel-flash serve --help")
    ap.add_argument("files", nargs="*", help="audio/video files (anything ffmpeg reads)")
    ap.add_argument("--mic", action="store_true", help="transcribe the microphone live (needs: pip install sounddevice)")
    ap.add_argument("--model", default=None, help="Hub repo or local directory (default: the packaged model)")
    ap.add_argument("--words", default="", help="comma-separated domain word list")
    ap.add_argument("--beam", type=int, default=4); ap.add_argument("--mode", default="beam", choices=["beam", "ctc"])
    ap.add_argument("--device", default=None); ap.add_argument("--json", default=None, help="write <stem>.json into this directory")
    ap.add_argument("--stream", action="store_true", help="simulate live input: feed the file in 0.5 s chunks, print partials and finals")
    args = ap.parse_args(argv)
    if not args.files and not args.mic:
        ap.error("give one or more files, or --mic")
    from .inference import DEFAULT_MODEL, Transcriber, load_audio
    tr = Transcriber.from_pretrained(args.model or DEFAULT_MODEL, device=args.device)
    words = [w.strip() for w in args.words.split(",") if w.strip()] or None
    if args.mic:
        _mic(tr, words, args)
        return None
    for f in args.files:
        wave = load_audio(f)
        if args.stream:
            _stream(tr, wave, words, args)
            continue
        res = tr.transcribe(wave, mode=args.mode, beam=args.beam, hotwords=words)
        if len(args.files) > 1:
            print(f"== {f}")
        print(res["text"])
        if args.json:
            out = Path(args.json); out.mkdir(parents=True, exist_ok=True)
            (out / f"{Path(f).stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return None


def _stream(tr, wave, words, args) -> None:
    from .inference import SR
    session = tr.stream(mode=args.mode, beam=args.beam, hotwords=words)
    step = SR // 2
    for i in range(0, len(wave), step):
        for ev in session.feed(wave[i : i + step]):
            if ev["type"] == "partial":
                print("\r  ... " + ev["text"][-110:], end="", flush=True)
            else:
                seg = ev["segment"]; print(f"\r[{seg['start']:7.2f}-{seg['end']:7.2f}] {seg['text']}", flush=True)
    for ev in session.finish():
        seg = ev["segment"]; print(f"\r[{seg['start']:7.2f}-{seg['end']:7.2f}] {seg['text']}", flush=True)


def _print_events(events) -> None:
    for ev in events:
        if ev["type"] == "partial":
            print("\r  ... " + ev["text"][-110:] + " " * 8, end="", flush=True)
        else:
            seg = ev["segment"]; print(f"\r[{seg['start']:7.2f}-{seg['end']:7.2f}] {seg['text']}" + " " * 8, flush=True)


def _mic(tr, words, args) -> None:
    """Live transcription of the default microphone until Ctrl-C."""
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("microphone input needs the sounddevice package:  pip install sounddevice")
    import queue

    from .inference import SR
    session = tr.stream(mode=args.mode, beam=args.beam, hotwords=words)
    q: queue.Queue = queue.Queue()
    dev = sd.query_devices(kind="input")
    print(f"listening on '{dev['name']}' (Ctrl-C to stop)", flush=True)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=SR // 4, callback=lambda data, *_: q.put(data[:, 0].copy())):
        try:
            while True:
                try:
                    _print_events(session.feed(q.get(timeout=0.5)))     # timeout keeps Ctrl-C responsive
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            pass
    _print_events(session.finish())
    print()


if __name__ == "__main__":
    sys.exit(main())
