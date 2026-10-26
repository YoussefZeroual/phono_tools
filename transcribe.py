#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def transcribe(url, model, path, language=None, timeout=600):
    body = {"model": model, "audio": str(path.resolve())}
    if language:
        body["language"] = language
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["text"].strip()


def main():
    p = argparse.ArgumentParser(description="Transcribe a file or folder with a running audiocpp_server.")
    p.add_argument("input", help="audio file or folder")
    p.add_argument("-o", "--output", help="output folder (default: next to each input file)")
    p.add_argument("--suffix", default="lab", help="output extension without dot (default: lab)")
    p.add_argument("--model", default="my-asr", help="model id in server.json (default: my-asr)")
    p.add_argument("--host", default="http://127.0.0.1:8080", help="server address (default: http://127.0.0.1:8080)")
    p.add_argument("--language", default="fr", help="sent as 'language' in the request (default: fr)")
    p.add_argument("--overwrite", action="store_true", help="redo files that already have an output")
    args = p.parse_args()

    src = Path(args.input)
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    elif src.is_file():
        files = [src]
    else:
        sys.exit(f"not found: {src}")
    if not files:
        sys.exit("no audio files found")

    out_dir = Path(args.output) if args.output else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    url = args.host.rstrip("/") + "/v1/audio/transcriptions"
    failed = 0
    for i, f in enumerate(files, 1):
        out = (out_dir or f.parent) / f"{f.stem}.{args.suffix}"
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(files)}] skip {f.name}")
            continue
        try:
            text = transcribe(url, args.model, f, args.language)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as e:
            failed += 1
            print(f"[{i}/{len(files)}] FAILED {f.name}: {e}", file=sys.stderr)
            continue
        out.write_text(text, encoding="utf-8")
        print(f"[{i}/{len(files)}] {f.name} -> {text[:60]}")

    if failed:
        sys.exit(f"{failed} file(s) failed")


if __name__ == "__main__":
    main()
