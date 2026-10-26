#!/usr/bin/env python3
"""Speaker diarization with pyannote. Outputs JSON (and optionally RTTM)."""
import argparse
import json
import os
import sys


def load_pipeline(model, token, device):
    import torch
    from pyannote.audio import Pipeline

    try:
        pipe = Pipeline.from_pretrained(model, token=token)
    except TypeError:  # older pyannote versions
        pipe = Pipeline.from_pretrained(model, use_auth_token=token)
    if pipe is None:
        sys.exit("could not load pipeline: accept the model terms on HF and check your token")
    pipe.to(torch.device(device))
    return pipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="input audio file")
    p.add_argument("-o", "--out", help="output JSON (default: audio name + .diar.json)")
    p.add_argument("--rttm", action="store_true", help="also write an .rttm file")
    p.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token (default: $HF_TOKEN)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda)")
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    a = p.parse_args()

    if not a.token:
        sys.exit("no HF token: set HF_TOKEN or pass --token")

    pipe = load_pipeline(a.model, a.token, a.device)

    kw = {k: v for k, v in {
        "num_speakers": a.num_speakers,
        "min_speakers": a.min_speakers,
        "max_speakers": a.max_speakers,
    }.items() if v is not None}
    import soundfile as sf
    import torch

    data, sr = sf.read(a.audio, dtype="float32", always_2d=True)  # (time, channel)
    audio = {"waveform": torch.from_numpy(data.T.copy()), "sample_rate": sr}
    result = pipe(audio, **kw)
    # newer pyannote versions wrap the annotation
    diar = getattr(result, "speaker_diarization", result)

    turns = [{"start": round(t.start, 3), "end": round(t.end, 3), "speaker": spk}
             for t, _, spk in diar.itertracks(yield_label=True)]

    base = a.audio.rsplit(".", 1)[0]
    out = a.out or base + ".diar.json"
    json.dump({"turns": turns}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(out)

    if a.rttm:
        rttm = base + ".rttm"
        with open(rttm, "w") as f:
            diar.write_rttm(f)
        print(rttm)


if __name__ == "__main__":
    main()
