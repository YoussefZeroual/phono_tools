#!/usr/bin/env python3
"""Convert word-level JSON (start_sample/end_sample) to a Praat TextGrid."""
import argparse
import json
import sys
import wave


def load_words(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["words"] if isinstance(data, dict) else data


def wav_duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def build_intervals(words, sr, xmax):
    ivs, t = [], 0.0
    for d in words:
        s = max(d["start_sample"] / sr, t)  # clamp overlaps
        e = max(d["end_sample"] / sr, s)
        if s > t:
            ivs.append((t, s, ""))
        ivs.append((s, e, d["word"]))
        t = e
    xmax = max(xmax, t)
    if t < xmax:
        ivs.append((t, xmax, ""))
    return ivs, xmax


def write_textgrid(ivs, xmax, tier, path):
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax}", "tiers? <exists>", "size = 1", "item []:",
           "    item [1]:", '        class = "IntervalTier"', f'        name = "{tier}"',
           "        xmin = 0", f"        xmax = {xmax}",
           f"        intervals: size = {len(ivs)}"]
    for i, (s, e, w) in enumerate(ivs, 1):
        w = w.replace('"', '""')
        out += [f"        intervals [{i}]:", f"            xmin = {s}",
                f"            xmax = {e}", f'            text = "{w}"']
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json", help="input JSON (object with 'words', or a bare list)")
    p.add_argument("-o", "--out", help="output .TextGrid (default: input name)")
    p.add_argument("--wav", help="audio file, used for total duration")
    p.add_argument("--duration", type=float, help="total duration in seconds (overrides --wav)")
    p.add_argument("--sr", type=int, default=16000, help="sample rate (default 16000)")
    p.add_argument("--tier", default="words", help="tier name (default: words)")
    p.add_argument("--skip-tags", action="store_true",
                   help="drop bracketed tags like [laughter]")
    a = p.parse_args()

    words = load_words(a.json)
    if a.skip_tags:
        words = [w for w in words if not (w["word"].startswith("[") and w["word"].endswith("]"))]
    if not words:
        sys.exit("no words found")

    if a.duration is not None:
        xmax = a.duration
    elif a.wav:
        xmax = wav_duration(a.wav)
    else:
        xmax = words[-1]["end_sample"] / a.sr

    ivs, xmax = build_intervals(words, a.sr, xmax)
    out = a.out or a.json.rsplit(".", 1)[0] + ".TextGrid"
    write_textgrid(ivs, xmax, a.tier, out)
    print(out)


if __name__ == "__main__":
    main()
