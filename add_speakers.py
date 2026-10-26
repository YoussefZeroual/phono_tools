#!/usr/bin/env python3
"""Add one tier per speaker to an MFA TextGrid, using a diarization JSON."""
import argparse
import json
import re
import sys


def parse(path):
    txt = open(path, encoding="utf-8").read()
    xmax = float(re.search(r"xmax = ([\d.eE+-]+)", txt).group(1))
    tiers = []
    for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
        name = re.search(r'name = "(.*)"', block).group(1)
        ivs = re.findall(
            r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "((?:[^"]|"")*)"', block)
        tiers.append((name, [(float(a), float(b), t.replace('""', '"')) for a, b, t in ivs]))
    return xmax, tiers


def speaker_of(s, e, turns):
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(e, t["end"]) - max(s, t["start"])
        if ov > best_ov:
            best, best_ov = t["speaker"], ov
    if best is None:
        mid = (s + e) / 2
        best = min(turns, key=lambda t: 0 if t["start"] <= mid <= t["end"]
                   else min(abs(mid - t["start"]), abs(mid - t["end"])))["speaker"]
    return best


def fill_gaps(items, xmax):
    ivs, t = [], 0.0
    for s, e, w in items:
        if s > t:
            ivs.append((t, s, ""))
        ivs.append((s, e, w))
        t = e
    if t < xmax:
        ivs.append((t, xmax, ""))
    return ivs


def write(tiers, xmax, path):
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           "xmin = 0", f"xmax = {xmax}", "tiers? <exists>", f"size = {len(tiers)}", "item []:"]
    for n, (name, ivs) in enumerate(tiers, 1):
        out += [f"    item [{n}]:", '        class = "IntervalTier"', f'        name = "{name}"',
                "        xmin = 0", f"        xmax = {xmax}", f"        intervals: size = {len(ivs)}"]
        for i, (s, e, w) in enumerate(ivs, 1):
            w = w.replace('"', '""')
            out += [f"        intervals [{i}]:", f"            xmin = {s}",
                    f"            xmax = {e}", f'            text = "{w}"']
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("textgrid", help="MFA output TextGrid")
    p.add_argument("diar", help="diarization JSON ({'turns': [...]})")
    p.add_argument("-o", "--out", help="output (default: <input>.spk.TextGrid)")
    p.add_argument("--keep-phones", action="store_true",
                   help="also keep the original MFA tiers (words/phones)")
    a = p.parse_args()

    xmax, tiers = parse(a.textgrid)
    data = json.load(open(a.diar, encoding="utf-8"))
    turns = data["turns"] if isinstance(data, dict) else data

    wt = next((t for t in tiers if t[0].lower().endswith("words")), None)
    if wt is None:
        sys.exit("no words tier found: " + ", ".join(n for n, _ in tiers))

    by_spk = {}
    for s, e, w in wt[1]:
        if w.strip():
            by_spk.setdefault(speaker_of(s, e, turns), []).append((s, e, w))

    new = [(spk, fill_gaps(by_spk[spk], xmax)) for spk in sorted(by_spk)]
    out = a.out or a.textgrid.rsplit(".", 1)[0] + ".spk.TextGrid"
    write((tiers if a.keep_phones else []) + new, xmax, out)
    print(out)


if __name__ == "__main__":
    main()
