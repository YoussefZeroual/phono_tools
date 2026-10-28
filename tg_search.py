#!/usr/bin/env python3
"""Search a word (or part of a word) in a TextGrid tier and play occurrences.
Usage: python tg_search.py [file.TextGrid] [file.wav]   (both optional, openable in the GUI)
Needs: pip install sounddevice soundfile matplotlib"""
import os
import re
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import sounddevice as sd
import soundfile as sf


def parse(path):
    raw = open(path, "rb").read()
    txt = None
    for enc in ("utf-8", "utf-16"):
        try:
            txt = raw.decode(enc)
            break
        except UnicodeError:
            continue
    if txt is None:
        raise ValueError("cannot decode file")
    tiers = {}
    for block in re.split(r"\n\s*item \[\d+\]:", txt)[1:]:
        name = re.search(r'name = "(.*)"', block).group(1)
        ivs = re.findall(
            r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "((?:[^"]|"")*)"', block)
        words = [(float(a), float(b), t.replace('""', '"')) for a, b, t in ivs if t.strip()]
        if words:
            tiers[name] = words
    return tiers


def search(words, query, ctx=4):
    q = query.lower()
    res = []
    for i, (s, e, w) in enumerate(words):
        if q and q in w.lower():
            left = " ".join(x[2] for x in words[max(0, i - ctx):i])
            right = " ".join(x[2] for x in words[i + 1:i + 1 + ctx])
            ls = words[max(0, i - ctx)][0]  # start of the left context
            re_ = words[min(len(words) - 1, i + ctx)][1]  # end of the right context
            res.append((s, e, w, left, right, ls, re_))
    return res


def draw_segment(png, x, sr, t0, tiers, hl_tier, hl, fmax=5000):
    """Waveform + spectrogram + all tiers of the window starting at t0, saved to png."""
    import numpy as np
    from matplotlib import mlab
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    if x.ndim > 1:
        x = x.mean(axis=1)
    t1 = t0 + len(x) / sr
    names = list(tiers)
    fig = Figure(figsize=(12, 4.5 + 0.35 * len(names)), dpi=150)
    gs = fig.add_gridspec(2 + len(names), 1, height_ratios=[1, 3] + [0.6] * len(names), hspace=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(2 + len(names))]

    axes[0].plot(t0 + np.arange(len(x)) / sr, x, lw=0.5, color="k")
    axes[0].set_yticks([])

    nfft = max(16, int(0.005 * sr))  # ~5 ms window (wideband)
    hop = max(1, int(0.001 * sr))
    if len(x) > nfft:
        spec, f, t = mlab.specgram(x, NFFT=nfft, Fs=sr, noverlap=nfft - hop,
                                   pad_to=max(nfft, 2048))  # zero-padding = finer frequency grid
        keep = f <= fmax
        db = 10 * np.log10(spec[keep] + 1e-12)
        axes[1].imshow(db, origin="lower", aspect="auto", interpolation="bilinear", cmap="Greys",
                       extent=[t0 + t[0], t0 + t[-1], f[keep][0], f[keep][-1]],
                       vmin=db.max() - 70, vmax=db.max())
    axes[1].set_ylabel("Hz")

    for ax, name in zip(axes[2:], names):
        for a, b, txt in tiers[name]:
            if b <= t0 or a >= t1:
                continue
            a2, b2 = max(a, t0), min(b, t1)
            on = name == hl_tier and abs(a - hl[0]) < 1e-6 and abs(b - hl[1]) < 1e-6
            ax.add_patch(Rectangle((a2, 0), b2 - a2, 1, facecolor="#ffd54f" if on else "none",
                                   edgecolor="k", lw=0.6))
            ax.text((a2 + b2) / 2, 0.5, txt, ha="center", va="center", fontsize=8, clip_on=True)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    for ax in axes:
        ax.set_xlim(t0, t1)
        ax.tick_params(labelbottom=False)
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(png, bbox_inches="tight")


class App:
    def __init__(self, root):
        self.root = root
        self.tiers = {}
        self.data = None
        self.sr = None
        self.hits = []
        self.tg_path = None
        self.wav_path = None
        self.out_dir = None

        bar = ttk.Frame(root, padding=(6, 6, 6, 0))
        bar.pack(fill="x")
        ttk.Button(bar, text="Open TextGrid…", command=self.open_tg).pack(side="left")
        ttk.Button(bar, text="Open WAV…", command=self.open_wav).pack(side="left", padx=4)
        self.files = ttk.Label(bar, text="no files loaded")
        self.files.pack(side="left", padx=8)

        top = ttk.Frame(root, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Tier").pack(side="left")
        self.tier = ttk.Combobox(top, values=[], state="readonly", width=24)
        self.tier.pack(side="left", padx=4)
        ttk.Label(top, text="Search").pack(side="left")
        self.q = ttk.Entry(top, width=20)
        self.q.pack(side="left", padx=4)
        self.q.bind("<Return>", lambda _: self.run())
        ttk.Button(top, text="Search", command=self.run).pack(side="left")
        ttk.Label(top, text="Pad (s)").pack(side="left", padx=(10, 2))
        self.pad = ttk.Spinbox(top, from_=0, to=5, increment=0.1, width=4)
        self.pad.set("0.3")
        self.pad.pack(side="left")
        ttk.Button(top, text="Play", command=self.play).pack(side="left", padx=(10, 2))
        ttk.Button(top, text="Stop", command=sd.stop).pack(side="left")
        ttk.Button(top, text="Open in Praat", command=self.praat).pack(side="left", padx=(10, 0))

        exp = ttk.Frame(root, padding=(6, 0, 6, 6))
        exp.pack(fill="x")
        ttk.Label(exp, text="Export").pack(side="left")
        self.ctx = ttk.Combobox(exp, values=["word", "word + left", "word + right", "word + both"],
                                state="readonly", width=14)
        self.ctx.current(0)
        self.ctx.pack(side="left", padx=4)
        ttk.Label(exp, text="Pad (s)").pack(side="left", padx=(10, 2))
        self.xpad = ttk.Spinbox(exp, from_=0, to=5, increment=0.05, width=4)
        self.xpad.set("0.0")
        self.xpad.pack(side="left")
        ttk.Label(exp, text="Suffix").pack(side="left", padx=(10, 2))
        self.suffix = ttk.Entry(exp, width=12)
        self.suffix.pack(side="left")
        ttk.Button(exp, text="Export wav + picture…", command=self.export).pack(side="left", padx=(10, 0))
        ttk.Button(exp, text="Select all", command=lambda: self.set_all(True)).pack(side="left", padx=(10, 2))
        ttk.Button(exp, text="Deselect all", command=lambda: self.set_all(False)).pack(side="left")

        cols = ("chk", "time", "left", "word", "right")
        self.tree = ttk.Treeview(root, columns=cols, show="headings", height=20)
        for c, w in zip(cols, (30, 80, 300, 120, 300)):
            self.tree.heading(c, text="" if c == "chk" else c)
            self.tree.column(c, width=w, anchor="center" if c == "chk" else ("e" if c == "left" else "w"),
                             stretch=c != "chk")
        self.tree.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.checked = set()
        self.tree.bind("<Button-1>", self.on_click)
        self.tree.bind("<Double-1>", lambda _: self.play())
        self.tree.bind("<Return>", lambda _: self.play())

        self.status = ttk.Label(root, text="")
        self.status.pack(fill="x", padx=6, pady=(0, 6))

    def update_files_label(self):
        tg = getattr(self, "tg_name", None) or "no TextGrid"
        wav = getattr(self, "wav_name", None) or "no WAV"
        self.files.config(text=f"{tg}  |  {wav}")

    def open_tg(self):
        path = filedialog.askopenfilename(
            title="Open TextGrid", filetypes=[("TextGrid", "*.TextGrid *.textgrid"), ("All", "*")])
        if path:
            self.load_tg(path)

    def open_wav(self):
        path = filedialog.askopenfilename(
            title="Open audio", filetypes=[("Audio", "*.wav *.WAV *.flac *.ogg"), ("All", "*")])
        if path:
            self.load_wav(path)

    def load_tg(self, path):
        try:
            tiers = parse(path)
        except Exception as ex:
            messagebox.showerror("TextGrid", f"Cannot read file:\n{ex}")
            return
        if not tiers:
            messagebox.showerror("TextGrid", "No tiers with text found (long format only).")
            return
        self.tiers = tiers
        self.tier.config(values=list(tiers))
        self.tier.current(0)
        self.tg_name = os.path.basename(path)
        self.tg_path = os.path.abspath(path)
        self.hits = []
        self.tree.delete(*self.tree.get_children())
        self.status.config(text=f"{len(tiers)} tier(s) loaded")
        self.update_files_label()
        if self.data is None:  # try the wav with the same name
            base = path.rsplit(".", 1)[0]
            for ext in (".wav", ".WAV"):
                if os.path.exists(base + ext):
                    self.load_wav(base + ext)
                    break

    def load_wav(self, path):
        try:
            self.data, self.sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as ex:
            messagebox.showerror("Audio", f"Cannot read file:\n{ex}")
            return
        self.wav_name = os.path.basename(path)
        self.wav_path = os.path.abspath(path)
        self.update_files_label()

    def on_click(self, ev):
        if self.tree.identify_region(ev.x, ev.y) != "cell" or self.tree.identify_column(ev.x) != "#1":
            return
        iid = self.tree.identify_row(ev.y)
        if not iid:
            return
        i = int(iid)
        if i in self.checked:
            self.checked.discard(i)
        else:
            self.checked.add(i)
        self.tree.set(iid, "chk", "☑" if i in self.checked else "☐")
        return "break"

    def set_all(self, state):
        self.checked = set(range(len(self.hits))) if state else set()
        for iid in self.tree.get_children():
            self.tree.set(iid, "chk", "☑" if state else "☐")

    def run(self):
        if not self.tiers:
            messagebox.showinfo("Search", "Open a TextGrid first.")
            return
        self.hits = search(self.tiers[self.tier.get()], self.q.get())
        self.checked.clear()
        self.tree.delete(*self.tree.get_children())
        for i, (s, e, w, l, r, *_) in enumerate(self.hits):
            self.tree.insert("", "end", iid=str(i), values=("☐", f"{s:.2f}", l, w, r))
        self.status.config(text=f"{len(self.hits)} occurrence(s)")

    def play(self):
        if self.data is None:
            messagebox.showinfo("Play", "Open a WAV file first.")
            return
        sel = self.tree.selection()
        if not sel:
            return
        s, e, *_ = self.hits[int(sel[0])]
        pad = float(self.pad.get())
        i0 = max(0, int((s - pad) * self.sr))
        i1 = min(len(self.data), int((e + pad) * self.sr))
        sd.stop()
        sd.play(self.data[i0:i1], self.sr)

    def export(self):
        if self.data is None:
            messagebox.showinfo("Export", "Open a WAV file first.")
            return
        idx = sorted(self.checked) or [int(i) for i in self.tree.selection()[:1]]
        if not idx:
            messagebox.showinfo("Export", "Check or select an occurrence first.")
            return
        folder = filedialog.askdirectory(title="Export to folder", initialdir=self.out_dir or os.getcwd())
        if not folder:
            return
        self.out_dir = folder
        mode = self.ctx.get()
        pad = float(self.xpad.get())
        suffix = re.sub(r"\W+", "_", self.suffix.get().strip()).strip("_")
        stem = os.path.splitext(getattr(self, "wav_name", "audio"))[0]
        try:
            for i in idx:
                s, e, w, _, _, ls, re_ = self.hits[i]
                a = ls if mode in ("word + left", "word + both") else s
                b = re_ if mode in ("word + right", "word + both") else e
                t0 = max(0.0, a - pad)
                t1 = min(len(self.data) / self.sr, b + pad)
                safe = re.sub(r"\W+", "_", w)[:30]
                base = os.path.join(folder, f"{stem}_{t0:.2f}_{safe}" + (f"_{suffix}" if suffix else ""))
                seg = self.data[int(t0 * self.sr):int(t1 * self.sr)]
                sf.write(base + ".wav", seg, self.sr)
                draw_segment(base + ".png", seg, self.sr, t0, self.tiers, self.tier.get(), (s, e))
        except Exception as ex:
            messagebox.showerror("Export", str(ex))
            return
        self.status.config(text=f"exported {len(idx)} occurrence(s) to {folder}")

    def praat(self):
        sel = self.tree.selection()
        if not sel:
            return
        if not (self.tg_path and self.wav_path):
            messagebox.showinfo("Praat", "Open both a TextGrid and a WAV first.")
            return
        s, e, *_ = self.hits[int(sel[0])]
        pad = float(self.pad.get())
        q = lambda p: p.replace('"', '""')
        script = (
            f'snd = Read from file: "{q(self.wav_path)}"\n'
            f'grid = Read from file: "{q(self.tg_path)}"\n'
            "selectObject: snd, grid\n"
            "View & Edit\n"
            "editor: grid\n"
            f"  Zoom: {max(0.0, s - pad)}, {e + pad}\n"
            f"  Select: {s}, {e}\n"
            "endeditor\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".praat", delete=False, encoding="utf-8") as f:
            f.write(script)
        try:
            proc = subprocess.Popen(["praat", "--send", f.name], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    text=True, start_new_session=True)
        except FileNotFoundError:
            messagebox.showerror("Praat", "'praat' not found in PATH.")
            return
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            return  # still running: leave it alone, never kill it
        if proc.returncode != 0:
            messagebox.showerror("Praat", proc.stderr.read() or "praat --send failed")

def main():
    if len(sys.argv) > 3 or any(a in ("-h", "--help") for a in sys.argv[1:]):
        sys.exit(__doc__)
    root = tk.Tk()
    root.title("TextGrid search")
    app = App(root)
    for a in sys.argv[1:]:
        if a.lower().endswith(".textgrid"):
            app.load_tg(a)
        else:
            app.load_wav(a)
    root.mainloop()


if __name__ == "__main__":
    main()
