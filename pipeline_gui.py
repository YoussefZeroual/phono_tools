#!/usr/bin/env python3
"""GUI pipeline: chunk -> transcribe -> MFA align -> merge -> diarize -> add speakers.
Put this file in the same folder as chunk_audio.py, merge_textgrids.py,
diarize.py and add_speakers.py."""
import os
import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

HERE = Path(__file__).resolve().parent
STEPS = ["1. Chunk", "2. Transcribe", "3. MFA align", "4. Merge TextGrids",
         "5. Diarize", "6. Add speaker tiers"]


def wav_duration(path):
    import wave
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # non-PCM wav
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        return len(data) / sr


def empty_wavs(folder):
    import numpy as np
    from scipy.io import wavfile
    bad = []
    for w in sorted(Path(folder).glob("*.wav")):
        try:
            _, data = wavfile.read(str(w))
            if len(data) == 0 or not np.any(data):
                bad.append(w.name)
        except Exception as e:
            bad.append(f"{w.name} (unreadable: {e})")
    return bad


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio → aligned TextGrid")
        self.q = queue.Queue()
        self.v = {k: tk.StringVar(value=d) for k, d in {
            "wav": "", "n": "10", "work": "", "chunks": "", "aligned": "",
            "model_asr": "/home/joe/models/CrisperWhisper2.0-GGUF/crisperwhisper-2-large-q4_k.gguf",
            "family": "", "lang": "fr", "backend": "cuda", "cli_extra": "",
            "beam": "100", "conda": "aligner", "dict": "french_mfa", "model": "french_mfa",
            "py_diar": sys.executable, "device": "cuda",
            "nspk": "", "token": os.environ.get("HF_TOKEN", ""),
            "start": STEPS[0],
        }.items()}
        f = ttk.Frame(self, padding=8)
        f.pack(fill="both", expand=True)
        self.r = 0

        def row(label, key, browse=None, show=None):
            ttk.Label(f, text=label).grid(row=self.r, column=0, sticky="w", pady=2)
            ttk.Entry(f, textvariable=self.v[key], width=55, show=show).grid(
                row=self.r, column=1, sticky="we", padx=4)
            if browse:
                ttk.Button(f, text="…", width=3, command=browse).grid(row=self.r, column=2)
            self.r += 1

        ttk.Label(f, text="Start at step").grid(row=self.r, column=0, sticky="w", pady=2)
        ttk.Combobox(f, textvariable=self.v["start"], values=STEPS, state="readonly").grid(
            row=self.r, column=1, sticky="w", padx=4)
        self.r += 1
        row("WAV (original)", "wav", self.pick_wav)
        row("Work dir", "work", lambda: self.pick_dir("work"))
        row("Chunks dir (default: work/chunks)", "chunks", lambda: self.pick_dir("chunks"))
        row("Aligned dir (default: work/aligned)", "aligned", lambda: self.pick_dir("aligned"))
        row("Nb chunks", "n")
        row("ASR model (.gguf)", "model_asr", self.pick_model)
        row("ASR family (--family)", "family")
        row("Language", "lang")
        row("ASR backend", "backend")
        row("audiocpp_cli extra args", "cli_extra")
        row("Conda env (MFA)", "conda")
        row("MFA beam", "beam")
        row("MFA dictionary", "dict")
        row("MFA acoustic model", "model")
        row("Python for diarize", "py_diar")
        row("Device (diarize)", "device")
        row("Nb speakers (opt.)", "nspk")
        row("HF token", "token", show="*")
        self.keep = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Keep MFA word/phone tiers in final file", variable=self.keep).grid(
            row=self.r, column=1, sticky="w")
        self.r += 1
        self.ask = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Ask confirmation between steps", variable=self.ask).grid(
            row=self.r, column=1, sticky="w")
        self.r += 1
        self.btn = ttk.Button(f, text="Run", command=self.start)
        self.btn.grid(row=self.r, column=1, sticky="w", pady=6)
        self.r += 1
        self.log = scrolledtext.ScrolledText(f, height=18, width=90, state="disabled")
        self.log.grid(row=self.r, column=0, columnspan=3, sticky="nsew")
        f.columnconfigure(1, weight=1)
        f.rowconfigure(self.r, weight=1)
        self.after(100, self.poll)

    # ---------- UI helpers ----------
    def pick_wav(self):
        p = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if p:
            self.v["wav"].set(p)
            if not self.v["work"].get():
                self.v["work"].set(str(Path(p).with_suffix("")) + "_work")
            try:
                dur = wav_duration(p)
                n = max(1, round(dur / 600))
                self.v["n"].set(str(n))
                self.say(f"Duration: {dur / 60:.1f} min -> suggested {n} chunk(s) (~10 min each)\n")
            except Exception as e:
                self.say(f"Could not read duration: {e}\n")

    def pick_dir(self, key):
        p = filedialog.askdirectory()
        if p:
            self.v[key].set(p)

    def pick_model(self):
        p = filedialog.askopenfilename(filetypes=[("GGUF", "*.gguf")])
        if p:
            self.v["model_asr"].set(p)

    def say(self, s):
        self.q.put(s)

    def poll(self):
        try:
            while True:
                s = self.q.get_nowait()
                if s == "\0done":
                    self.btn.config(state="normal")
                    continue
                self.log.config(state="normal")
                self.log.insert("end", s)
                self.log.see("end")
                self.log.config(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def confirm(self, msg):
        """Ask from the worker thread, block until the user answers."""
        ev, ans = threading.Event(), []

        def ask():
            ans.append(messagebox.askyesno("Pipeline", msg, parent=self))
            ev.set()

        self.after(0, ask)
        ev.wait()
        return ans[0]

    def start(self):
        try:
            n = int(self.v["n"].get())
            assert n >= 1 and Path(self.v["wav"].get()).is_file()
        except Exception:
            messagebox.showerror("Error", "Choose the WAV file and a valid number of chunks.")
            return
        self.btn.config(state="disabled")
        threading.Thread(target=self.run, args=(n,), daemon=True).start()

    def sh(self, cmd, shell=False, env=None):
        shown = cmd if shell else " ".join(shlex.quote(str(c)) for c in cmd)
        self.say(f"\n$ {shown}\n")
        p = subprocess.Popen(cmd if shell else [str(c) for c in cmd], shell=shell, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            self.say(line)
        if p.wait() != 0:
            raise RuntimeError(f"exit code {p.returncode}")

    # ---------- pipeline ----------
    def run(self, n):
        g = lambda k: self.v[k].get().strip()
        try:
            wav = Path(g("wav")).resolve()
            work = Path(g("work") or str(wav.with_suffix("")) + "_work").resolve()
            chunks = Path(g("chunks")).resolve() if g("chunks") else work / "chunks"
            aligned = Path(g("aligned")).resolve() if g("aligned") else work / "aligned"
            merged = work / f"{wav.stem}.merged.TextGrid"
            diar = work / f"{wav.stem}.diar.json"
            final = work / f"{wav.stem}.final.TextGrid"
            py = sys.executable
            work.mkdir(parents=True, exist_ok=True)

            def chunk():
                self.sh([py, HERE / "chunk_audio.py", "-i", wav, "-n", n, "-o", chunks])
                bad = empty_wavs(chunks)
                if bad:
                    msg = "⚠️ Empty/silent chunk(s): " + ", ".join(bad)
                    self.say("\n" + msg + "\n")
                    return msg
                self.say("\nNo empty chunks.\n")

            def transcribe():
                wavs = sorted(chunks.glob("*.wav"))
                if not wavs:
                    raise RuntimeError(f"no wav files in {chunks}")
                for k, w in enumerate(wavs, 1):
                    self.say(f"\n[{k}/{len(wavs)}] {w.name}")
                    cmd = ["audiocpp_cli", "--task", "asr", "--model", g("model_asr"),
                           "--backend", g("backend"), "--language", g("lang"),
                           "--audio", w, "--text-out", w.with_suffix(".lab"), "--log"]
                    if g("family"):
                        cmd[3:3] = ["--family", g("family")]
                    cmd += shlex.split(g("cli_extra"))
                    self.sh(cmd)
                nl = len(list(chunks.glob("*.lab")))
                self.say(f"\n{nl} .lab file(s) for {len(wavs)} wav file(s) in {chunks}\n")
                if nl < len(wavs):
                    return "⚠️ Some .lab files are missing."

            def align():
                mfa = (f"conda run --no-capture-output -n {shlex.quote(g('conda'))} mfa align "
                       f"{shlex.quote(str(chunks))} {shlex.quote(g('dict'))} {shlex.quote(g('model'))} "
                       f"{shlex.quote(str(aligned))} --clean --single_speaker --beam {shlex.quote(g('beam') or '100')}")
                self.sh(["bash", "-lc", mfa])

            def merge():
                self.sh([py, HERE / "merge_textgrids.py", "-i", aligned, "-o", merged])

            def diarize():
                cmd = [g("py_diar") or py, HERE / "diarize.py", wav, "-o", diar,
                       "--device", g("device")]
                if g("nspk"):
                    cmd += ["--num-speakers", g("nspk")]
                env = dict(os.environ)
                if g("token"):
                    env["HF_TOKEN"] = g("token")
                self.sh(cmd, env=env)

            def speakers():
                cmd = [py, HERE / "add_speakers.py", merged, diar, "-o", final]
                if self.keep.get():
                    cmd.append("--keep-phones")
                self.sh(cmd)

            funcs = [chunk, transcribe, align, merge, diarize, speakers]
            first = STEPS.index(g("start"))
            for i in range(first, len(funcs)):
                self.say(f"\n== {STEPS[i]} ==")
                note = funcs[i]() or ""
                self.say(f"\n✔ {STEPS[i]} done\n")
                if i < len(funcs) - 1 and (self.ask.get() or note):
                    if not self.confirm((f"{note}\n\n" if note else "") + f"{STEPS[i]} finished.\nContinue with {STEPS[i + 1]}?"):
                        self.say(f"Stopped before {STEPS[i + 1]}.\n")
                        return
            self.say(f"\n✅ Done: {final}\n")
        except Exception as e:
            self.say(f"\n❌ Failed: {e}\n")
        finally:
            self.say("\0done")


if __name__ == "__main__":
    App().mainloop()
