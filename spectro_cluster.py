import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import (AutoFeatureExtractor, AutoImageProcessor, AutoModel,
                          AutoModelForAudioClassification,
                          AutoModelForImageClassification, ClapModel)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32


PRESETS = [
    "facebook/convnext-tiny-224",
    "facebook/dinov2-base",
    "openai/whisper-small",
    "facebook/wav2vec2-xls-r-300m",
    "facebook/hubert-base-ls960",
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    "laion/clap-htsat-unfused",
]


def kind_of(name):
    n = name.lower()
    if "dinov2" in n:
        return "dino"
    if "whisper" in n:
        return "whisper"
    if "clap" in n:
        return "clap"
    if "/ast" in n or "audio-spectrogram" in n:
        return "ast"
    if any(k in n for k in ("wav2vec2", "hubert", "wavlm")):
        return "ssl"
    return "image"


def load_wav(path, sr):
    x, s = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if int(s) != int(sr):
        g = int(np.gcd(int(s), int(sr)))
        x = resample_poly(x, int(sr) // g, int(s) // g).astype("float32")
    return x


def extract(folder, model_name, progress):
    files = sorted(Path(folder).glob("*.png"))
    if not files:
        raise ValueError("No PNG found in folder")
    kind = kind_of(model_name)
    labels, confs, feats = [""] * len(files), [0.0] * len(files), []

    if kind in ("image", "dino"):  # works on the PNGs
        proc = AutoImageProcessor.from_pretrained(model_name)
        cls = AutoModel if kind == "dino" else AutoModelForImageClassification
        model = cls.from_pretrained(model_name).to(DEVICE).eval()
        for i in range(0, len(files), BATCH):
            imgs = [Image.open(f).convert("RGB") for f in files[i:i + BATCH]]
            inp = proc(imgs, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                if kind == "dino":
                    h = model(**inp).last_hidden_state[:, 0]
                else:
                    out = model(**inp, output_hidden_states=True)
                    h = out.hidden_states[-1]
                    h = h.mean(dim=(2, 3)) if h.dim() == 4 else h[:, 0]  # CNN: GAP, ViT: CLS
                    conf, idx = out.logits.softmax(-1).max(-1)
                    for j, (c, k) in enumerate(zip(conf.tolist(), idx.tolist())):
                        labels[i + j] = model.config.id2label[k]
                        confs[i + j] = c
            feats.append(h.cpu().numpy())
            progress(min(i + BATCH, len(files)), len(files))
        return files, np.vstack(feats), labels, confs

    # audio models: use the wavs matching the PNG names
    wavs = [f.with_suffix(".wav") for f in files]
    miss = [w.name for w in wavs if not w.exists()]
    if miss:
        raise ValueError(f"Missing wav: {miss[0]} (+{len(miss) - 1} more)")
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    sr = fe.sampling_rate
    if kind == "ast":
        model = AutoModelForAudioClassification.from_pretrained(model_name)
    elif kind == "clap":
        model = ClapModel.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    if kind == "whisper":
        model = model.encoder
    model = model.to(DEVICE).eval()
    for i, w in enumerate(wavs):
        x = load_wav(w, sr)
        with torch.no_grad():
            if kind == "whisper":  # 30 s window, pool only over real audio frames (50 Hz)
                inp = fe(x, sampling_rate=sr, return_tensors="pt").input_features.to(DEVICE)
                n = min(1500, max(1, int(len(x) / sr * 50)))
                h = model(inp).last_hidden_state[:, :n].mean(1)
            else:
                inp = fe(x, sampling_rate=sr, return_tensors="pt").to(DEVICE)
                if kind == "ast":
                    out = model(**inp, output_hidden_states=True)
                    h = out.hidden_states[-1][:, :2].mean(1)
                    c, k = out.logits.softmax(-1).max(-1)
                    labels[i] = model.config.id2label[int(k)]
                    confs[i] = float(c)
                elif kind == "clap":
                    h = model.get_audio_features(**inp)
                    h = getattr(h, "pooler_output", h)
                else:
                    h = model(**inp).last_hidden_state.mean(1)
        feats.append(h.cpu().numpy())
        progress(i + 1, len(wavs))
    return files, np.vstack(feats), labels, confs


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Spectrogram clustering")
        self.geometry("1200x700")
        self.df = None
        self.folder = tk.StringVar()
        self.model = tk.StringVar(value="facebook/convnext-tiny-224")
        self.k = tk.IntVar(value=5)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Button(top, text="Folder…", command=self.pick).pack(side="left")
        ttk.Entry(top, textvariable=self.folder, width=40).pack(side="left", padx=4)
        ttk.Label(top, text="Model").pack(side="left")
        ttk.Combobox(top, textvariable=self.model, values=PRESETS, width=38).pack(side="left", padx=4)
        ttk.Label(top, text="n clusters").pack(side="left")
        ttk.Spinbox(top, from_=2, to=100, textvariable=self.k, width=4).pack(side="left", padx=4)
        self.run_btn = ttk.Button(top, text="Run", command=self.run)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(top, text="Save PNG", command=self.save_png).pack(side="right", padx=2)
        ttk.Button(top, text="Save CSV", command=self.save_csv).pack(side="right", padx=2)

        self.status = tk.StringVar()
        ttk.Label(self, textvariable=self.status).pack(fill="x", padx=6)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(body)
        cols = ("cluster", "file", "label", "conf")
        self.tree = ttk.Treeview(left, columns=cols, show="headings")
        for c, w in zip(cols, (60, 220, 120, 60)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        sb = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self.play)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        body.add(left, weight=1)

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.sc = self.annot = None
        self.canvas.mpl_connect("motion_notify_event", self.hover)
        self.canvas.mpl_connect("button_press_event", self.click)
        body.add(self.canvas.get_tk_widget(), weight=2)

        self.spec = ttk.Label(body, anchor="n")
        body.add(self.spec, weight=1)

    def pick(self):
        d = filedialog.askdirectory()
        if d:
            self.folder.set(d)

    def run(self):
        if not self.folder.get():
            return messagebox.showwarning("Missing", "Pick a folder")
        self.run_folder = self.folder.get()
        self.run_btn.state(["disabled"])
        threading.Thread(target=self.work, daemon=True).start()

    def work(self):
        try:
            prog = lambda a, b: self.after(0, self.status.set, f"Classifying {a}/{b}…")
            files, X, labels, confs = extract(self.folder.get(), self.model.get(), prog)
            k = self.k.get()
            if k > len(files):
                raise ValueError(f"n clusters ({k}) > number of images ({len(files)})")
            self.after(0, self.status.set, "KMeans…")
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
            # relabel clusters by size (0 = largest)
            order = pd.Series(km.labels_).value_counts().index.tolist()
            remap = {old: new for new, old in enumerate(order)}
            cl = np.array([remap[c] for c in km.labels_])
            dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
            xy = PCA(n_components=2).fit_transform(X)
            df = pd.DataFrame({
                "cluster": cl, "file": [f.name for f in files], "label": labels,
                "conf": np.round(confs, 4), "dist_to_centroid": np.round(dist, 4),
                "pc1": xy[:, 0], "pc2": xy[:, 1],
            }).sort_values(["cluster", "dist_to_centroid"]).reset_index(drop=True)
            self.after(0, self.show, df)
        except Exception as e:
            self.after(0, messagebox.showerror, "Error", str(e))
            self.after(0, self.status.set, "Error")
        finally:
            self.after(0, self.run_btn.state, ["!disabled"])

    def show(self, df):
        self.df = df
        self.tree.delete(*self.tree.get_children())
        for r in df.itertuples():
            self.tree.insert("", "end", values=(r.cluster, r.file, r.label, f"{r.conf:.2f}" if r.label else ""))
        self.ax.clear()
        sc = self.ax.scatter(df.pc1, df.pc2, c=df.cluster, cmap="tab10", s=18)
        self.sc = sc
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"), visible=False)
        self.ax.legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.set_title(f"{len(df)} images, {df.cluster.nunique()} clusters")
        self.fig.tight_layout()
        self.canvas.draw()
        self.status.set("Done")

    def hover(self, event):
        if self.sc is None or event.inaxes != self.ax:
            return
        hit, ind = self.sc.contains(event)
        if hit:
            i = ind["ind"][0]
            self.annot.xy = self.sc.get_offsets()[i]
            self.annot.set_text(self.df.file.iloc[i])
            self.annot.set_visible(True)
        elif self.annot.get_visible():
            self.annot.set_visible(False)
        else:
            return
        self.canvas.draw_idle()

    def click(self, event):
        if self.sc is None or event.inaxes != self.ax or event.button != 1:
            return
        hit, ind = self.sc.contains(event)
        if not hit:
            return
        item = self.tree.get_children()[ind["ind"][0]]
        if self.tree.selection() == (item,):
            self.play()  # already selected: no select event, so replay manually
        else:
            self.tree.selection_set(item)  # triggers play + spectrogram
        self.tree.see(item)
        self.tree.focus(item)

    def show_spec(self, name):
        img = Image.open(Path(self.run_folder) / name).convert("RGB")
        img.thumbnail((450, 450))
        self.tkimg = ImageTk.PhotoImage(img)  # keep a reference
        self.spec.configure(image=self.tkimg)

    def play(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        name = self.tree.item(sel[0], "values")[1]
        self.show_spec(name)
        wav = Path(self.run_folder) / (Path(name).stem + ".wav")
        sd.stop()
        if not wav.exists():
            return self.status.set(f"Missing: {wav.name}")
        data, sr = sf.read(wav)
        sd.play(data, sr)
        self.status.set(f"Playing {wav.name}")

    def save_png(self):
        if self.df is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if p:
            self.fig.savefig(p, dpi=200)

    def save_csv(self):
        if self.df is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self.df.to_csv(p, index=False)


if __name__ == "__main__":
    App().mainloop()
