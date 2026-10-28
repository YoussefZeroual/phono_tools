import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModelForImageClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32


def extract(folder, model_name, progress):
    files = sorted(Path(folder).glob("*.png"))
    if not files:
        raise ValueError("No PNG found in folder")
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(DEVICE).eval()
    id2label = model.config.id2label
    feats, labels, confs = [], [], []
    for i in range(0, len(files), BATCH):
        imgs = [Image.open(f).convert("RGB") for f in files[i:i + BATCH]]
        inp = proc(imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        h = out.hidden_states[-1]
        h = h.mean(dim=(2, 3)) if h.dim() == 4 else h[:, 0]  # CNN: GAP, ViT: CLS
        probs = out.logits.softmax(-1)
        conf, idx = probs.max(-1)
        feats.append(h.cpu().numpy())
        labels += [id2label[int(k)] for k in idx]
        confs += conf.cpu().tolist()
        progress(min(i + BATCH, len(files)), len(files))
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
        ttk.Entry(top, textvariable=self.model, width=30).pack(side="left", padx=4)
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
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        body.add(left, weight=1)

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        body.add(self.canvas.get_tk_widget(), weight=2)

    def pick(self):
        d = filedialog.askdirectory()
        if d:
            self.folder.set(d)

    def run(self):
        if not self.folder.get():
            return messagebox.showwarning("Missing", "Pick a folder")
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
            self.tree.insert("", "end", values=(r.cluster, r.file, r.label, f"{r.conf:.2f}"))
        self.ax.clear()
        sc = self.ax.scatter(df.pc1, df.pc2, c=df.cluster, cmap="tab10", s=18)
        self.ax.legend(*sc.legend_elements(), title="cluster", loc="best", fontsize=8)
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.set_title(f"{len(df)} images, {df.cluster.nunique()} clusters")
        self.fig.tight_layout()
        self.canvas.draw()
        self.status.set("Done")

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
