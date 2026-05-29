#!/usr/bin/env python
"""Assemble the reconstruction panel from results/ produced by reproduce.sh.
Rows = the 5 representative configurations; columns = GT | DeepCSI | CasVAE | LaMP.
rMSE/SSIM (from results.csv) annotated under each reconstruction.
Usage: python scripts/plot_results.py   (writes figures/reconstructions.png)
"""
import os, glob, csv, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

BASE = os.path.join(os.path.dirname(__file__), "..")
os.chdir(BASE)

def to_grid(eps, shape):
    if hasattr(eps, "cpu"): eps = eps.cpu().numpy()
    eps = np.asarray(eps, np.float32)
    if eps.shape != shape:
        t = torch.from_numpy(eps).float().unsqueeze(0).unsqueeze(0)
        eps = interpolate(t, size=shape, mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    return eps

def load(d, shape):
    f = sorted([x for x in os.listdir(d) if x.startswith("testset_") and x.endswith(".npy") and "_params" not in x])
    return to_grid(np.load(d + "/" + f[-1], allow_pickle=True).item()["epsilon"], shape)

def rc(path):
    out = {}
    if not os.path.exists(path): return out
    for row in csv.DictReader(open(path)):
        s, t = row["Shape"], row["Task"]
        m, tag = (s, t) if s in ("dc", "casvae", "lamp") else (t, s)
        out[(tag, m)] = (float(row["Model Misfit"]), float(row["SSIM"]))
    return out

# (label, glob pattern with {m}, gt, results.csv, csv-tag, vmax)
CASES = [
 ("Fig3 TwinCircle 8x12", "results/fig3_8x12/fig3_{m}_*", "data/testcases/epsilon_twinCircle.npy", "results/fig3_8x12/results.csv", "twinCircle", 2),
 ("Fig6 Star 16x32",      "results/star_16x32/star_{m}_*", "data/testcases/epsilon_Star.npy", "results/star_16x32/results.csv", "Star", 2),
 ("Fresnel FoamTwinDiel 8x241", "results/fresnel_foamtwindiel/ft_8x241_{m}_*", "data/Fresnel/FoamTwinDielTM.npy", "results/fresnel_foamtwindiel/results.csv", "8x241", 3),
 ("Fresnel FoamTwinDiel 4x8",   "results/fresnel_foamtwindiel/ft_4x8_{m}_*",   "data/Fresnel/FoamTwinDielTM.npy", "results/fresnel_foamtwindiel/results.csv", "4x8", 3),
 ("Fresnel FoamTwinDiel 4x4",   "results/fresnel_foamtwindiel/ft_4x4_{m}_*",   "data/Fresnel/FoamTwinDielTM.npy", "results/fresnel_foamtwindiel/results.csv", "4x4", 3),
]
M = [("GT", None), ("DeepCSI", "dc"), ("CasVAE", "casvae"), ("LaMP", "lamp")]
fig, axes = plt.subplots(len(CASES), len(M), figsize=(len(M)*2.4, len(CASES)*2.4))
for r, (lab, pat, gtp, csvp, tag, vmax) in enumerate(CASES):
    gt = np.load(gtp).astype(np.float32); met = rc(csvp)
    for c, (ml, mk) in enumerate(M):
        ax = axes[r, c]
        if mk is None:
            img, sub = gt, ""
        else:
            dd = glob.glob(pat.format(m=mk)); img = load(dd[0], gt.shape) if dd else np.zeros_like(gt)
            mv = met.get((tag, mk)); sub = f"r{mv[0]:.3f} s{mv[1]:.3f}" if mv else ""
        im = ax.imshow(img, cmap="jet", vmin=1, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(ml, fontsize=11)
        if c == 0: ax.set_ylabel(lab, fontsize=8)
        if sub: ax.set_xlabel(sub, fontsize=7)
    fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.012, pad=0.01)
os.makedirs("figures", exist_ok=True)
out = "figures/reconstructions.png"; fig.savefig(out, dpi=125, bbox_inches="tight")
print("saved:", os.path.abspath(out))
