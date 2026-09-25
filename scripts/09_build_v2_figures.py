# Repository version: run from any directory; paths are relative to the repository root.
# -*- coding: utf-8 -*-
"""Figures 5 and 6 for the v2 manuscript, drawn from the analysis outputs.
CPU only; read-only for inputs; writes ../figures (PNG/PDF/SVG)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "v2_resolution")
FIG = os.path.join(REPO, "figures", "main")
SRC = os.path.join(REPO, "results", "raw", "TableS1_full_raw_results.csv")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 300})
METHODS = ["rdkit_rf", "rdkit_xgb", "rdkit_mlp", "morgan_rf", "morgan_xgb", "morgan_mlp", "gine"]
LABELS = {"rdkit_rf": "RDKit + RF", "rdkit_xgb": "RDKit + XGB", "rdkit_mlp": "RDKit + MLP",
          "morgan_rf": "Morgan + RF", "morgan_xgb": "Morgan + XGB", "morgan_mlp": "Morgan + MLP",
          "gine": "2D graph + GINE"}
COL = {"rdkit_rf": "#8fb3d9", "rdkit_xgb": "#4a7fb5", "rdkit_mlp": "#1f4e8c", "gine": "#c0392b"}


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------- Figure 5: label fidelity and error scales ----------------
lf = json.load(open(os.path.join(OUT, "label_fidelity_results.json")))
pairs = pd.read_csv(os.path.join(OUT, "label_fidelity_matched_pairs.csv"))
top = pd.DataFrame(json.load(open(os.path.join(OUT, "rank_top_groups.json"))))
ag = lf["agreement"]

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
x, y = pairs["eox"].to_numpy(), pairs["eox_exp_vs_agcl"].to_numpy()
ax.scatter(x, y, s=18, color="#34495e", alpha=0.8, edgecolor="none", zorder=3)
lo, hi = min(x.min(), y.min()) - 0.15, max(x.max(), y.max()) + 0.15
ax.plot([lo, hi], [lo, hi], color="#999", lw=0.8, ls="--", label="y = x")
a, b = ag["linear_recalibration_in_sample"]["slope"], ag["linear_recalibration_in_sample"]["intercept"]
gx = np.linspace(lo, hi, 50)
ax.plot(gx, a * gx + b, color="#c0392b", lw=1.2, label=f"linear recalibration (slope {a:.2f})")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("OxPot label / V vs Ag/AgCl (sat. KCl)")
ax.set_ylabel("Experimental $E_{ox}$ (acetonitrile) / V vs Ag/AgCl")
loo = ag["linear_recalibration_loo"]
ax.text(0.03, 0.97,
        f"n = {ag['n']}, r = {ag['pearson_r']:.2f}\n"
        f"offset (reference-converted): {ag['reference_converted_only']['mean_signed']:+.2f} V\n"
        f"LOO-recalibrated MAE = {loo['mae']:.3f} V\n(95% CI {loo['mae_ci95'][0]:.3f}–{loo['mae_ci95'][1]:.3f})",
        transform=ax.transAxes, va="top", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.95), zorder=5)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.set_title("A  OxPot labels vs independent experiment", loc="left", fontsize=10, fontweight="bold")

ax = axes[1]
b10r = top[(top.split == "random") & (top.train_size == 10000)].best_mae_V.iloc[0]
b10s = top[(top.split == "scaffold") & (top.train_size == 10000)].best_mae_V.iloc[0]
b5r = top[(top.split == "random") & (top.train_size == 500)].best_mae_V.iloc[0]
b5s = top[(top.split == "scaffold") & (top.train_size == 500)].best_mae_V.iloc[0]
spread = top.spread_top3_V.median()
rep = lf["experimental_reproducibility"]
rows = [
    ("Spread of top-3 methods (median over cells)", spread, "#bdc3c7", "model vs label"),
    ("Best model MAE, random, N = 10,000", b10r, "#1f4e8c", "model vs label"),
    ("Best model MAE, scaffold, N = 10,000", b10s, "#1f4e8c", "model vs label"),
    ("Best model MAE, random, N = 500", b5r, "#1f4e8c", "model vs label"),
    ("Best model MAE, scaffold, N = 500", b5s, "#1f4e8c", "model vs label"),
    ("Replicate SD between literature reports (median)", rep["sd_median"], "#27ae60", "experiment vs experiment"),
    ("Source-paper calibration RMSE (in-sample)", lf["source_paper_calibration"]["rmse_V"], "#e67e22", "label vs experiment"),
    ("Label vs experiment, LOO-recalibrated MAE", loo["mae"], "#c0392b", "label vs experiment"),
]
ypos = np.arange(len(rows))[::-1]
ci = loo["mae_ci95"]
for yi, (lab, v, c, _) in zip(ypos, rows):
    ax.barh(yi, v, color=c, height=0.62)
    xt = (ci[1] + 0.004) if yi == ypos[-1] else (v + 0.004)
    ax.text(xt, yi, f"{v:.3f}", va="center", fontsize=7.6)
ax.plot(ci, [ypos[-1]] * 2, color="k", lw=1.0)
ax.plot([ci[0]] * 2, [ypos[-1] - 0.15, ypos[-1] + 0.15], color="k", lw=1.0)
ax.plot([ci[1]] * 2, [ypos[-1] - 0.15, ypos[-1] + 0.15], color="k", lw=1.0)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=7.8)
ax.set_xlabel("Error scale / V")
ax.set_xlim(0, 0.34)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#bdc3c7", label="difference between models"),
                   Patch(color="#1f4e8c", label="model vs DFT-derived label"),
                   Patch(color="#27ae60", label="experiment vs experiment"),
                   Patch(color="#e67e22", label="label vs experiment (source paper)"),
                   Patch(color="#c0392b", label="label vs independent experiment (this work)")],
          frameon=False, fontsize=7.2, loc="upper right")
ax.set_title("B  Model errors against label and experimental error scales", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout()
save(fig, "Figure5_label_fidelity_error_scales")

# ---------------- Figure 6: crossovers and statistical resolution ----------------
raw = pd.read_csv(SRC)
raw = raw[raw.status == "ok"]
params = np.load(os.path.join(OUT, "_curve_params.npy"), allow_pickle=True).item()
cross = pd.DataFrame(json.load(open(os.path.join(OUT, "crossovers.json"))))
SIZES = sorted(raw.train_size.unique())

fig = plt.figure(figsize=(10.4, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
for k, split in enumerate(["random", "scaffold"]):
    ax = fig.add_subplot(gs[0, k])
    grid = np.geomspace(450, 11000, 200)
    for m in ["rdkit_rf", "rdkit_xgb", "rdkit_mlp", "gine"]:
        sub = raw[(raw.split == split) & (raw.method_id == m)].groupby("train_size").mae
        mu, sd = sub.mean().reindex(SIZES), sub.std().reindex(SIZES)
        ax.errorbar(SIZES, mu, yerr=sd, fmt="o", ms=3.5, color=COL[m], capsize=2, lw=0.8, label=LABELS[m])
        p = params[str((split, m))]
        ax.plot(grid, p[0] * grid ** (-p[1]) + p[2], color=COL[m], lw=1.0, alpha=0.9)
    row = cross[(cross.split == split) & (cross.pair == "2D graph + GINE vs RDKit + MLP")].iloc[0]
    if row.crossover_N_ci95 is not None and not np.isnan(row.crossover_N_point):
        ax.axvspan(*row.crossover_N_ci95, color="#c0392b", alpha=0.10, lw=0)
        ax.axvline(row.crossover_N_point, color="#c0392b", lw=0.8, ls="--")
        ax.text(row.crossover_N_point * 1.05, ax.get_ylim()[1] if False else 0.97, "", transform=ax.get_xaxis_transform())
        ax.text(0.97, 0.96, f"GINE–RDKit+MLP crossover\nN* ≈ {row.crossover_N_point:,.0f}\n(95% CI {row.crossover_N_ci95[0]:,.0f}–{row.crossover_N_ci95[1]:,.0f})",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color="#c0392b")
    ax.set_xscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels(["" if s == 8000 else (f"{s // 1000}k" if s >= 1000 else str(s)) for s in SIZES], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("Training molecules N")
    ax.set_ylabel("Test MAE / V")
    ax.set_title(f"{'AB'[k]}  {split.capitalize()} split: power-law fits", loc="left", fontsize=10, fontweight="bold")
    if k == 0:
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")

tops = pd.DataFrame(json.load(open(os.path.join(OUT, "rank_top_groups.json"))))
for k, split in enumerate(["random", "scaffold"]):
    ax = fig.add_subplot(gs[1, k])
    sub = raw[raw.split == split].groupby(["method_id", "train_size"]).mae.mean().unstack()[SIZES].reindex(METHODS)
    ranks = sub.rank(axis=0).to_numpy()
    for j, n in enumerate(SIZES):
        t = tops[(tops.split == split) & (tops.train_size == n)].iloc[0]
        group = set(t.top_group)
        for i, m in enumerate(METHODS):
            inside = LABELS[m] in group
            is1 = ranks[i, j] == 1
            face = "#c0392b" if (inside and t.top_group_size == 1) else ("#f1a9a0" if inside else "white")
            ax.add_patch(plt.Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, facecolor=face,
                                       edgecolor="#555" if is1 else "#ddd", lw=1.4 if is1 else 0.6))
            ax.text(j, i, int(ranks[i, j]), ha="center", va="center", fontsize=7.2,
                    color="white" if (inside and t.top_group_size == 1) else "#333")
    ax.set_xlim(-0.6, len(SIZES) - 0.4); ax.set_ylim(len(METHODS) - 0.4, -0.6)
    ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([f"{s // 1000}k" if s >= 1000 else str(s) for s in SIZES], fontsize=7.5)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([LABELS[m] for m in METHODS], fontsize=7.8)
    ax.set_xlabel("Training molecules N")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{'CD'[k]}  {split.capitalize()} split: statistical top group", loc="left", fontsize=10, fontweight="bold")
fig.text(0.5, 0.005,
         "C–D: numbers are mean-MAE ranks; dark outline = rank 1; shaded = not statistically separable from rank 1 "
         "(all five seeds agree in sign and paired t-test p < 0.05 required to separate); dark red = unique winner.",
         ha="center", fontsize=7.4)
save(fig, "Figure4_crossovers_statistical_resolution")
