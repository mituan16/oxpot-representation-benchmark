# Repository version: run from any directory; paths are relative to the repository root.
# -*- coding: utf-8 -*-
"""(1) Learning-curve fits and crossover estimation; (2) statistical resolution of
model rankings. Uses only the existing 560 run-level results (no retraining).

CPU only (pandas/numpy/scipy). Read-only for inputs; writes ../outputs.

(1) Per method x split, fit MAE(N) = a * N^(-b) + c to the five-seed mean at the
    eight training sizes. Uncertainty by resampling seeds within each size
    (B = 2000). Crossovers between curves are reported only inside the observed
    range [500, 10000]; extrapolated asymptotes c are reported with the caveat that
    N_max = 10000 limits their identifiability.
(2) Within each split x training size, methods are ranked by mean MAE. Every pair
    of methods is compared seed-paired (same split files per seed). A difference is
    called "resolved" when all five seeds agree in sign AND a two-sided paired
    t-test gives p < 0.05. The "top group" is the set of methods not resolved from
    the rank-1 method.
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "results", "raw", "TableS1_full_raw_results.csv")
OUT = os.path.join(REPO, "results", "v2_resolution")
os.makedirs(OUT, exist_ok=True)

EXP_REPRO_SD_MEDIAN = 0.049   # V, from label_fidelity_results.json (Lee replicate SD, median)
LABEL_FIDELITY_LOO = 0.219    # V, from label_fidelity_results.json

raw = pd.read_csv(SRC)
raw = raw[raw.status == "ok"].copy()
METHODS = ["rdkit_rf", "rdkit_xgb", "rdkit_mlp", "morgan_rf", "morgan_xgb", "morgan_mlp", "gine"]
LABEL = dict(zip(raw.method_id, raw.method_label))
SIZES = sorted(raw.train_size.unique())
SEEDS = sorted(raw.seed.unique())
rng = np.random.default_rng(20260925)


def model(n, a, b, c):
    return a * np.power(n, -b) + c


def fit(ns, ys):
    ymin = float(np.min(ys))
    p0 = [ys[0] * ns[0] ** 0.3, 0.3, ymin * 0.5]
    try:
        p, _ = optimize.curve_fit(model, ns, ys, p0=p0,
                                  bounds=([0, 0.01, 0], [np.inf, 2.0, ymin]), maxfev=20000)
        return p
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def crossover(p1, p2, lo=500, hi=10000):
    grid = np.geomspace(lo, hi, 4000)
    d = model(grid, *p1) - model(grid, *p2)
    s = np.sign(d)
    idx = np.where(np.diff(s) != 0)[0]
    return float(grid[idx[0]]) if len(idx) else np.nan


# ---------------- (1) learning curves ----------------
tab = raw.pivot_table(index=["split", "method_id", "train_size"], columns="seed", values="mae")
curves, boots = {}, {}
for split in ["random", "scaffold"]:
    for m in METHODS:
        mat = tab.loc[(split, m)].reindex(SIZES)[SEEDS].to_numpy()  # sizes x seeds
        ns = np.array(SIZES, float)
        p = fit(ns, mat.mean(axis=1))
        bp = []
        for _ in range(2000):
            cols = rng.integers(0, len(SEEDS), size=(len(SIZES), len(SEEDS)))
            y = np.take_along_axis(mat, cols, axis=1).mean(axis=1)
            bp.append(fit(ns, y))
        curves[(split, m)] = p
        boots[(split, m)] = np.array(bp)

fit_rows = []
for (split, m), p in curves.items():
    bp = boots[(split, m)]
    ok = ~np.isnan(bp).any(axis=1)
    q = lambda k: [float(np.quantile(bp[ok, k], .025)), float(np.quantile(bp[ok, k], .975))]
    pred10k = model(10000, *p)
    fit_rows.append({"split": split, "method": LABEL[m], "a": p[0], "b": p[1], "c": p[2],
                     "b_ci95": q(1), "c_ci95": q(2),
                     "c_at_upper_bound_fraction": float(np.mean(np.isclose(bp[ok, 2], bp[ok, 2].max()))),
                     "fit_mae_at_10000": float(pred10k), "observed_mae_at_10000": float(tab.loc[(split, m, 10000)].mean())})

pairs = [("gine", "rdkit_mlp"), ("gine", "rdkit_xgb"), ("gine", "rdkit_rf"), ("rdkit_mlp", "rdkit_xgb")]
cross_rows = []
for split in ["random", "scaffold"]:
    for m1, m2 in pairs:
        n_star = crossover(curves[(split, m1)], curves[(split, m2)])
        b1, b2 = boots[(split, m1)], boots[(split, m2)]
        bs = np.array([crossover(x, y) for x, y in zip(b1, b2)])
        found = ~np.isnan(bs)
        # observed paired sign change between adjacent sizes
        d = (tab.loc[(split, m1)] - tab.loc[(split, m2)]).reindex(SIZES)
        mean_d = d.mean(axis=1)
        cross_rows.append({
            "split": split, "pair": f"{LABEL[m1]} vs {LABEL[m2]}",
            "crossover_N_point": n_star,
            "bootstrap_fraction_with_crossover_in_range": float(found.mean()),
            "crossover_N_ci95": [float(np.quantile(bs[found], .025)), float(np.quantile(bs[found], .975))] if found.sum() > 20 else None,
            "paired_mean_diff_by_size": {int(k): float(v) for k, v in mean_d.items()},
            "seeds_m1_better_by_size": {int(k): int((d.loc[k] < 0).sum()) for k in SIZES},
        })

# ---------------- (2) statistical resolution ----------------
res_rows, top_rows = [], []
for split in ["random", "scaffold"]:
    for n in SIZES:
        sub = raw[(raw.split == split) & (raw.train_size == n)].pivot_table(index="seed", columns="method_id", values="mae")[METHODS]
        order = sub.mean().sort_values().index.tolist()
        pv = {}
        for m1, m2 in itertools.combinations(METHODS, 2):
            d = (sub[m1] - sub[m2]).to_numpy()
            t, p = stats.ttest_rel(sub[m1], sub[m2])
            same_sign = bool((d > 0).all() or (d < 0).all())
            pv[(m1, m2)] = pv[(m2, m1)] = (same_sign and p < 0.05, float(np.mean(d)), float(p), same_sign)
        for k in range(len(order) - 1):
            a, b = order[k], order[k + 1]
            resolved, md, p, ss = pv[(a, b)]
            res_rows.append({"split": split, "train_size": int(n), "rank_pair": f"{k + 1}-{k + 2}",
                             "better": LABEL[a], "worse": LABEL[b], "mean_diff_V": -md if md < 0 else md,
                             "abs_mean_diff_V": abs(md), "p_paired_t": p, "all_seeds_agree": ss, "resolved": resolved,
                             "below_experimental_repro_sd": abs(md) < EXP_REPRO_SD_MEDIAN})
        best = order[0]
        tied = [m for m in order[1:] if not pv[(best, m)][0]]
        top_rows.append({"split": split, "train_size": int(n), "rank1": LABEL[best],
                         "top_group": [LABEL[best]] + [LABEL[m] for m in tied], "top_group_size": 1 + len(tied),
                         "spread_top3_V": float(sub.mean()[order[2]] - sub.mean()[order[0]]),
                         "best_mae_V": float(sub.mean()[best])})

res = pd.DataFrame(res_rows)
top = pd.DataFrame(top_rows)
summary = {
    "adjacent_rank_pairs": int(len(res)),
    "adjacent_pairs_resolved": int(res.resolved.sum()),
    "adjacent_pairs_resolved_fraction": float(res.resolved.mean()),
    "adjacent_pairs_below_experimental_repro_sd": int(res.below_experimental_repro_sd.sum()),
    "rank1_cells": int(len(top)),
    "rank1_cells_with_unique_winner": int((top.top_group_size == 1).sum()),
    "mean_top_group_size": float(top.top_group_size.mean()),
    "experimental_repro_sd_median_V": EXP_REPRO_SD_MEDIAN,
    "label_fidelity_loo_mae_V": LABEL_FIDELITY_LOO,
}

pd.DataFrame(fit_rows).to_csv(os.path.join(OUT, "learning_curve_fits.csv"), index=False)
pd.DataFrame(cross_rows).to_json(os.path.join(OUT, "crossovers.json"), orient="records", indent=2)
res.to_csv(os.path.join(OUT, "rank_resolution_adjacent.csv"), index=False)
top.to_json(os.path.join(OUT, "rank_top_groups.json"), orient="records", indent=2)
json.dump(summary, open(os.path.join(OUT, "resolution_summary.json"), "w"), indent=2)
np.save(os.path.join(OUT, "_curve_params.npy"), {str(k): v for k, v in curves.items()}, allow_pickle=True)

print(json.dumps(summary, indent=2))
print(pd.DataFrame(fit_rows)[["split", "method", "b", "b_ci95", "c", "c_ci95", "c_at_upper_bound_fraction", "fit_mae_at_10000", "observed_mae_at_10000"]].to_string(index=False))
for r in cross_rows:
    print(r["split"], r["pair"], "N*=", r["crossover_N_point"], "frac", round(r["bootstrap_fraction_with_crossover_in_range"], 3), "CI", r["crossover_N_ci95"], "seeds m1 better:", r["seeds_m1_better_by_size"])
print(top[["split", "train_size", "rank1", "top_group_size", "top_group", "spread_top3_V", "best_mae_V"]].to_string(index=False))
