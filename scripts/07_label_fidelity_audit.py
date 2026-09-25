# Repository version: run from any directory; paths are relative to the repository root.
# -*- coding: utf-8 -*-
"""Label-fidelity audit: OxPot labels vs independent experimental oxidation
potentials (Lee et al., acetonitrile, vs SCE).

CPU only; imports RDKit/pandas/numpy only (no torch). Read-only for all inputs.
Writes outputs into ../outputs and ../figures.

Molecule matching is done on standardized parent structures:
  RDKit Cleanup -> FragmentParent (largest organic fragment) -> Uncharger ->
  canonical tautomer, then InChIKey. Three matching levels are reported:
  (1) exact canonical SMILES, (2) standardized InChIKey including stereo,
  (3) standardized InChIKey with stereochemistry removed (primary analysis,
      because Lee SMILES often omit stereo).

Reference electrodes: OxPot is reported vs Ag/AgCl (sat. KCl); Lee vs SCE.
E(vs Ag/AgCl sat. KCl) = E(vs SCE) + (0.241 - 0.197) V = E(vs SCE) + 0.044 V.
Linear recalibration absorbs any residual constant offset and scale.
"""
import ast
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OXPOT = os.path.join(REPO, "data", "processed", "oxpot_deduplicated_molecule_level.csv")
# Obtain grouped_dataset_acetonitrile_neutral.csv from the data release of Lee et al.,
# Mach. Learn.: Sci. Technol. 2024, 5, 015052 (not redistributed here).
LEE = os.path.join(REPO, "data", "raw", "grouped_dataset_acetonitrile_neutral.csv")
OUT = os.path.join(REPO, "results", "v2_resolution")
os.makedirs(OUT, exist_ok=True)

SCE_TO_AGCL = 0.241 - 0.197  # V

_cleanup = rdMolStandardize.CleanupParameters()
_frag = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_unch = rdMolStandardize.Uncharger()
_taut = rdMolStandardize.TautomerEnumerator()


def std_keys(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None, None
    try:
        m = rdMolStandardize.Cleanup(mol, _cleanup)
        m = _frag.choose(m)
        m = _unch.uncharge(m)
        m = _taut.Canonicalize(m)
        k_stereo = Chem.MolToInchiKey(m)
        m2 = Chem.Mol(m)
        Chem.RemoveStereochemistry(m2)
        k_nostereo = Chem.MolToInchiKey(m2)
        can = Chem.MolToSmiles(Chem.MolFromSmiles(str(smiles)))
        return can, k_stereo, k_nostereo
    except Exception:
        return None, None, None


def metrics(y_true, y_pred):
    e = y_pred - y_true
    return {"n": int(len(e)), "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "mean_signed": float(np.mean(e))}


def linfit(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def main():
    ox = pd.read_csv(OXPOT, encoding="utf-8-sig")
    lee = pd.read_csv(LEE)
    lee = lee.rename(columns={"Oxidation potential SCE cleaned": "eox_sce",
                              "Standard deviation": "sd_reports"})

    def n_reports(v):
        try:
            return len(ast.literal_eval(v)) if isinstance(v, str) else 1
        except Exception:
            return 1
    lee["n_reports"] = lee["Oxidation potential SCE"].apply(n_reports)

    for df, col in ((ox, "canonical_smiles"), (lee, "SMILES")):
        keys = df[col].apply(std_keys)
        df["can"] = [k[0] for k in keys]
        df["ik"] = [k[1] for k in keys]
        df["ik_ns"] = [k[2] for k in keys]

    res = {"inputs": {"oxpot_rows": int(len(ox)), "lee_rows": int(len(lee)),
                      "oxpot_std_fail": int(ox.ik.isna().sum()),
                      "lee_std_fail": int(lee.ik.isna().sum())},
           "reference_shift_V": SCE_TO_AGCL}

    # OxPot is deduplicated at molecule level; after standardization several
    # entries may collapse to one parent. Average them and record the count.
    def collapse(df, key):
        g = df.dropna(subset=[key]).groupby(key).agg(eox=("Eox", "mean"), n=("Eox", "size"),
                                                     spread=("Eox", lambda s: float(s.max() - s.min())))
        return g
    levels = {}
    for level, okey, lkey in (("exact_canonical_smiles", "can", "can"),
                              ("std_inchikey_stereo", "ik", "ik"),
                              ("std_inchikey_nostereo", "ik_ns", "ik_ns")):
        og = collapse(ox, okey)
        lg = lee.dropna(subset=[lkey]).groupby(lkey).agg(
            eox_sce=("eox_sce", "mean"), n_lee=("eox_sce", "size"),
            n_reports=("n_reports", "sum"), sd_reports=("sd_reports", "max"),
            smiles=("SMILES", "first"), name=("Compound name", "first"))
        m = lg.join(og, how="inner")
        levels[level] = {"n_matched": int(len(m)),
                         "oxpot_parents_collapsed": int((og.n > 1).sum()),
                         "matched_with_collapsed_oxpot": int((m.n > 1).sum())}
        if level == "std_inchikey_nostereo":
            primary = m.copy()
    res["matching"] = levels

    d = primary.reset_index().rename(columns={"index": "key"})
    y_exp = d["eox_sce"].to_numpy() + SCE_TO_AGCL      # experimental, vs Ag/AgCl
    y_lab = d["eox"].to_numpy()                        # OxPot label, vs Ag/AgCl
    r = float(np.corrcoef(y_exp, y_lab)[0, 1])

    # (a) reference conversion only
    raw = metrics(y_exp, y_lab)
    # (b) in-sample linear recalibration label -> experiment (optimistic)
    a, b = linfit(y_lab, y_exp)
    ins = metrics(y_exp, a * y_lab + b)
    # (c) leave-one-out recalibration (honest out-of-sample)
    loo_pred = np.empty_like(y_exp)
    for i in range(len(y_exp)):
        mask = np.arange(len(y_exp)) != i
        ai, bi = linfit(y_lab[mask], y_exp[mask])
        loo_pred[i] = ai * y_lab[i] + bi
    loo = metrics(y_exp, loo_pred)
    rng = np.random.default_rng(20260925)
    abs_err = np.abs(loo_pred - y_exp)
    boot = [np.mean(abs_err[rng.integers(0, len(abs_err), len(abs_err))]) for _ in range(20000)]
    loo["mae_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    # robustness: exclude matched pairs whose OxPot parent collapsed >1 entries
    keep = d["n"].to_numpy() == 1
    ya, yl = y_exp[keep], y_lab[keep]
    lp = np.empty_like(ya)
    for i in range(len(ya)):
        mask = np.arange(len(ya)) != i
        ai, bi = linfit(yl[mask], ya[mask])
        lp[i] = ai * yl[i] + bi
    loo_uncollapsed = metrics(ya, lp)

    res["agreement"] = {
        "n": int(len(d)), "pearson_r": r, "r2": r * r,
        "reference_converted_only": raw,
        "linear_recalibration_in_sample": {**ins, "slope": a, "intercept": b},
        "linear_recalibration_loo": loo,
        "loo_excluding_collapsed_oxpot": loo_uncollapsed,
    }

    # Experimental reproducibility inside Lee (all 592, and matched subset)
    multi = lee[lee.n_reports > 1]
    res["experimental_reproducibility"] = {
        "lee_molecules_with_multiple_reports": int(len(multi)),
        "fraction": float(len(multi) / len(lee)),
        "sd_mean": float(multi.sd_reports.mean()), "sd_median": float(multi.sd_reports.median()),
        "range_mean": float(multi["Range"].mean()) if "Range" in multi else None,
        "matched_subset_with_multiple_reports": int((d.n_reports > d.n_lee).sum()),
    }

    # Sharma et al. own calibration statistics (quoted from the source paper)
    res["source_paper_calibration"] = {"r2": 0.977, "rmse_V": 0.064, "slope": -0.661, "intercept": -2.773,
                                       "note": "in-sample fit on the source paper's calibration set, mostly aqueous"}

    d.assign(eox_exp_vs_agcl=y_exp, loo_recalibrated=loo_pred).to_csv(
        os.path.join(OUT, "label_fidelity_matched_pairs.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "label_fidelity_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
