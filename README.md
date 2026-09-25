# Benchmarking RDKit descriptors, Morgan fingerprints, and GINE for oxidation potential prediction

This repository contains code, split files, source data tables, and figures for a deduplicated molecule-level OxPot benchmark comparing RDKit descriptors, Morgan fingerprints, and 2D graph GINE models for oxidation-potential prediction.

## Dataset summary

- OxPot oxidation potential dataset
- 15,238 deduplicated molecule-level entries
- SMILES input
- Target: Eox as reported in the source dataset
- Random and Bemis-Murcko scaffold splits
- Seeds: 11, 22, 33, 44, 55
- Train sizes: 500, 1000, 2000, 3000, 5000, 7000, 8000, 10000

## Model families

- RDKit + RF/XGB/MLP
- Morgan + RF/XGB/MLP
- 2D graph + GINE

## Main results summary

- 560 successful raw results
- 480 tabular results
- 80 GINE results
- RDKit descriptors outperform Morgan fingerprints in matched comparisons
- RDKit + MLP is strongest in low-data regimes
- GINE becomes competitive at larger training sizes, especially under scaffold splitting

## How to reproduce

1. Create an environment with the packages in `requirements.txt`.
2. Obtain the raw OxPot spreadsheet from the original publication if raw-data preparation is needed.
3. Place the raw spreadsheet at `data/raw/ci5c00159_si_002.xlsx`.
4. Run scripts `01`-`06`, or use the provided processed data, split files, final result tables, and source data tables.

## Data note

Do not redistribute publisher-provided raw supplementary Excel files unless the license permits. This repository provides processed benchmark files and scripts for reproducibility.

## Citation

Manuscript under review; citation to be added.

## Resolution and label-fidelity analyses (v1.1.0)

These analyses use only the existing run-level results (`results/raw/TableS1_full_raw_results.csv`)
and CPU computation.

1. Place `grouped_dataset_acetonitrile_neutral.csv` from the data release of Lee et al.
   (Mach. Learn.: Sci. Technol. 2024, 5, 015052) in `data/raw/` (not redistributed here).
2. Run, from the repository root:

```
python scripts/07_label_fidelity_audit.py
python scripts/08_resolution_and_learning_curves.py
python scripts/09_build_v2_figures.py
```

Outputs are written to `results/v2_resolution/` and `figures/main/`. A difference between two
methods is called resolved when all five seeds agree in sign and a two-sided paired t-test gives
p < 0.05; see the manuscript for details.
