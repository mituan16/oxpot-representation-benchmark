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
