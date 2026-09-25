# Changelog

## v1.1.0

- Added resolution and label-fidelity analyses used in the revised manuscript
  (no model was retrained; all analyses use the existing 560 run-level results):
  - `scripts/07_label_fidelity_audit.py`: matches OxPot molecules to independent
    experimental oxidation potentials (Lee et al., acetonitrile, vs SCE) on
    standardized parent structures and reports leave-one-out recalibrated agreement
    and the reproducibility of replicate literature reports.
  - `scripts/08_resolution_and_learning_curves.py`: seed-paired resolution of model
    rankings, statistical top groups, saturating power-law learning-curve fits with
    bootstrap confidence intervals, and crossover sizes.
  - `scripts/09_build_v2_figures.py`: Figures 4 and 5 of the revised manuscript.
- Added outputs in `results/v2_resolution/` and figures in `figures/main/`.
- Figure numbering in the revised manuscript: the former Figure 4 (RDKit + XGB
  descriptor importance) is Figure 6.
- Added `scipy` to `requirements.txt`.

## v1.0

- Initial clean repository draft for manuscript submission.
