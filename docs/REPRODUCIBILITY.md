# Reproducibility

## Environment

Install the packages listed in `requirements.txt`. GINE training requires a compatible PyTorch and PyTorch Geometric installation.

## Run order

1. `python scripts/01_prepare_oxpot_deduplicated.py`
2. `python scripts/02_make_random_scaffold_splits.py`
3. `python scripts/03_run_tabular_models.py`
4. `python scripts/04_run_gine_model.py`
5. `python scripts/05_build_figures_and_tables.py`
6. `python scripts/06_descriptor_importance.py`

## CPU/GPU notes

Tabular models can run on CPU. GINE training benefits from GPU acceleration.

## Plotting

Figures can be reproduced directly from the included source tables without rerunning model training.
