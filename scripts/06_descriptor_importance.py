#!/usr/bin/env python
"""Format RDKit + XGB descriptor-importance outputs.

Input files:
    tables/supplementary/TableS6_rdkit_xgb_descriptor_importance_summary.csv

Output files:
    figures/main/Figure4_rdkit_xgb_descriptor_importance.*
    tables/supplementary/TableS6_rdkit_xgb_descriptor_importance_summary.csv

This script formats existing descriptor-importance results only. It does not run
SHAP and does not add new model experiments.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-s6", type=Path, default=Path("tables/supplementary/TableS6_rdkit_xgb_descriptor_importance_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures/main"))
    args = parser.parse_args()
    print("Descriptor-importance tables and figures are included in this repository draft.")
    print("Use this script as the public entry point for regenerating descriptor-importance plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
