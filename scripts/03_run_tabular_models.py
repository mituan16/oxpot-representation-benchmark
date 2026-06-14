#!/usr/bin/env python
"""Run tabular OxPot baseline models.

Purpose:
    Train RDKit descriptor and Morgan fingerprint baselines with RF, XGB, and MLP.

Input files:
    data/processed/oxpot_deduplicated_molecule_level.csv
    data/splits/*.csv

Output files:
    results/raw/tabular_raw_results.csv
    results/summary/tabular_summary_by_condition.csv

This script trains tabular benchmark models.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/oxpot_deduplicated_molecule_level.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--out-raw", type=Path, default=Path("results/raw/tabular_raw_results.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("results/summary/tabular_summary_by_condition.csv"))
    args = parser.parse_args()
    raise SystemExit(
        "This clean repository includes final result tables. "
        "Use the archived training environment to rerun tabular models, or adapt this public entry point."
    )


if __name__ == "__main__":
    main()
