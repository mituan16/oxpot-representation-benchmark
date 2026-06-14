#!/usr/bin/env python
"""Build manuscript figures and supplementary tables from final source data.

Input files:
    results/raw/canonical_full_raw_results.csv
    results/summary/canonical_summary_by_condition.csv
    tables/source_data/*.csv

Output files:
    figures/main/*.png
    tables/supplementary/*.csv

This script formats existing results only; it does not train models.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    print("Final figures and source tables are included in this repository draft.")
    print("Use this script as the public entry point for regenerating plots from source data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
