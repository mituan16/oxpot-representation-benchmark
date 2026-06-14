#!/usr/bin/env python
"""Prepare a deduplicated molecule-level OxPot table.

Purpose:
    Read the OxPot supporting-information spreadsheet, canonicalize SMILES with
    RDKit, remove invalid records, and create a molecule-level deduplicated table.

Input files:
    data/raw/ci5c00159_si_002.xlsx

Output files:
    data/processed/oxpot_deduplicated_molecule_level.csv

This script prepares data only; it does not train models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def canonicalize(smiles: str) -> str | None:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def murcko_scaffold(canonical_smiles: str) -> str:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        return f"INVALID::{canonical_smiles}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else f"NO_SCAFFOLD::{canonical_smiles}"


def build_table(raw_excel: Path, output_csv: Path, sheet: str = "oxpot") -> pd.DataFrame:
    raw = pd.read_excel(raw_excel, sheet_name=sheet)
    required = {"SMILES", "Eox"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw[["SMILES", "Eox"]].copy()
    df["Eox"] = pd.to_numeric(df["Eox"], errors="coerce")
    df["canonical_smiles"] = df["SMILES"].map(canonicalize)
    df = df[df["canonical_smiles"].notna() & np.isfinite(df["Eox"])].copy()
    df["Eox_key"] = df["Eox"].round(10)

    conflicts = df.groupby("canonical_smiles")["Eox_key"].nunique()
    conflict_smiles = set(conflicts[conflicts > 1].index)
    df = df[~df["canonical_smiles"].isin(conflict_smiles)].copy()
    df = df.sort_values(["canonical_smiles", "Eox_key"]).drop_duplicates(["canonical_smiles", "Eox_key"])
    df = df.reset_index().rename(columns={"index": "original_row_index", "SMILES": "original_smiles"})
    df.insert(0, "mol_id", range(len(df)))
    df["murcko_scaffold"] = df["canonical_smiles"].map(murcko_scaffold)
    df["molecular_target_key"] = df["canonical_smiles"].astype(str) + "||" + df["Eox"].round(10).astype(str)
    out = df[["mol_id", "original_row_index", "original_smiles", "canonical_smiles", "Eox", "murcko_scaffold", "molecular_target_key"]]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-excel", type=Path, default=Path("data/raw/ci5c00159_si_002.xlsx"))
    parser.add_argument("--sheet", default="oxpot")
    parser.add_argument("--output", type=Path, default=Path("data/processed/oxpot_deduplicated_molecule_level.csv"))
    args = parser.parse_args()
    table = build_table(args.raw_excel, args.output, args.sheet)
    print(f"Wrote {args.output} with {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
