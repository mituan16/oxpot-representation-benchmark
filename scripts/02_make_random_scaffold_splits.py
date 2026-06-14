#!/usr/bin/env python
"""Create random and Bemis-Murcko scaffold splits for the OxPot benchmark.

Input files:
    data/processed/oxpot_deduplicated_molecule_level.csv

Output files:
    data/splits/OxPot_dedup_{split}_seed{seed}_train{train_size}_indices.csv

This script creates split indices only; it does not train models.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_SIZES = [500, 1000, 2000, 3000, 5000, 7000, 8000, 10000]
SEEDS = [11, 22, 33, 44, 55]
VAL_FRAC = 0.10
TEST_FRAC = 0.20


def make_split_df(train_ids, val_ids, test_ids, split, seed, train_size):
    rows = [(int(i), "train") for i in train_ids]
    rows += [(int(i), "val") for i in val_ids]
    rows += [(int(i), "test") for i in test_ids]
    df = pd.DataFrame(rows, columns=["mol_id", "split_role"])
    df.insert(0, "split", split)
    df.insert(1, "seed", seed)
    df.insert(2, "target_train_size", train_size)
    return df


def random_splits(data, out_dir):
    n_test = int(math.floor(len(data) * TEST_FRAC))
    n_val = int(math.floor(len(data) * VAL_FRAC))
    all_ids = data["mol_id"].to_numpy(dtype=int)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_ids)
        test_ids = shuffled[:n_test]
        val_ids = shuffled[n_test:n_test + n_val]
        train_pool = shuffled[n_test + n_val:]
        for size in TRAIN_SIZES:
            df = make_split_df(train_pool[:size], val_ids, test_ids, "random", seed, size)
            df.to_csv(out_dir / f"OxPot_dedup_random_seed{seed}_train{size}_indices.csv", index=False, encoding="utf-8-sig")


def scaffold_splits(data, out_dir):
    group_rows = []
    for scaffold, sub in data.groupby("murcko_scaffold"):
        group_rows.append({"murcko_scaffold": scaffold, "size": len(sub), "mol_ids": list(sub["mol_id"].astype(int))})
    groups = pd.DataFrame(group_rows).sort_values(["size", "murcko_scaffold"], ascending=[False, True]).reset_index(drop=True)
    max_train = max(TRAIN_SIZES)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        g = groups.copy()
        g["tie_noise"] = rng.random(len(g))
        g = g.sort_values(["size", "tie_noise"], ascending=[False, True]).reset_index(drop=True)
        train_groups, remaining = [], []
        count = 0
        for _, row in g.iterrows():
            if count < max_train:
                train_groups.append(row)
                count += int(row["size"])
            else:
                remaining.append(row)
        val_target = int(round(sum(int(row["size"]) for row in remaining) / 3.0))
        remaining = [remaining[int(i)] for i in rng.permutation(len(remaining))]
        val_groups, test_groups, val_count = [], [], 0
        for row in remaining:
            if val_count < val_target:
                val_groups.append(row)
                val_count += int(row["size"])
            else:
                test_groups.append(row)

        def flatten(rows):
            ids = []
            for row in rows:
                ids.extend(int(x) for x in row["mol_ids"])
            return np.array(ids, dtype=int)

        train_pool = rng.permutation(flatten(train_groups))
        val_ids = rng.permutation(flatten(val_groups))
        test_ids = rng.permutation(flatten(test_groups))
        for size in TRAIN_SIZES:
            df = make_split_df(train_pool[:size], val_ids, test_ids, "scaffold", seed, size)
            df.to_csv(out_dir / f"OxPot_dedup_scaffold_seed{seed}_train{size}_indices.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/processed/oxpot_deduplicated_molecule_level.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    random_splits(data, args.out_dir)
    scaffold_splits(data, args.out_dir)
    print(f"Wrote split files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
