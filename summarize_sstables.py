#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize per-run SSTable entry CSVs into a single wide CSV.

Input:
    - A directory containing per-run CSVs with columns:
            level,id,size,entry_n,min_seq,max_seq,min_key,max_key

Output:
    - A single CSV (default: derived from input dir name) with one row per input file.
    Columns include:
      run_id, n_levels,
      L{lvl}_count, L{lvl}_min_key, L{lvl}_max_key,
                        L{lvl}_kd_mean, L{lvl}_kd_std, L{lvl}_kd_min, L{lvl}_kd_max, L{lvl}_kd_q1, L{lvl}_kd_q2, L{lvl}_kd_q3,
                        L{lvl}_gap_mean, L{lvl}_gap_std, L{lvl}_gap_min, L{lvl}_gap_max, L{lvl}_gap_q1, L{lvl}_gap_q2, L{lvl}_gap_q3
    where kd = (max_key - min_key) / entry_n (only if entry_n>0 and keys valid)
    and gaps are non-overlap distances between consecutive SST ranges within
    the same level: gap_i = max(0, next.min_key - prev.max_key).

Usage:
    python summarize_sstables.py <sst_dir> [--pattern "*.csv"] [--out OUT.csv]
"""

import argparse
import os
import glob
import re
from typing import Dict, List, Any, Tuple, Set

import numpy as np
import pandas as pd
import scipy.stats as st


def q1_q2_q3(values: np.ndarray) -> Tuple[float, float, float]:
    """Return (Q1, Q2/median, Q3) for a numeric array; NaN if empty."""
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    q = np.nanquantile(values.astype(float), [0.25, 0.50, 0.75])
    return (float(q[0]), float(q[1]), float(q[2]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize SSTable entry CSVs into a single wide CSV.")
    p.add_argument("sst_dir", help="Directory containing per-run CSV files.")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern for entry CSVs.")
    p.add_argument(
        "--out",
        default=None,
        help="Output summary CSV path (default: <sst_dir_basename without _csv>_summary.csv next to sst_dir)",
    )
    p.add_argument(
        "--group-by-size",
        action="store_true",
        help="Group results by DB size (extracted from filename) and save into separate CSVs in a folder.",
    )
    return p.parse_args()


def default_out_path(sst_dir: str) -> str:
    sst_dir_abs = os.path.abspath(sst_dir)
    base = os.path.basename(sst_dir_abs.rstrip(os.sep))
    if base.endswith("_csv"):
        base = base[: -len("_csv")]
    parent = os.path.dirname(sst_dir_abs)
    return os.path.join(parent, f"{base}_summary.csv")


def infer_run_id(csv_path: str) -> str:
    """Derive a run identifier from the filename (strip common suffix)."""
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    if name.endswith("_entries"):
        name = name[: -len("_entries")]
    return name


def extract_db_size(run_id: str) -> str:
    """Extract DB size (e.g., 100GB, 1TB) from run_id using regex."""
    m = re.search(r"(\d+(?:[TGMK]B|B))", run_id, re.IGNORECASE)
    return m.group(1).upper() if m else "unknown"


def compute_key_density(df: pd.DataFrame) -> pd.Series:
    """Compute per-row key density = (max_key - min_key + 1) / entry_n when valid, else NaN."""
    min_k = pd.to_numeric(df["min_key"], errors="coerce")
    max_k = pd.to_numeric(df["max_key"], errors="coerce")
    en = pd.to_numeric(df["entry_n"], errors="coerce")
    valid = (~min_k.isna()) & (~max_k.isna()) & (~en.isna()) & (en > 0) & (max_k >= min_k)
    kd = pd.Series(np.nan, index=df.index, dtype=float)
    kd[valid] = (max_k[valid] - min_k[valid] + 1) / en[valid]
    return kd


def level_gaps(level_df: pd.DataFrame) -> np.ndarray:
    """
    Compute non-overlap gaps between consecutive SST ranges within a level.
    Sort by min_key ascending; gap_i = max(0, next.min_key - prev.max_key).
    Returns an array of gaps (can be empty).
    """
    sub = level_df.copy()
    sub["min_key"] = pd.to_numeric(sub["min_key"], errors="coerce")
    sub["max_key"] = pd.to_numeric(sub["max_key"], errors="coerce")
    sub = sub.dropna(subset=["min_key", "max_key"])
    if sub.empty:
        return np.array([], dtype=float)
    sub = sub.sort_values(["min_key", "max_key"], kind="mergesort")
    mins = sub["min_key"].to_numpy()
    maxs = sub["max_key"].to_numpy()
    if len(mins) < 2:
        return np.array([], dtype=float)
    prev_max = maxs[:-1]
    next_min = mins[1:]
    gaps = np.maximum(0.0, next_min - prev_max)
    return gaps.astype(float)


def fit_pareto_params(gaps: np.ndarray) -> Tuple[float, float, float]:
    """Fit Pareto distribution to gaps; return (shape, loc, scale). Returns (NaN, NaN, NaN) if failed."""
    # Ensure some positive values for pareto fit and filter non-positive for safety
    x_safe = gaps[gaps > 0]
    if x_safe.size < 20:  # Require minimum samples for a sensible fit
        return (np.nan, np.nan, np.nan)
    try:
        # floc=0 anchors the distribution at zero
        params = st.pareto.fit(x_safe, floc=0)
        return (float(params[0]), float(params[1]), float(params[2]))
    except:
        return (np.nan, np.nan, np.nan)


def fit_fisk_params(values: np.ndarray) -> Tuple[float, float, float]:
    """Fit Fisk (log-logistic) distribution to values; return (c, loc, scale). Returns (NaN, NaN, NaN) if failed."""
    # Fisk requires positive values
    x_safe = values[values > 0]
    if x_safe.size < 20:
        return (np.nan, np.nan, np.nan)
    try:
        # floc=0 anchors the distribution at zero
        params = st.fisk.fit(x_safe, floc=0)
        return (float(params[0]), float(params[1]), float(params[2]))
    except:
        return (np.nan, np.nan, np.nan)


def fit_uniform_params(values: np.ndarray) -> Tuple[float, float, float]:
    """Fit Uniform distribution; return (NaN, loc, scale). Shape is NaN."""
    x_safe = values[~np.isnan(values)]
    if x_safe.size < 1:
        return (np.nan, np.nan, np.nan)
    try:
        loc, scale = st.uniform.fit(x_safe)
        return (np.nan, float(loc), float(scale))
    except:
        return (np.nan, np.nan, np.nan)


def fit_lognormal_params(values: np.ndarray) -> Tuple[float, float, float]:
    """Fit Lognormal distribution; return (s, loc, scale)."""
    x_safe = values[values > 0]
    if x_safe.size < 20:
        return (np.nan, np.nan, np.nan)
    try:
        # floc=0 anchors distribution at zero
        params = st.lognorm.fit(x_safe, floc=0)
        return (float(params[0]), float(params[1]), float(params[2]))
    except:
        return (np.nan, np.nan, np.nan)


def summarize_one(csv_path: str) -> Tuple[Dict[str, Any], Set[int]]:
    """
    Summarize a single *_entries.csv into per-level aggregates.
    Returns:
      - flat dict with scalar fields ('run_id', 'n_levels', and L{lvl}_* fields)
      - set of levels present in this file
    """
    df = pd.read_csv(csv_path)
    required = {"level", "entry_n", "min_key", "max_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    # Normalize dtypes
    df["level"] = pd.to_numeric(df["level"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["level"])
    df["level"] = df["level"].astype(int)

    # Compute key density per row
    df["key_density"] = compute_key_density(df)

    # Per-level aggregates
    res: Dict[str, Any] = {"run_id": infer_run_id(csv_path)}
    levels_present = set(df["level"].unique().tolist())
    res["n_levels"] = len(levels_present)

    # Determine max level for fitting logic
    max_level = max(levels_present) if levels_present else 0

    for lvl in sorted(levels_present):
        sub = df[df["level"] == lvl]
        # Count of SST files (rows) in this level
        count = int(len(sub))

        # min/max key across this level
        level_min_key = pd.to_numeric(sub["min_key"], errors="coerce").min()
        level_max_key = pd.to_numeric(sub["max_key"], errors="coerce").max()

        # key density stats (population std: ddof=0)
        kd = pd.to_numeric(sub["key_density"], errors="coerce")
        kd_mean = float(np.nanmean(kd)) if kd.notna().any() else np.nan
        kd_std = float(np.nanstd(kd)) if kd.notna().any() else np.nan
        kd_vals = kd.dropna().to_numpy(dtype=float)
        kd_min = float(np.nanmin(kd_vals)) if kd_vals.size > 0 else np.nan
        kd_max = float(np.nanmax(kd_vals)) if kd_vals.size > 0 else np.nan
        kd_q1, kd_q2, kd_q3 = q1_q2_q3(kd_vals)

        # Distribution fitting logic
        kd_model = "none"
        kd_shape, kd_loc, kd_scale = np.nan, np.nan, np.nan

        if lvl == 1:
            kd_model = "none"
        elif lvl == 2:
            kd_model = "uniform"
            # Uniform: fit returns (loc, scale) -> shape=NaN
            _, kd_loc, kd_scale = fit_uniform_params(kd_vals)
        elif lvl == max_level:
            # Lognormal for last level
            kd_model = "lognormal"
            kd_shape, kd_loc, kd_scale = fit_lognormal_params(kd_vals)
        elif 2 < lvl < max_level:
            # Fisk for intermediate levels
            kd_model = "fisk"
            kd_shape, kd_loc, kd_scale = fit_fisk_params(kd_vals)
        else:
            kd_model = "none"
        
        if kd_model != "none":
            print(f"Fit L{lvl} kd ({kd_model}): shape={kd_shape}, loc={kd_loc}, scale={kd_scale}")

        # gaps between consecutive SST ranges
        gaps = level_gaps(sub)
        gap_mean = float(np.nanmean(gaps)) if gaps.size > 0 else np.nan
        gap_std = float(np.nanstd(gaps)) if gaps.size > 0 else np.nan
        gap_min = float(np.nanmin(gaps)) if gaps.size > 0 else np.nan
        gap_max = float(np.nanmax(gaps)) if gaps.size > 0 else np.nan
        gap_q1, gap_q2, gap_q3 = q1_q2_q3(gaps)

        # Pareto fitting
        gap_p_shape, gap_p_loc, gap_p_scale = fit_pareto_params(gaps)

        # Fill into result
        prefix = f"L{lvl}"
        res[f"{prefix}_count"] = count
        res[f"{prefix}_min_key"] = level_min_key if pd.notna(level_min_key) else np.nan
        res[f"{prefix}_max_key"] = level_max_key if pd.notna(level_max_key) else np.nan
        res[f"{prefix}_kd_mean"] = kd_mean
        res[f"{prefix}_kd_std"] = kd_std
        res[f"{prefix}_kd_min"] = kd_min
        res[f"{prefix}_kd_max"] = kd_max
        res[f"{prefix}_kd_q1"] = kd_q1
        res[f"{prefix}_kd_q2"] = kd_q2
        res[f"{prefix}_kd_q3"] = kd_q3
        res[f"{prefix}_kd_model"] = kd_model
        res[f"{prefix}_kd_shape"] = kd_shape
        res[f"{prefix}_kd_loc"] = kd_loc
        res[f"{prefix}_kd_scale"] = kd_scale
        res[f"{prefix}_gap_mean"] = gap_mean
        res[f"{prefix}_gap_std"] = gap_std
        res[f"{prefix}_gap_min"] = gap_min
        res[f"{prefix}_gap_max"] = gap_max
        res[f"{prefix}_gap_q1"] = gap_q1
        res[f"{prefix}_gap_q2"] = gap_q2
        res[f"{prefix}_gap_q3"] = gap_q3
        res[f"{prefix}_gap_pareto_shape"] = gap_p_shape
        res[f"{prefix}_gap_pareto_scale"] = gap_p_scale

    return res, levels_present


def main() -> None:
    args = parse_args()
    in_dir = os.path.abspath(args.sst_dir)
    pattern = os.path.join(in_dir, args.pattern)
    paths = sorted(glob.glob(pattern))

    out_path = args.out or default_out_path(in_dir)
    if not paths:
        print(f"[WARN] No files matched: {pattern}")
        # still write an empty CSV with just run_id,n_levels?
        pd.DataFrame(columns=["run_id", "n_levels"]).to_csv(out_path, index=False)
        return

    summaries: List[Dict[str, Any]] = []
    all_levels: Set[int] = set()

    level_metrics = [
        "count",
        "min_key",
        "max_key",
        "kd_mean",
        "kd_std",
        "kd_min",
        "kd_max",
        "kd_q1",
        "kd_q2",
        "kd_q3",
        "kd_model",
        "kd_shape",
        "kd_loc",
        "kd_scale",
        "gap_mean",
        "gap_std",
        "gap_min",
        "gap_max",
        "gap_q1",
        "gap_q2",
        "gap_q3",
        "gap_pareto_shape",
        "gap_pareto_scale",
    ]

    for p in paths:
        try:
            row, lvls = summarize_one(p)
            row["db_size"] = extract_db_size(row["run_id"])
            summaries.append(row)
            all_levels |= lvls
            print(f"[OK] {os.path.basename(p)} -> summarized (levels: {sorted(lvls)})")
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    # Build a consistent column order
    level_cols: List[str] = []
    for lvl in sorted(all_levels):
        prefix = f"L{lvl}"
        level_cols.extend([f"{prefix}_{m}" for m in level_metrics])

    base_cols = ["run_id", "n_levels", "db_size"]
    cols = base_cols + level_cols

    # Assemble DataFrame, fill missing level columns as NaN
    df_all = pd.DataFrame(summaries)
    for c in cols:
        if c not in df_all.columns:
            df_all[c] = np.nan
    df_all = df_all[cols]

    if args.group_by_size:
        # Create output directory
        out_dir = args.out if args.out else in_dir + "_summaries"
        os.makedirs(out_dir, exist_ok=True)
        
        # Group by db_size and write separate CSVs
        for size, group_df in df_all.groupby("db_size"):
            size_out = os.path.join(out_dir, f"sstables_{size}_summary.csv")
            group_df.to_csv(size_out, index=False)
            print(f"[DONE] Wrote size-grouped summary: {size_out} (rows={len(group_df)})")
    else:
        df_all.to_csv(out_path, index=False)
        print(f"[DONE] Wrote summary CSV: {out_path} (rows={len(df_all)})")


if __name__ == "__main__":
    main()
