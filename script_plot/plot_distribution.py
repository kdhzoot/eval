#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated plotting script for per-level SSTable distributions (Entry, Gap, KD).

Processes a single SSTable metadata CSV to generate subplots for each level, 
showing the distribution of the selected metric.

Input:
    - A single SSTable CSV file (e.g., from sstable2csv.py).

Metrics:
    - entry: Number of KV entries per SSTable.
    - gap: Non-overlap distance between consecutive SSTables in a level.
    - kd: Key Density = (max_key - min_key + 1) / entry_n.

Usage:
    python plot_distribution.py <run_csv> --metric {entry, gap, kd} [options]
"""

import argparse
import sys
import math
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply global styling
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 14,
})

def compute_level_gaps(level_df: pd.DataFrame) -> pd.Series:
    """Compute gaps between consecutive SSTables in a level."""
    sub = level_df.copy()
    sub["min_key"] = pd.to_numeric(sub.get("min_key"), errors="coerce")
    sub["max_key"] = pd.to_numeric(sub.get("max_key"), errors="coerce")
    sub = sub.dropna(subset=["min_key", "max_key"])

    if len(sub) < 2:
        return pd.Series([], dtype=float)

    sub = sub.sort_values(["min_key", "max_key"], kind="mergesort")
    prev_max = sub["max_key"].to_numpy()[:-1]
    next_min = sub["min_key"].to_numpy()[1:]

    gaps = next_min - prev_max
    gaps = gaps.clip(min=0)
    return pd.Series(gaps, dtype=float)

def compute_kd(df: pd.DataFrame) -> pd.Series:
    """Compute Key Density = (max_key - min_key + 1) / entry_n."""
    min_k = pd.to_numeric(df["min_key"], errors="coerce")
    max_k = pd.to_numeric(df["max_key"], errors="coerce")
    en = pd.to_numeric(df["entry_n"], errors="coerce")
    
    valid = (en > 0) & (max_k >= min_k)
    kd = pd.Series(np.nan, index=df.index)
    kd[valid] = (max_k[valid] - min_k[valid] + 1) / en[valid]
    return kd

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-level SSTable metrics distribution")
    p.add_argument("run_csv", help="Path to per-run SSTable CSV file")
    p.add_argument(
        "--metric", 
        choices=["entry", "gap", "kd"], 
        default="kd",
        help="Metric to plot (default: kd)"
    )
    p.add_argument("--bins", type=int, default=50, help="Number of bins for histogram (default: 50)")
    p.add_argument("--output", help="Output image path (default: <run_name>_<metric>.png)")
    p.add_argument("--xlog", action="store_true", help="Use log scale for X-axis")
    return p.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.run_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    run_name = csv_path.stem.replace("_sstables", "")
    
    # Pre-process levels
    df["level"] = pd.to_numeric(df.get("level"), errors="coerce")
    df = df.dropna(subset=["level"])
    df["level"] = df["level"].astype(int)
    levels = sorted(df["level"].unique())

    # Data extraction based on metric
    level_data_map = {}
    for lvl in levels:
        sub = df[df["level"] == lvl]
        if args.metric == "entry":
            data = pd.to_numeric(sub["entry_n"], errors="coerce").dropna()
            data = data[data > 0]
        elif args.metric == "gap":
            data = compute_level_gaps(sub)
        elif args.metric == "kd":
            data = compute_kd(sub).dropna()
        
        if not data.empty:
            level_data_map[lvl] = data

    if not level_data_map:
        print(f"No valid data found for metric '{args.metric}' in {csv_path}")
        return

    # Plotting setup
    n_levels = len(level_data_map)
    cols = min(3, n_levels)
    rows = math.ceil(n_levels / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows), squeeze=False)
    # Titles are removed by default as per user preference in earlier tasks
    # fig.suptitle(f"{args.metric.upper()} distribution - {run_name}", fontsize=28)

    axes_flat = axes.flatten()
    
    for i, (lvl, data) in enumerate(level_data_map.items()):
        ax = axes_flat[i]
        
        color = "steelblue"
        ax.hist(data, bins=args.bins, color=color, alpha=0.7, edgecolor='black')
        
        # Labeling
        xlabel = {
            "entry": "Num keys (entry_n)",
            "gap": "Gap size",
            "kd": "Key Density"
        }[args.metric]
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Level {lvl} (n={len(data)})", fontweight="bold")
        ax.grid(True, alpha=0.3)

        if args.xlog:
            ax.set_xscale("log")

        # Stats box
        stats_text = (
            f"Mean: {data.mean():,.1f}\n"
            f"Std: {data.std():,.1f}\n"
            f"Min: {data.min():,.1f}\n"
            f"Max: {data.max():,.1f}"
        )
        ax.text(
            0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    
    out_name = args.output if args.output else f"{run_name}_{args.metric}_distribution.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
