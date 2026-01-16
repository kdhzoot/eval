#!/usr/bin/env python3
"""
Plot overlapping gap histograms for multiple SSTable data runs.

Each level gets its own subplot showing the distribution of gaps between SSTables.

Usage:
  python3 plot_gap_overlap.py <csv_file1> <csv_file2> ... [--xlog]
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
import numpy as np

def compute_level_gaps(level_df: pd.DataFrame) -> pd.Series:
    """Compute non-overlap gaps between consecutive SST ranges within a level."""
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

def log10p1(values: pd.Series) -> pd.Series:
    """Compute log10(values + 1) safely for gap visualization."""
    return (values + 1.0).apply(np.log10)

def auto_trim_upper_iqr(values: pd.Series, iqr_mult: float) -> pd.Series:
    """Auto-trim using an upper Tukey fence (Q3 + k*IQR)."""
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(v) < 4:
        return v
    q1 = float(v.quantile(0.25))
    q3 = float(v.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return v
    upper = q3 + iqr_mult * iqr
    return v[v <= upper]

def main():
    parser = argparse.ArgumentParser(description="Plot overlapping gap histograms from multiple runs.")
    parser.add_argument("csv_files", nargs="+", help="SSTable CSV files")
    parser.add_argument("--xlog", action="store_true", help="Plot x-axis as log10(gap+1)")
    parser.add_argument("--auto-trim", action="store_true", help="Automatically detect and trim upper-tail outliers")
    parser.add_argument("--auto-iqr-mult", type=float, default=1.5, help="IQR multiplier for the upper fence (default: 1.5)")
    args = parser.parse_args()
    
    run_data = {}
    all_levels = set()

    # Load data for each CSV
    for csv_file in args.csv_files:
        path = Path(csv_file)
        if not path.exists():
            print(f"Warning: {csv_file} not found. Skipping.")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            df['level'] = pd.to_numeric(df['level'], errors='coerce')
            df = df.dropna(subset=['level'])
            
            # Shorten run name, e.g., fillrandom_1TB_001
            parts = path.stem.split('_')
            if len(parts) >= 6:
                run_name = f"{parts[2]}_{parts[4]}_{parts[5]}"
            else:
                run_name = path.stem.replace('_sstables', '')
            
            run_data[run_name] = df
            all_levels.update(df['level'].unique())
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not run_data:
        print("No valid data found.")
        sys.exit(1)

    levels = sorted(list(all_levels))
    n_levels = len(levels)
    cols = min(3, n_levels) if n_levels > 0 else 1
    rows = math.ceil(n_levels / cols) if n_levels > 0 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows), squeeze=False)
    title = 'Gap Distribution Comparison by Level'
    if args.xlog:
        title += ' (Log Scale)'
    if args.auto_trim:
        title += f' [Auto-Trim k={args.auto_iqr_mult}]'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()
    colors = plt.cm.tab10.colors 
    
    for idx, lvl in enumerate(levels):
        ax = axes_flat[idx]
        
        # Collect all data for this level first to determine common range
        lvl_plot_data = []
        run_names_present = []
        
        for run_name, df in run_data.items():
            lvl_df = df[df['level'] == lvl]
            gaps = compute_level_gaps(lvl_df)
            if gaps.empty:
                continue
                
            if args.auto_trim:
                gaps = auto_trim_upper_iqr(gaps, args.auto_iqr_mult)
            
            p_data = log10p1(gaps) if args.xlog else gaps
            lvl_plot_data.append(p_data)
            run_names_present.append(run_name)
            
        if not lvl_plot_data:
            continue
            
        # Determine global range for this level
        all_vals = pd.concat(lvl_plot_data)
        g_min, g_max = all_vals.min(), all_vals.max()
        
        for color_idx, (p_data, run_name) in enumerate(zip(lvl_plot_data, run_names_present)):
            color = colors[color_idx % len(colors)]
            
            # Plot histogram with common range and bins
            ax.hist(p_data, bins=50, range=(g_min, g_max), edgecolor='black', alpha=0.4, 
                    color=color, label=run_name)
            
            print(f"L{lvl} [{run_name}]: gaps_count={len(p_data):4d}, mean={p_data.mean():>15,.1f}")

        ax.set_xlabel('log10(Gap + 1)' if args.xlog else 'Gap', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Level {lvl}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    # Hide empty subplots
    for idx in range(n_levels, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    suffix = "_log" if args.xlog else ""
    output_file = f"gap_distribution_overlap{suffix}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved overlap plot: {output_file}")

if __name__ == "__main__":
    main()
