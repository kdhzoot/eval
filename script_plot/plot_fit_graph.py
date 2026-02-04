#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot theoretical distribution shapes (from scipy.stats) and optionally compare them
with empirical data extracted from SSTable CSVs.

Key Features:
- Supports various distributions (e.g., weibull_min, lognorm, fisk, uniform).
- Extracts metric data (gap, kd, entry) for specific levels from SSTable CSV files.
- Supports spike/dense-range removal for tail analysis.
- Generates a PDF curve overlaid on an empirical histogram.

Usage:
    python plot_shape.py --dist fisk --params 1.5 0 100 --csv data.csv --target kd --lvl 3
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import sys
from typing import Optional

def get_real_data(csv_path: str, target: str, level: int) -> Optional[np.ndarray]:
    """Extract empirical data from SSTable CSV."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    if "level" not in df.columns:
        print(f"Column 'level' not found in {csv_path}")
        return None
    
    df = df[df["level"] == level]
    if len(df) == 0:
        print(f"No data found for level {level} in {csv_path}")
        return None

    if target == "gap":
        sub = df.copy()
        sub["min_key"] = pd.to_numeric(sub.get("min_key"), errors="coerce")
        sub["max_key"] = pd.to_numeric(sub.get("max_key"), errors="coerce")
        sub = sub.dropna(subset=["min_key", "max_key"])
        if len(sub) < 2: return None
        sub = sub.sort_values(["min_key", "max_key"], kind="mergesort")
        gaps = sub["min_key"].to_numpy()[1:] - sub["max_key"].to_numpy()[:-1]
        return gaps[gaps >= 0]
    
    elif target == "kd":
        min_k = pd.to_numeric(df["min_key"], errors="coerce")
        max_k = pd.to_numeric(df["max_key"], errors="coerce")
        en = pd.to_numeric(df["entry_n"], errors="coerce")
        valid = (en > 0) & (max_k >= min_k)
        kd = (max_k[valid] - min_k[valid] + 1) / en[valid]
        return kd.dropna().to_numpy()
    
    elif target == "entry":
        en = pd.to_numeric(df["entry_n"], errors="coerce")
        return en[en > 0].dropna().to_numpy()
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot a distribution shape and optionally compare with real data.")
    parser.add_argument("--dist", type=str, required=True, help="Distribution name (e.g., weibull_min, lognorm)")
    parser.add_argument("--params", type=float, nargs='+', required=True, help="Parameters (e.g., shape loc scale)")
    parser.add_argument("--range", type=float, nargs=2, help="Plot range [min, max]")
    parser.add_argument("--out", type=str, default="dist_compare.png", help="Output image path")
    
    # Optional arguments for real data comparison
    parser.add_argument("--csv", help="Original SSTable CSV path for empirical data")
    parser.add_argument("--target", choices=["gap", "kd", "entry"], default="kd", help="Data type to extract from CSV")
    parser.add_argument("--lvl", type=int, default=1, help="Level to extract from CSV")
    parser.add_argument("--remove-spike", action="store_true", help="Remove spike data (most dense range) before plotting")
    parser.add_argument("--window-width", type=float, default=1000.0, help="Window width for spike detection")
    
    args = parser.parse_args()
    
    try:
        dist = getattr(st, args.dist)
    except AttributeError:
        print(f"Error: Distribution '{args.dist}' not found in scipy.stats")
        return

    # 1. Fetch Real Data if CSV provided
    real_data = None
    if args.csv:
        real_data = get_real_data(args.csv, args.target, args.lvl)
        
        if real_data is not None and args.remove_spike:
            # Replicate Spike Removal logic from fit_distribution_entry.py
            x_all = real_data
            total_count = len(x_all)
            if total_count > 10:
                x_sorted = np.sort(x_all)
                max_count = 0
                best_lower = x_sorted[0]
                
                # Sliding Window scan
                right = 0
                for left in range(total_count):
                    while right < total_count and x_sorted[right] <= x_sorted[left] + args.window_width:
                        right += 1
                    if (right - left) > max_count:
                        max_count = right - left
                        best_lower = x_sorted[left]
                
                lower_bound = best_lower
                upper_bound = best_lower + args.window_width
                mask_spike = (x_all >= lower_bound) & (x_all <= upper_bound)
                real_data = x_all[~mask_spike]
                print(f"Removed spike range: [{lower_bound:.2f}, {upper_bound:.2f}]. Samples remaining: {len(real_data)}")

    # 2. Determine range
    if args.range:
        x_min, x_max = args.range
    elif real_data is not None:
        x_min, x_max = np.min(real_data), np.max(real_data)
        # Add 5% padding
        pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        x_min = max(0, x_min - pad)
        x_max = x_max + pad
    else:
        # Heuristic for default range based on loc and scale
        loc = args.params[-2] if len(args.params) >= 2 else 0
        scale = args.params[-1] if len(args.params) >= 1 else 1
        x_min = loc
        x_max = loc + 4 * scale
        
    x = np.linspace(x_min, x_max, 1000)
    
    try:
        y = dist.pdf(x, *args.params)
    except Exception as e:
        print(f"Error calculating PDF: {e}")
        return

    plt.figure(figsize=(10, 6))

    # 3. Plot Empirical Data
    if real_data is not None:
        label = f'Empirical Data ({args.target}, L{args.lvl})'
        if args.remove_spike:
            label += ' [Spike Removed]'
        plt.hist(real_data, bins=50, density=True, alpha=0.4, color='gray', 
                 edgecolor='white', label=label)
    
    # 4. Plot Theoretical PDF
    plt.plot(x, y, 'r-', lw=2.5, label=f"Fitted {args.dist}\nparams={args.params}")
    plt.fill_between(x, y, color='red', alpha=0.1)
    
    title = f"Distribution Comparison: {args.dist}"
    if args.csv:
        suffix = f" ({args.target.upper()} L{args.lvl})"
        if args.remove_spike:
            suffix = f" ({args.target.upper()} L{args.lvl}, Tail only)"
        title += f" vs Real Data{suffix}"
    
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(args.out)
    print(f"Graph saved to {args.out}")

if __name__ == "__main__":
    main()
