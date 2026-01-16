#!/usr/bin/env python3
"""
Plot overlapping key density histograms for multiple SSTable data runs.

Each level gets its own subplot showing the distribution of key density
for all SSTables in that level, comparing different runs.

Usage:
  python3 plot_kd_overlap.py <csv_file1> <csv_file2> ...
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

def main():
    # If no arguments, try to find CSVs in current directory or suggest usage
    if len(sys.argv) < 2:
        print("Usage: python3 plot_kd_overlap.py <csv_file1> <csv_file2> ...")
        print("Example: python3 plot_kd_overlap.py ../sstables_csvs/*.csv")
        sys.exit(1)
    
    csv_files = sys.argv[1:]
    run_data = {}
    all_levels = set()

    # Load data for each CSV
    for csv_file in csv_files:
        path = Path(csv_file)
        if not path.exists():
            print(f"Warning: {csv_file} not found. Skipping.")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            
            # Compute key density
            # Ensure max_key and min_key are treated as numeric
            df['kd'] = pd.to_numeric(df['max_key']) - pd.to_numeric(df['min_key'])
            
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
    cols = min(3, n_levels)
    rows = math.ceil(n_levels / cols)
    
    # Increase height if more rows
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows), squeeze=False)
    fig.suptitle('Key Density Distribution Comparison by Level', fontsize=16, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten()
    
    # Use tab10 colormap for distinct colors
    colors = plt.cm.tab10.colors 
    
    for idx, lvl in enumerate(levels):
        ax = axes_flat[idx]
        
        # Collect data for this level to determine common range
        lvl_plot_data = []
        run_names_present = []
        
        for run_name, df in run_data.items():
            kd_data = df[df['level'] == lvl]['kd']
            if kd_data.empty:
                continue
            lvl_plot_data.append(kd_data)
            run_names_present.append(run_name)
            
        if not lvl_plot_data:
            continue
            
        # Global range for this subplot
        all_vals = pd.concat(lvl_plot_data)
        g_min, g_max = all_vals.min(), all_vals.max()
        
        for color_idx, (kd_data, run_name) in enumerate(zip(lvl_plot_data, run_names_present)):
            color = colors[color_idx % len(colors)]
            
            # Plot histogram with shared range ensuring identical binning width
            ax.hist(kd_data, bins=50, range=(g_min, g_max), edgecolor='black', alpha=0.4, 
                    color=color, label=run_name)
            
            print(f"L{lvl} [{run_name}]: count={len(kd_data):4d}, mean={kd_data.mean():>15,.0f}, std={kd_data.std():>15,.0f}")

        ax.set_xlabel('Key Density (max_key - min_key)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Level {lvl}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    
    # Hide empty subplots
    for idx in range(n_levels, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    output_file = "kd_distribution_overlap.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved overlap plot: {output_file}")

if __name__ == "__main__":
    main()
