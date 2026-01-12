#!/usr/bin/env python3
"""
Plot key density histogram for each level in SSTable data from a single run.

Each level gets its own subplot showing the distribution of key density
for all SSTables in that level.

Usage:
  python3 kd_distribution.py [run_csv]
  
Example:
  python3 kd_distribution.py ../sstables_csvs/sstables_001_20251124_124704_sstables.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

def main():
    # Default to first available run CSV, or use command-line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        import glob
        csvs = sorted(glob.glob("../sstables_csvs/*_sstables.csv"))
        if not csvs:
            print("Error: No CSV files found in ../sstables_csvs/")
            sys.exit(1)
        csv_file = csvs[0]
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Compute key density for each SSTable
    df['kd'] = df['max_key'].astype(int) - df['min_key'].astype(int)
    
    # Get run name from filename
    run_name = Path(csv_file).stem.replace('_sstables', '')
    
    # Get sorted levels
    levels = sorted(df['level'].unique())
    
    print(f"Analyzing: {run_name}")
    print(f"Total SSTable: {len(df)}")
    print(f"Levels: {levels}")
    print()
    
    # Calculate grid dimensions (arrange subplots in 2 rows if possible)
    n_levels = len(levels)
    cols = min(3, n_levels)
    rows = math.ceil(n_levels / cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    fig.suptitle(f'Key Density Distribution by Level - {run_name}', fontsize=14, fontweight='bold')
    
    # Flatten axes for easier iteration
    if n_levels == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if rows * cols > 1 else [axes]
    
    # Plot histogram for each level
    for idx, lvl in enumerate(levels):
        ax = axes_flat[idx]
        lvl_data = df[df['level'] == lvl]['kd']
        
        ax.hist(lvl_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Key Density', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Level {lvl} (n={len(lvl_data)})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {lvl_data.mean():,.0f}\nStd: {lvl_data.std():,.0f}\nMin: {lvl_data.min():,.0f}\nMax: {lvl_data.max():,.0f}"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"L{lvl}: count={len(lvl_data):4d}, mean={lvl_data.mean():>12,.0f}, std={lvl_data.std():>12,.0f}, min={lvl_data.min():>12,.0f}, max={lvl_data.max():>12,.0f}")
    
    # Hide empty subplots
    for idx in range(n_levels, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Save figure
    output_file = f"{run_name}_kd_distribution.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()
