#!/usr/bin/env python3
"""
Integrated plot script for SSTable distributions (KD, Entry, Gaps, Ranges).
Combines functionality from plot_kd_overlap.py, plot_entry_overlap.py, and plot_gap_overlap.py.
"""

import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import math
import numpy as np

def shorten_run_name(path: Path) -> str:
    """Shorten run name for legend, e.g., fillrandom_1TB_001"""
    parts = path.stem.split('_')
    if len(parts) >= 6:
        return f"{parts[2]}_{parts[4]}_{parts[5]}"
    else:
        return path.stem.replace('_sstables', '')

def compute_level_gaps(level_df: pd.DataFrame) -> pd.Series:
    """Compute non-overlap gaps between consecutive SST ranges within a level."""
    sub = level_df.copy()
    sub["min_key"] = pd.to_numeric(sub.get("min_key"), errors="coerce")
    sub["max_key"] = pd.to_numeric(sub.get("max_key"), errors="coerce")
    sub = sub.dropna(subset=["min_key", "max_key"])

    if len(sub) < 2:
        return pd.Series([], dtype=float)

    # Sort to find neighbors
    sub = sub.sort_values(["min_key", "max_key"], kind="mergesort")
    prev_max = sub["max_key"].to_numpy()[:-1]
    next_min = sub["min_key"].to_numpy()[1:]

    gaps = next_min - prev_max
    gaps = gaps.clip(min=0)
    return pd.Series(gaps, dtype=float)

def compute_level_ranges(level_df: pd.DataFrame) -> pd.Series:
    """Compute SST key range per file: max_key - min_key."""
    sub = level_df.copy()
    sub["min_key"] = pd.to_numeric(sub.get("min_key"), errors="coerce")
    sub["max_key"] = pd.to_numeric(sub.get("max_key"), errors="coerce")
    sub = sub.dropna(subset=["min_key", "max_key"])

    if len(sub) == 0:
        return pd.Series([], dtype=float)

    ranges = sub["max_key"] - sub["min_key"]
    ranges = ranges[ranges >= 0]
    return pd.Series(ranges, dtype=float)

def log10p1(values: pd.Series) -> pd.Series:
    """Compute log10(values + 1) safely."""
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

def find_spike_range(values: np.ndarray, window_width: float = 1000.0):
    """Find the densest [min, max] interval with fixed window_width."""
    if values.size == 0:
        return np.nan, np.nan, 0.0, 0

    x_sorted = np.sort(values)
    total_count = len(x_sorted)
    max_count = 0
    best_lower = x_sorted[0]

    right = 0
    for left in range(total_count):
        while right < total_count and x_sorted[right] <= x_sorted[left] + window_width:
            right += 1
        if (right - left) > max_count:
            max_count = right - left
            best_lower = x_sorted[left]

    lower = float(best_lower)
    upper = float(best_lower + window_width)
    ratio = max_count / total_count if total_count > 0 else 0.0
    return lower, upper, ratio, int(max_count)

def main():
    # --- Plot Style Configuration ---
    STYLE = {
        'fonts': ['Calibri', 'Liberation Sans', 'Arial', 'Ubuntu'],
        'fs_label': 22,        # Font size for axis labels
        'fs_title': 24,        # Font size for subplot titles
        'fs_tick': 18,         # Font size for axis ticks
        'fs_legend': 14,       # Font size for legend
        'label_pad': 10,       # Padding for axis labels
        'hspace': 0.4,         # Height space between subplots
        'wspace': 0.3,         # Width space between subplots
        'fig_width': 18,       # Figure width
        'fig_height_per_row': 5, # Figure height per row
        'highlight_color': '#D35400', # Color for highlighted data
        'grid_color': '#e0e0e0',      # Color for grid lines
    }

    parser = argparse.ArgumentParser(description="Integrated plot script for SSTable distributions.")
    parser.add_argument("csv_files", nargs="+", help="SSTable CSV files")
    parser.add_argument("--metric", choices=["kd", "entry", "gap", "range"], default="kd",
                        help="Metric to plot: 'kd', 'entry', 'gap', or 'range' (SST max_key - min_key)")
    parser.add_argument("--xlog", action="store_true", help="Plot x-axis as log10(val+1)")
    parser.add_argument("--auto-trim", action="store_true", help="Automatically detect and trim upper-tail outliers using IQR")
    parser.add_argument("--auto-iqr-mult", type=float, default=1.5, help="IQR multiplier for the upper fence (default: 1.5)")
    parser.add_argument(
        "--remove-spike",
        action="store_true",
        help="For --metric entry, remove spike range and plot tail values only.",
    )
    parser.add_argument(
        "--entry-spike-window",
        type=float,
        default=1000.0,
        help="Spike window width for --remove-spike (default: 1000).",
    )
    parser.add_argument("--labels", nargs="+", help="Custom labels for the legend")
    parser.add_argument("--output", help="Output filename (optional)")
    
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
            
            run_name = shorten_run_name(path)
            run_data[run_name] = df
            all_levels.update(df['level'].unique())
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if not run_data:
        print("No valid data found.")
        sys.exit(1)

    # Label handling (plot_summary 규격 반영)
    if args.labels:
        labels = args.labels
        if len(labels) < len(run_data):
            labels.extend([list(run_data.keys())[i] for i in range(len(labels), len(run_data))])
        else:
            labels = labels[:len(run_data)]
    else:
        labels = list(run_data.keys())

    levels = sorted([int(l) for l in all_levels])
    n_levels = len(levels)
    cols = min(3, n_levels) if n_levels > 0 else 1
    rows = math.ceil(n_levels / cols) if n_levels > 0 else 1
    
    # Font setup
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    target_font = 'sans-serif'
    for candidate in STYLE['fonts']:
        if candidate in available_fonts:
            target_font = candidate
            break
    plt.rcParams['font.family'] = target_font

    fig, axes = plt.subplots(rows, cols, figsize=(STYLE['fig_width'], STYLE['fig_height_per_row'] * rows), squeeze=False)
    fig.subplots_adjust(hspace=STYLE['hspace'], wspace=STYLE['wspace'])
    
    title_map = {
        "kd": "Key Density Distribution Comparison by Level",
        "entry": "Entry Count (num keys) Distribution Comparison by Level",
        "gap": "Gap Distribution Comparison by Level",
        "range": "Key Range Distribution Comparison by Level"
    }
    xlabel_map = {
        "kd": "Key Density",
        "entry": "Num keys",
        "gap": "Gap",
        "range": "Range (max_key - min_key)"
    }
    
    # Keep title mapping for consistency, even though suptitle is intentionally disabled.
    base_title = title_map[args.metric]
    if args.xlog:
        base_title += ' (Log Scale)'
    if args.auto_trim:
        base_title += f' [Auto-Trim k={args.auto_iqr_mult}]'
    
    axes_flat = axes.flatten()
    
    # Use tab10 colormap for distinct colors
    colors = plt.cm.tab10.colors
    run_order = list(run_data.keys())
    run_idx_map = {name: i for i, name in enumerate(run_order)}
    label_map = {}
    for i, run_name in enumerate(run_order):
        if i < len(labels):
            label_map[run_name] = labels[i]
        else:
            label_map[run_name] = run_name
    
    for idx, lvl in enumerate(levels):
        ax = axes_flat[idx]
        ax.set_facecolor('#ffffff')
        ax.grid(True, axis='y', linestyle='--', color=STYLE['grid_color'], zorder=0)

        lvl_plot_data = []
        run_names_present = []
        
        for run_name, df in run_data.items():
            lvl_df = df[df['level'] == lvl]
            if lvl_df.empty:
                continue
                
            if args.metric == "kd":
                min_k = pd.to_numeric(lvl_df['min_key'], errors='coerce')
                max_k = pd.to_numeric(lvl_df['max_key'], errors='coerce')
                en = pd.to_numeric(lvl_df['entry_n'], errors='coerce')
                valid = (en > 0) & (max_k >= min_k)
                res = (max_k[valid] - min_k[valid] + 1) / en[valid]
            elif args.metric == "entry":
                res = pd.to_numeric(lvl_df["entry_n"], errors="coerce").dropna()
                res = res[res > 0]
            elif args.metric == "gap":
                res = compute_level_gaps(lvl_df)
            elif args.metric == "range":
                res = compute_level_ranges(lvl_df)
            
            if res.empty:
                continue

            if args.metric == "entry" and args.remove_spike:
                en_vals = res.to_numpy(dtype=float)
                sp_min, sp_max, sp_ratio, sp_count = find_spike_range(
                    en_vals, window_width=args.entry_spike_window
                )
                mask_spike = (en_vals >= sp_min) & (en_vals <= sp_max)
                tail_vals = en_vals[~mask_spike]
                print(
                    f"[entry-spike] L{lvl} {run_name}: spike=[{sp_min:.1f},{sp_max:.1f}] "
                    f"ratio={sp_ratio:.3f} count={sp_count}, tail={tail_vals.size}"
                )
                res = pd.Series(tail_vals)
                if res.empty:
                    continue
            
            if args.auto_trim:
                res = auto_trim_upper_iqr(res, args.auto_iqr_mult)
            
            p_data = log10p1(res) if args.xlog else res
            lvl_plot_data.append(p_data)
            run_names_present.append(run_name)
            
        if not lvl_plot_data:
            continue
            
        # Determine global range for this level
        all_vals = pd.concat(lvl_plot_data)
        g_min, g_max = all_vals.min(), all_vals.max()
        if g_min == g_max:
            eps = max(1e-9, abs(float(g_min)) * 1e-6)
            g_min -= eps
            g_max += eps
        
        for color_idx, (p_data, run_name) in enumerate(zip(lvl_plot_data, run_names_present)):
            run_color_idx = run_idx_map.get(run_name, color_idx)
            current_color = colors[run_color_idx % len(colors)]

            # Layer 1: light fill for overlap context
            ax.hist(
                p_data,
                bins=50,
                range=(g_min, g_max),
                histtype='stepfilled',
                edgecolor='none',
                alpha=0.18,
                color=current_color,
                zorder=2,
            )
            # Layer 2: crisp outline so overlapping runs remain distinguishable
            ax.hist(
                p_data,
                bins=50,
                range=(g_min, g_max),
                histtype='step',
                linewidth=2.0,
                alpha=0.95,
                color=current_color,
                zorder=4,
            )
            print(
                f"L{lvl} [{run_name}]: count={len(p_data):4d}, "
                f"mean={float(p_data.mean()):>15,.1f}"
            )

        # Axis labeling with style config
        ax.set_xlabel(f"log10({xlabel_map[args.metric]} + 1)" if args.xlog else xlabel_map[args.metric], 
                      fontsize=STYLE['fs_label'], labelpad=STYLE['label_pad'])
        ax.set_ylabel('Count', fontsize=STYLE['fs_label'], labelpad=STYLE['label_pad'])
        ax.set_title(f'Level {lvl}', fontsize=STYLE['fs_title'], fontweight='bold')
        
        # Tick parameters
        ax.tick_params(axis='both', labelsize=STYLE['fs_tick'])
        
        # entry 메트릭일 경우 x축 눈금을 64k 하나로 고정
        if args.metric == "entry":
            tick_val = 64000
            tick_pos = np.log10(tick_val + 1) if args.xlog else tick_val
            ax.set_xticks([tick_pos])
            ax.set_xticklabels(["64k"])
        
        # Y축 상단 여유
        curr_ylim = ax.get_ylim()
        ax.set_ylim(0, curr_ylim[1] * 1.2)
        
    # Shared legend: prefer first empty grid cell; fallback to outside-right
    legend_handles = [
        Line2D([0], [0], color=colors[run_idx_map[name] % len(colors)], lw=2.5)
        for name in run_order
    ]
    legend_labels = [label_map[name] for name in run_order]

    if n_levels < len(axes_flat):
        legend_ax = axes_flat[n_levels]
        legend_ax.axis("off")
        legend_ax.legend(
            legend_handles,
            legend_labels,
            fontsize=STYLE['fs_legend'],
            loc='center',
            frameon=False,
        )
        for idx in range(n_levels + 1, len(axes_flat)):
            axes_flat[idx].set_visible(False)
    else:
        fig.subplots_adjust(right=0.82)
        fig.legend(
            legend_handles,
            legend_labels,
            fontsize=STYLE['fs_legend'],
            loc='center left',
            bbox_to_anchor=(0.84, 0.5),
            frameon=False,
        )
    
    if args.output:
        output_file = args.output
    else:
        suffix = "_log" if args.xlog else ""
        spike_suffix = "_tail" if (args.metric == "entry" and args.remove_spike) else ""
        output_file = f"{args.metric}_distribution_overlap{spike_suffix}{suffix}.png"
        
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved overlap plot: {output_file}")

if __name__ == "__main__":
    main()
