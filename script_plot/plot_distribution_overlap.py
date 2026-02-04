#!/usr/bin/env python3
"""
Integrated plot script for SSTable distributions (KD, Entry, Gaps).
Combines functionality from plot_kd_overlap.py, plot_entry_overlap.py, and plot_gap_overlap.py.
"""

import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def main():
    parser = argparse.ArgumentParser(description="Integrated plot script for SSTable distributions.")
    parser.add_argument("csv_files", nargs="+", help="SSTable CSV files")
    parser.add_argument("--metric", choices=["kd", "entry", "gap"], default="kd", 
                        help="Metric to plot: 'kd' (Key Density), 'entry' (Entry Count), or 'gap' (Gaps between SSTables)")
    parser.add_argument("--xlog", action="store_true", help="Plot x-axis as log10(val+1)")
    parser.add_argument("--auto-trim", action="store_true", help="Automatically detect and trim upper-tail outliers using IQR")
    parser.add_argument("--auto-iqr-mult", type=float, default=1.5, help="IQR multiplier for the upper fence (default: 1.5)")
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
    
    # 폰트 설정 (plot_summary 규격 반영)
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    target_font = 'sans-serif'
    for candidate in ['Calibri', 'Liberation Sans', 'Arial', 'Ubuntu']:
        if candidate in available_fonts:
            target_font = candidate
            break
    plt.rcParams['font.family'] = target_font

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows), squeeze=False)
    # 여백 조절 (plot_summary 규격 반영)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    title_map = {
        "kd": "Key Density Distribution Comparison by Level",
        "entry": "Entry Count (num keys) Distribution Comparison by Level",
        "gap": "Gap Distribution Comparison by Level"
    }
    xlabel_map = {
        "kd": "Key Density",
        "entry": "Num keys",
        "gap": "Gap"
    }
    
    base_title = title_map[args.metric]
    if args.xlog:
        base_title += ' (Log Scale)'
    if args.auto_trim:
        base_title += f' [Auto-Trim k={args.auto_iqr_mult}]'
    
    # fig.suptitle(base_title, fontsize=28, fontweight='bold', y=0.98) # 제목 제거 (user request)
    
    axes_flat = axes.flatten()
    
    # Use tab10 colormap for distinct colors
    colors = plt.cm.tab10.colors
    
    for idx, lvl in enumerate(levels):
        ax = axes_flat[idx]
        ax.set_facecolor('#ffffff')
        ax.grid(True, axis='y', linestyle='--', color='#e0e0e0', zorder=0)

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
            elif args.metric == "gap":
                res = compute_level_gaps(lvl_df)
            
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
        
        n_runs_present = len(run_data)
        for color_idx, (p_data, run_name) in enumerate(zip(lvl_plot_data, run_names_present)):
            # 1번 전략: 1~5번은 연하게(alpha 0.3), 6번만 진하게(alpha 0.8) 스타일링
            is_highlight = (color_idx == 5 and n_runs_present == 6)
            
            if is_highlight:
                alpha = 0.8
                edge_color = 'black'
                zorder = 5
                current_color = '#D35400' # 강조 색상 (찐한 주황)
            else:
                alpha = 0.3
                edge_color = 'none'  # 배경 데이터는 외곽선 제거하여 부드럽게
                zorder = 3
                current_color = colors[color_idx % len(colors)]
            
            current_label = labels[color_idx]

            ax.hist(p_data, bins=50, range=(g_min, g_max), edgecolor=edge_color, 
                    alpha=alpha, color=current_color, label=current_label, zorder=zorder)
            
            print(f"L{lvl} [{run_name}]: count={len(p_data):4d}, mean={p_data.mean():>15,.1f}")

        # 축 제목 및 설정 (plot_summary 규격 반영)
        ax.set_xlabel(f"log10({xlabel_map[args.metric]} + 1)" if args.xlog else xlabel_map[args.metric], 
                      fontsize=22, labelpad=10)
        ax.set_ylabel('Count', fontsize=22, labelpad=10)
        ax.set_title(f'Level {lvl}', fontsize=24, fontweight='bold')
        
        # 눈금 폰트 사이즈 (18pt)
        ax.tick_params(axis='both', labelsize=18)
        
        # entry 메트릭일 경우 x축 눈금을 64k 하나로 고정
        if args.metric == "entry":
            tick_val = 64000
            tick_pos = np.log10(tick_val + 1) if args.xlog else tick_val
            ax.set_xticks([tick_pos])
            ax.set_xticklabels(["64k"])
        
        # Y축 상단 여유
        curr_ylim = ax.get_ylim()
        ax.set_ylim(0, curr_ylim[1] * 1.2)
        
        # 범례 설정 (handle 크기 규격 반영)
        ax.legend(fontsize=14, loc='upper right', handlelength=1.0, handleheight=1.0)
    
    for idx in range(n_levels, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    if args.output:
        output_file = args.output
    else:
        suffix = "_log" if args.xlog else ""
        output_file = f"{args.metric}_distribution_overlap{suffix}.png"
        
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved overlap plot: {output_file}")

if __name__ == "__main__":
    main()
