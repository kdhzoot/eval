#!/usr/bin/env python3
"""
Plot gap histogram (Sum of heights = 1) with Scaled Pareto Overlay.
"""

import argparse
import sys
import math
from pathlib import Path
from typing import Tuple, Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto

# ==============================================================================
# [설정] 레벨별 Pareto 파라미터 (b, scale)
# ==============================================================================
"""
(0.179711, 0, 36.0)
(0.20803, 0, 3.0)
(0.585623, 0, 1.0)
(1.884638, 0, 1.0)

"""
LEVEL_PARAMS: Dict[int, Tuple[float, float]] = {
    2: (0.1797, 36.0),   # 예시: Level 2 (shape, scale)
    3: (0.2080, 3.0),   # 예시: Level 3
    4: (0.5856, 1.0),   # 예시: Level 4
    5: (1.8846, 1.0),   # 예시: Level 5
}

def compute_level_gaps(level_df: pd.DataFrame) -> pd.Series:
    sub = level_df.copy()
    sub["min_key"] = pd.to_numeric(sub.get("min_key"), errors="coerce")
    sub["max_key"] = pd.to_numeric(sub.get("max_key"), errors="coerce")
    sub = sub.dropna(subset=["min_key", "max_key"])
    if len(sub) < 2: return pd.Series([], dtype=float)
    sub = sub.sort_values(["min_key", "max_key"], kind="mergesort")
    gaps = sub["min_key"].to_numpy()[1:] - sub["max_key"].to_numpy()[:-1]
    return pd.Series(gaps.clip(min=0), dtype=float)

def auto_trim_upper_iqr(values: pd.Series, iqr_mult: float) -> Tuple[pd.Series, float, Optional[float]]:
    v = values.dropna()
    if len(v) < 4: return v, 0.0, None
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    upper = q3 + iqr_mult * (q3 - q1)
    kept = v[v <= upper]
    return kept, (len(v)-len(kept))/len(v), upper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_csv")
    parser.add_argument("--trim-percent", type=float, default=0.0)
    parser.add_argument("--auto-trim", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.run_csv)
    df["level"] = pd.to_numeric(df["level"], errors="coerce").fillna(-1).astype(int)
    levels = sorted([l for l in df["level"].unique() if l > 0])

    n_levels = len(levels)
    if n_levels == 0: return
    
    rows = math.ceil(n_levels / 3)
    fig, axes = plt.subplots(rows, min(3, n_levels), figsize=(15, 4 * rows))
    axes_flat = axes.flatten() if n_levels > 1 else [axes]

    for i, lvl in enumerate(levels):
        ax = axes_flat[i]
        raw_data = compute_level_gaps(df[df["level"] == lvl])
        
        # 1. Trimming
        plot_data = raw_data
        trim_info = ""
        if args.auto_trim:
            plot_data, pct, fence = auto_trim_upper_iqr(raw_data, 1.5)
            trim_info = f"\n(Auto-Trim: {pct*100:.1f}%)"
        elif args.trim_percent > 0:
            limit = raw_data.quantile(1.0 - args.trim_percent)
            plot_data = raw_data[raw_data <= limit]
            trim_info = f"\n(Trim: {args.trim_percent*100:.1f}%)"

        if len(plot_data) == 0: continue

        # 2. 히스토그램 (합이 1이 되도록 설정)
        # weights를 사용하여 각 데이터의 가중치를 1/N 로 설정하면 높이의 합이 1이 됨
        weights = np.ones_like(plot_data) / len(plot_data)
        
        # bins 값을 받아서 너비(width)를 계산해야 함
        counts, bin_edges, _ = ax.hist(plot_data, bins=50, weights=weights, 
                                       alpha=0.6, color="steelblue", edgecolor="black", label="Observed")
        
        ax.set_title(f"Level {lvl} (n={len(plot_data)}){trim_info}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probability (Sum=1)") # y축 라벨 변경

        # 3. Pareto Overlay (확률 단위로 변환)
        if lvl in LEVEL_PARAMS and lvl > 1:
            b, scale = LEVEL_PARAMS[lvl]
            trunc_limit = plot_data.max()
            x_min = max(scale, plot_data.min())
            
            if trunc_limit > x_min:
                x_line = np.linspace(x_min, trunc_limit, 1000)
                
                # (1) Truncated PDF 계산 (이전과 동일)
                pdf_vals = pareto.pdf(x_line, b, scale=scale)
                cdf_at_limit = pareto.cdf(trunc_limit, b, scale=scale)
                truncated_pdf = pdf_vals / cdf_at_limit if cdf_at_limit > 0 else pdf_vals
                
                # (2) [중요] PDF -> Probability 변환
                # PDF(밀도) * Bin_Width(구간 너비) = Probability(확률)
                # 히스토그램이 등간격 bin이라고 가정하고 첫 번째 bin의 너비를 사용
                bin_width = bin_edges[1] - bin_edges[0]
                y_line = truncated_pdf * bin_width 
                
                ax.plot(x_line, y_line, 'r-', lw=2.5, label=f"Pareto Model")
                ax.legend()

    plt.tight_layout()
    plt.savefig(f"{Path(args.run_csv).stem}_prob_overlay.png", dpi=150)
    print("Done.")

if __name__ == "__main__":
    main()