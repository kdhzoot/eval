#!/usr/bin/env python3
"""Plot gap histogram for each level in SSTable data from a single run.

This script is intentionally parallel to plot_kd_distribution.py, but plots
*gap* instead of key density.

Gap definition (within a level):
- Sort SSTables by (min_key, max_key)
- gap_i = max(0, next.min_key - prev.max_key)

Usage:
  python3 plot_gap_distribution.py [run_csv]

Example:
  python3 plot_gap_distribution.py ../sstables_icl_input_csv/sstables_001_....csv
"""

import argparse
import sys
import math
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import gaussian_kde  # type: ignore
except Exception:
    gaussian_kde = None


def log10p1(values: pd.Series) -> pd.Series:
    """Compute log10(values + 1) safely for non-negative series."""
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    # gaps are >=0 by construction; still guard against negatives just in case
    v = v[v >= 0]
    return (v + 1.0).apply(lambda x: math.log10(x))


def compute_level_gaps(level_df: pd.DataFrame) -> pd.Series:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-level gap histogram from a single run CSV")
    p.add_argument("run_csv", help="Path to per-run CSV file")

    def _percent_0_1(s: str) -> float:
        v = float(s)
        if v < 0 or v >= 1:
            raise argparse.ArgumentTypeError("must be in [0, 1)")
        return v

    p.add_argument(
        "--kde",
        action="store_true",
        help=(
            "Plot a smooth PDF using Gaussian KDE instead of a histogram. "
            "Recommended with --xlog for heavy-tailed gaps."
        ),
    )
    p.add_argument(
        "--kde-bw-method",
        choices=["scott", "silverman"],
        default="scott",
        help="KDE bandwidth rule (default: scott)",
    )
    p.add_argument(
        "--kde-bw-mult",
        type=float,
        default=1.0,
        help=(
            "KDE bandwidth multiplier (>1 smoother, <1 more wiggly). "
            "Applied on top of the chosen bw-method. (default: 1.0)"
        ),
    )
    p.add_argument(
        "--kde-gridsize",
        type=int,
        default=1024,
        help="Number of x points for KDE curve (default: 1024)",
    )
    p.add_argument(
        "--trim-percent",
        type=_percent_0_1,
        default=0.0,
        help="When plotting, drop the largest tail by this fraction (default: 0.0 == no trimming)",
    )
    p.add_argument(
        "--auto-trim",
        action="store_true",
        help=(
            "Automatically detect and trim upper-tail outliers per level (ignores --trim-percent). "
            "Uses Tukey IQR fence: keep <= Q3 + k*IQR."
        ),
    )
    p.add_argument(
        "--auto-iqr-mult",
        type=float,
        default=1.5,
        help="When --auto-trim: IQR multiplier k for the upper fence (default: 1.5)",
    )
    p.add_argument(
        "--dump-level",
        type=int,
        default=None,
        help="If set, dump ALL gap values for the given level (ascending) to stdout",
    )
    p.add_argument(
        "--xlog",
        action="store_true",
        help="Plot x as log10(gap+1) to compress long tails (stats remain on raw gap)",
    )
    return p.parse_args()


def plot_kde_pdf(
    ax: plt.Axes,
    values: pd.Series,
    bw_method: str,
    bw_mult: float,
    gridsize: int,
) -> None:
    """Plot a KDE-based PDF on the given axes."""
    if gaussian_kde is None:
        raise RuntimeError("--kde requires scipy (scipy.stats.gaussian_kde)")

    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(v) < 2:
        # Not enough points for KDE; fallback to empty plot.
        return

    data = v.to_numpy(dtype=float)
    kde = gaussian_kde(data, bw_method=bw_method)
    if bw_mult and float(bw_mult) != 1.0:
        # kde.factor is the internally chosen bandwidth factor.
        kde.set_bandwidth(bw_method=kde.factor * float(bw_mult))
    lo = float(np.min(data))
    hi = float(np.max(data))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    gs = int(gridsize)
    if gs < 128:
        gs = 128
    xs = np.linspace(lo, hi, gs)
    ys = kde(xs)
    ax.plot(xs, ys, color="steelblue", linewidth=1.5)
    ax.fill_between(xs, ys, color="steelblue", alpha=0.25)


def clip_series_for_plot(values: pd.Series, trim_percent: float) -> pd.Series:
    """Return a filtered series for plotting.

    - If trim_percent > 0: drop the largest tail by that fraction
    - Stats are computed on the raw (unclipped) series elsewhere
    """
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(v) == 0:
        return v

    # Trim upper tail (optional)
    tp = float(trim_percent)
    if tp > 0 and len(v) > 0:
        hi = float(v.quantile(1.0 - tp))
        v = v[v <= hi]

    return v


def auto_trim_upper_iqr(values: pd.Series, iqr_mult: float) -> Tuple[pd.Series, float, Optional[float]]:
    """Auto-trim using an upper Tukey fence.

    Returns:
      (trimmed_values, trim_percent, upper_fence)

    - trimmed_values is a filtered view of the raw values (<= upper_fence)
    - trim_percent is the *observed* fraction removed from the raw values
    - upper_fence is Q3 + k*IQR, or None if not enough data
    """
    v = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(v) == 0:
        return v, 0.0, None
    if len(v) < 4:
        # Too few points to estimate IQR robustly; keep all.
        return v, 0.0, None

    q1 = float(v.quantile(0.25))
    q3 = float(v.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        # Degenerate distribution; trimming would be arbitrary.
        return v, 0.0, None

    k = float(iqr_mult)
    upper = q3 + k * iqr
    kept = v[v <= upper]
    removed = len(v) - len(kept)
    pct = (removed / len(v)) if len(v) > 0 else 0.0
    return kept, pct, upper


def main():
    args = parse_args()
    csv_file = args.run_csv

    # Read CSV
    df = pd.read_csv(csv_file)

    # Get run name from filename
    run_name = Path(csv_file).stem.replace("_sstables", "")

    # Normalize level
    df["level"] = pd.to_numeric(df.get("level"), errors="coerce")
    df = df.dropna(subset=["level"])
    df["level"] = df["level"].astype(int)

    # Get sorted levels
    levels = sorted(df["level"].unique())

    print(f"Analyzing: {run_name}")
    print(f"Total SSTable: {len(df)}")
    print(f"Levels: {levels}")
    print()

    if args.dump_level is not None:
        lvl = int(args.dump_level)
        level_df = df[df["level"] == lvl]
        lvl_data = compute_level_gaps(level_df)
        lvl_data = lvl_data.sort_values(kind="mergesort")

        print(f"[DUMP] Level {lvl} gaps (ascending). count={len(lvl_data)}")
        for v in lvl_data.to_list():
            # keep it simple: one value per line, easy to redirect
            print(int(v))
        return

    # Calculate grid dimensions
    n_levels = len(levels)
    cols = min(3, n_levels) if n_levels > 0 else 1
    rows = math.ceil(n_levels / cols) if n_levels > 0 else 1

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    fig.suptitle(f"Gap Distribution by Level - {run_name}", fontsize=14, fontweight="bold")

    # Flatten axes for easier iteration
    if n_levels == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten() if rows * cols > 1 else [axes]

    # Plot histogram for each level
    for idx, lvl in enumerate(levels):
        ax = axes_flat[idx]
        level_df = df[df["level"] == lvl]
        lvl_data_raw = compute_level_gaps(level_df)

        if args.auto_trim:
            lvl_data, auto_pct, upper_fence = auto_trim_upper_iqr(
                lvl_data_raw,
                iqr_mult=args.auto_iqr_mult,
            )

            if upper_fence is None:
                # Not enough data / degenerate IQR; nothing removed.
                print(
                    f"L{lvl}: auto-trim k={args.auto_iqr_mult:g}, "
                    f"kept={len(lvl_data)}/{len(lvl_data_raw)} (trim=0.0%, fence=N/A)"
                )
            else:
                removed = len(lvl_data_raw) - len(lvl_data)
                print(
                    f"L{lvl}: auto-trim k={args.auto_iqr_mult:g}, fence={upper_fence:,.0f}, "
                    f"removed={removed}/{len(lvl_data_raw)} (trim={auto_pct*100:.2f}%), kept={len(lvl_data)}"
                )
        else:
            lvl_data = clip_series_for_plot(
                lvl_data_raw,
                trim_percent=args.trim_percent,
            )
            auto_pct = args.trim_percent
            upper_fence = None

        plot_data = log10p1(lvl_data) if args.xlog else lvl_data

        if args.kde:
            plot_kde_pdf(
                ax,
                plot_data,
                bw_method=args.kde_bw_method,
                bw_mult=args.kde_bw_mult,
                gridsize=args.kde_gridsize,
            )
        else:
            ax.hist(plot_data, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("log10(gap+1)" if args.xlog else "Gap", fontsize=10)
        ax.set_ylabel("Density" if args.kde else "Frequency", fontsize=10)
        if args.auto_trim:
            if upper_fence is None:
                ax.set_title(
                    f"Level {lvl} (n={len(lvl_data)}/{len(lvl_data_raw)})",
                    fontsize=11,
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    f"Level {lvl} (n={len(lvl_data)}/{len(lvl_data_raw)}, trim={auto_pct*100:.1f}%)",
                    fontsize=11,
                    fontweight="bold",
                )
        elif args.trim_percent > 0:
            ax.set_title(
                f"Level {lvl} (n={len(lvl_data)}/{len(lvl_data_raw)}, trim={args.trim_percent*100:.1f}%)",
                fontsize=11,
                fontweight="bold",
            )
        else:
            ax.set_title(f"Level {lvl} (n={len(lvl_data)})", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Stats should be computed on raw (unclipped) data.
        if len(lvl_data_raw) > 0:
            stats_text = (
                f"Mean: {lvl_data_raw.mean():,.0f}\n"
                f"Std: {lvl_data_raw.std():,.0f}\n"
                f"Min: {lvl_data_raw.min():,.0f}\n"
                f"Max: {lvl_data_raw.max():,.0f}"
            )
        else:
            stats_text = "(no data)"

        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        if len(lvl_data_raw) > 0:
            print(
                f"L{lvl}: count={len(lvl_data_raw):4d}, mean={lvl_data_raw.mean():>12,.0f}, std={lvl_data_raw.std():>12,.0f}, "
                f"min={lvl_data_raw.min():>12,.0f}, max={lvl_data_raw.max():>12,.0f}"
            )
        else:
            print(f"L{lvl}: count=   0")

    # Hide empty subplots
    for idx in range(n_levels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Save figure
    output_file = f"{run_name}_gap_distribution_logx.png" if args.xlog else f"{run_name}_gap_distribution.png"
    # Leave room for suptitle to avoid overlap.
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
