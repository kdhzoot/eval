#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""\
Draw 2x2 occurrence (histogram counts) of sequence for multiple levels over
[0, 280,000,000], overlaying multiple CSV files on the same axes.

Usage:
    python plot_level_sequence.py out1.csv [out2.csv out3.csv ...] \
            --levels 1,2,3,4 --bins 10000 --smooth 0 \
            --labels "Run A,Run B,Run C" -o seq_2x2.png --title "My Title"

Options:
- --bins   : number of histogram bins (>1)
- --smooth : moving-average window for occurrence (bins). 0=off.
- --labels : comma-separated legend labels for CSV files. If omitted, use basenames.
- --title  : override the default figure suptitle.
"""

import argparse
import os
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 24
})

SEQ_MIN = 0
SEQ_MAX = 280_000_000

def scan_csv_with_schema(csv_path: str):
    # Robust lazy scan with explicit dtypes
    common = dict(has_header=True, infer_schema_length=0, ignore_errors=True)
    try:
        return pl.scan_csv(csv_path, schema_overrides={"sequence": pl.Int64, "level": pl.Int64}, **common)
    except TypeError:
        return pl.scan_csv(csv_path, dtypes={"sequence": pl.Int64, "level": pl.Int64}, **common)

def groupby_compat(lf: pl.LazyFrame, col: str):
    # Polars API compatibility for older versions
    return lf.group_by(col) if hasattr(lf, "group_by") else lf.groupby(col)

def moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if not k or k <= 1:
        return arr
    k = int(k)
    pad = k // 2
    kernel = np.ones(k, dtype=float) / k
    padded = np.pad(arr, (pad, k - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")  # same length as input

def compute_hist_counts(csv_path: str, level: int, bins: int):
    span = SEQ_MAX - SEQ_MIN
    if bins <= 1 or span <= 0:
        raise ValueError("Invalid bins or sequence span")
    bin_width = span / bins

    lf = (
        scan_csv_with_schema(csv_path)
        .filter(
            (pl.col("level") == level) &
            (pl.col("sequence") >= SEQ_MIN) &
            (pl.col("sequence") <= SEQ_MAX)
        )
        .with_columns(
            (
                ((pl.col("sequence") - SEQ_MIN) * bins / span)
                .floor()
                .cast(pl.Int64)
                .clip(0, bins - 1)
                .alias("bin")
            )
        )
    )
    gb = groupby_compat(lf, "bin")
    grouped = gb.agg(pl.len().alias("n")).collect(streaming=True)

    all_bins = pl.DataFrame({"bin": pl.Series(range(bins), dtype=pl.Int64)})
    counts = all_bins.join(grouped, on="bin", how="left").fill_null(0)

    total = int(counts["n"].sum())
    bin_idx = np.array(counts["bin"].to_list(), dtype=np.int64)
    n_counts = np.array(counts["n"].to_list(), dtype=float)

    bin_centers = SEQ_MIN + (bin_idx + 0.5) * bin_width
    return bin_centers, n_counts, total, bin_width

def compute_occurrence(csv_path: str, level: int, bins: int, smooth: int):
    x, n_counts, total, _bin_width = compute_hist_counts(csv_path, level, bins)
    if total == 0:
        return x, np.zeros_like(x), total
    y = n_counts
    if smooth > 0:
        y = moving_average(y, smooth)
        y = np.clip(y, 0, None)
    return x, y, total

def parse_args():
    p = argparse.ArgumentParser(description="Overlay 2x2 occurrence (histogram counts) for multiple levels across multiple CSV files.")
    p.add_argument("csvs", nargs="+", help="One or more CSV files with columns: sequence,key,level")
    p.add_argument("--levels", type=str, default="1,2,3,4",
                   help="Comma-separated 4 levels, e.g., '1,2,3,4'")
    p.add_argument("--bins", type=int, default=10000, help="Number of histogram bins (>1)")
    p.add_argument("--smooth", type=int, default=0, help="Moving-average window for occurrence (bins). 0=off")
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated legend labels. If omitted, basenames of CSV files are used.")
    p.add_argument("--title", type=str, default=None,
                   help="Override the default figure suptitle.")
    p.add_argument("-o", "--out", default="sequence_2x2.png", help="Output PNG path")
    return p.parse_args()

def make_labels(csvs, labels_arg):
    if labels_arg:
        labels = [s.strip() for s in labels_arg.split(",")]
        if len(labels) != len(csvs):
            raise SystemExit(f"--labels count ({len(labels)}) must match number of CSVs ({len(csvs)}).")
        return labels
    # Default: file basenames
    return [os.path.basename(p) for p in csvs]

def main():
    args = parse_args()
    levels = [int(x) for x in args.levels.replace(" ", "").split(",") if x != ""]
    if len(levels) != 4:
        raise SystemExit(f"--levels must contain exactly 4 integers (got {len(levels)}). e.g., --levels 1,2,3,4")

    labels = make_labels(args.csvs, args.labels)

    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True, sharey=False)
    axes = np.array(axes).reshape(rows, cols)

    # Optional nicer x tick formatter (e.g., 1.2e8 -> 120M)
    def human_readable(x, pos):
        return f"{int(x/1e6):d}M" if x >= 1e6 else f"{int(x):d}"
    xfmt = FuncFormatter(human_readable)

    for idx, lvl in enumerate(levels):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        # Plot each CSV on the same axes
        totals_for_legend = []
        for csv_path, lab in zip(args.csvs, labels):
            x, y, total = compute_occurrence(csv_path, lvl, args.bins, args.smooth)
            ax.plot(x, y, linewidth=1.2, label=lab)
            totals_for_legend.append(total)

        # Axes cosmetics (no explicit fontsize; rcParams controls sizes)
        ax.set_title(f"level={lvl}")
        ax.set_xlim(SEQ_MIN, SEQ_MAX)
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_xlabel("Key Sequence")
        ax.set_ylabel("Occurrence")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Figure title
    if args.title:
        sup = args.title
    else:
        sup = f"Occurrence per level"
        if args.smooth > 0:
            sup += f" | smoothed k={int(args.smooth)}"
    fig.suptitle(sup)

    fig.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    fig.savefig(args.out, dpi=150)

if __name__ == "__main__":
    main()
