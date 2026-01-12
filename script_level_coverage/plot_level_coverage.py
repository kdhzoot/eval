#!/usr/bin/env python3
"""Plot per-level SST coverage for a directory of RocksDB SST metadata CSVs.

This script scans all `run_*_entries.csv` files in the current directory,
computes the total key-space coverage contributed by non-overlapping SST file
ranges within each level, and renders them on a single plot for comparison.

Usage:
    python plot_level_coverage.py [--key-space 280000000] [--pattern 'run_*_entries.csv']

Outputs:
    - Displays the plot interactively (if a display is available)
    - Saves the figure as `level_coverage.png` alongside this script
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

# Default total key space (number of keys); can be overridden via CLI.
DEFAULT_KEY_SPACE = 280_000_000
SIZE_PATTERN = re.compile(r"_(\d+)(GB|TB)(?:_|$)", re.IGNORECASE)


def merge_ranges(ranges: Iterable[Tuple[int, int]]) -> int:
    """Return total coverage length for the given inclusive ranges.

    The function first sorts the ranges by start key, then merges any overlapping
    or adjacent segments. Return value is the total number of keys covered by the
    merged segments.
    """

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    if not sorted_ranges:
        return 0

    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= cur_end + 1:
            # Overlapping or contiguous interval → extend current block.
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))

    # Convert to total length (inclusive ranges → add 1)
    total = sum(end - start + 1 for start, end in merged)
    return total


def compute_level_stats(csv_path: Path, key_space: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Compute per-level coverage percentage and SST counts for a CSV file."""

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}, {}

    if "entry_n" not in df.columns:
        raise ValueError(
            f"CSV '{csv_path.name}' is missing required 'entry_n' column. "
            "Re-run util_sstables2csv.py with updated parser to populate entry counts."
        )

    coverage: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for level, group in df.groupby("level"):
        ranges = list(zip(group["min_key"].astype(int), group["max_key"].astype(int)))
        covered_keys = merge_ranges(ranges)
        pct = (covered_keys / key_space) * 100.0 if key_space else 0.0
        coverage[int(level)] = pct
        counts[int(level)] = int(group["entry_n"].astype(int).sum())
    return coverage, counts


def infer_key_space_from_label(label: str, default: int) -> int:
    """Infer key-space size from filename label (looks for _<size><GB|TB>_)."""

    match = SIZE_PATTERN.search(label)
    if not match:
        return default

    value = int(match.group(1))
    unit = match.group(2).upper()
    if unit == "GB":
        return value * 1_000_000
    if unit == "TB":
        return value * 1_000_000_000
    return default


def collect_coverages(
    csv_files: List[Path], default_key_space: int
) -> Dict[str, Tuple[Dict[int, float], Dict[int, int], int]]:
    data: Dict[str, Tuple[Dict[int, float], Dict[int, int], int]] = {}
    for csv_file in csv_files:
        label = csv_file.stem
        key_space = infer_key_space_from_label(label, default_key_space)
        coverage, counts = compute_level_stats(csv_file, key_space)
        data[label] = (coverage, counts, key_space)
    return data


def expected_blocks_per_lookup(
    keys_per_level: List[float],
    coverage_per_level: List[float],
    p_valid: float,
) -> Dict[str, object]:
    """Compute expected block reads per lookup, following provided formula."""

    N = keys_per_level
    c_vals = coverage_per_level
    assert len(N) == len(c_vals), "keys_per_level and coverage_per_level must have same length"
    L = len(N)

    if L == 0:
        return {
            "E_total_blocks": 0.0,
            "E_filter_blocks": 0.0,
            "E_index_blocks": 0.0,
            "E_data_blocks": 0.0,
            "per_level": [],
        }

    N_tot = float(sum(N)) if N else 0.0
    if N_tot <= 0.0:
        # No keys → no hits, only coverage-driven filter reads.
        per_level = []
        E_filter_total = sum(c_vals)
        for idx, c_val in enumerate(c_vals):
            per_level.append(
                {
                    "level": idx + 1,
                    "P_reach": 1.0 if idx == 0 else 1.0,
                    "p_hit": 0.0,
                    "c": c_val,
                    "E_filter": c_val,
                    "E_index": 0.0,
                    "E_data": 0.0,
                    "E_total": c_val,
                }
            )
        return {
            "E_total_blocks": E_filter_total,
            "E_filter_blocks": E_filter_total,
            "E_index_blocks": 0.0,
            "E_data_blocks": 0.0,
            "per_level": per_level,
        }

    phi = [n / N_tot for n in N]
    p_hit = [p_valid * x for x in phi]

    P_reach = [0.0] * L
    P_reach[0] = 1.0
    cumulative = 0.0
    for i in range(1, L):
        cumulative += p_hit[i - 1]
        P_reach[i] = max(0.0, 1.0 - cumulative)

    per_level = []
    E_filter_total = 0.0
    E_index_total = 0.0
    E_data_total = 0.0
    E_total_blocks = 0.0

    for i in range(L):
        E_filter = P_reach[i] * c_vals[i]
        E_index = p_hit[i]
        E_data = p_hit[i]
        E_total = E_filter + E_index + E_data

        E_filter_total += E_filter
        E_index_total += E_index
        E_data_total += E_data
        E_total_blocks += E_total

        per_level.append(
            {
                "level": i + 1,
                "P_reach": P_reach[i],
                "p_hit": p_hit[i],
                "c": c_vals[i],
                "E_filter": E_filter,
                "E_index": E_index,
                "E_data": E_data,
                "E_total": E_total,
            }
        )

    return {
        "E_total_blocks": E_total_blocks,
        "E_filter_blocks": E_filter_total,
        "E_index_blocks": E_index_total,
        "E_data_blocks": E_data_total,
        "per_level": per_level,
    }


def plot_coverages(
    data: Dict[str, Tuple[Dict[int, float], Dict[int, int], int]], output_path: Path
) -> None:
    if not data:
        print("[WARN] No coverage data to plot.")
        return

    all_levels = sorted(
        {level for coverage, _counts, _space in data.values() for level in coverage.keys()}
    )
    if not all_levels:
        print("[WARN] Coverage data contained no level entries.")
        return

    plt.figure(figsize=(10, 6))
    for label, (coverage, _counts, _key_space) in data.items():
        display_label = label
        if label.startswith("sstables_"):
            parts = label.split("_")
            if len(parts) >= 5:
                display_label = "_".join(parts[2:5])
        y_vals = [coverage.get(level, 0.0) for level in all_levels]
        plt.plot(all_levels, y_vals, marker="o", linewidth=1.0, alpha=0.7, label=display_label)

    plt.xlabel("Level")
    plt.ylabel("Coverage (%)")
    plt.title("Per-level SST coverage across runs")
    plt.ylim(0, 100)
    plt.xticks(all_levels)
    plt.legend(
        ncol=3,
        loc="lower right",
        frameon=False,
        fontsize=12,
        markerscale=0.8,
        handlelength=1.5,
        handletextpad=0.3,
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    print(f"[OK] Saved plot to {output_path}")

    try:
        plt.show()
    except Exception:
        # headless environments will fail to show; that's fine.
        pass


def write_summary_csv(
    data: Dict[str, Tuple[Dict[int, float], Dict[int, int], int]],
    output_path: Path,
    detail_output_path: Path,
) -> None:
    if not data:
        print("[WARN] No data available to write CSV summary.")
        return

    all_levels = sorted({level for coverage, _counts, _space in data.values() for level in coverage})
    if not all_levels:
        print("[WARN] No level information found while writing CSV summary.")
        return

    rows = []
    detail_rows = []
    for label, (coverage, counts, key_space) in data.items():
        row = {"run": label}
        for level in all_levels:
            row[f"L{level}_coverage_pct"] = coverage.get(level, 0.0)
            row[f"L{level}_entry_count"] = counts.get(level, 0)

        keys_list = [float(counts.get(level, 0)) for level in all_levels]
        coverage_list = [float(coverage.get(level, 0.0)) / 100.0 for level in all_levels]
        total_keys = sum(keys_list)
        p_valid = 0.63  # Assume entire key space corresponds to valid keys

        stats = expected_blocks_per_lookup(keys_list, coverage_list, p_valid)
        row["expected_total_blocks"] = stats["E_total_blocks"]
        row["expected_filter_blocks"] = stats["E_filter_blocks"]
        row["expected_index_blocks"] = stats["E_index_blocks"]
        row["expected_data_blocks"] = stats["E_data_blocks"]
        row["p_valid"] = p_valid
        row["total_keys"] = total_keys
        row["key_space"] = key_space

        print(
            f"[INFO] {label}: key_space={key_space}, total_keys={total_keys:.0f}, coverage_sum={sum(coverage_list):.3f}, "
            f"expected_total_blocks={stats['E_total_blocks']:.4f}"
        )

        for idx, level_stat in enumerate(stats["per_level"]):
            actual_level = all_levels[idx]
            detail_rows.append(
                {
                    "run": label,
                    "level": actual_level,
                    "coverage_pct": coverage.get(actual_level, 0.0),
                    "entry_count": counts.get(actual_level, 0),
                    "P_reach": level_stat["P_reach"],
                    "p_hit": level_stat["p_hit"],
                    "E_filter": level_stat["E_filter"],
                    "E_index": level_stat["E_index"],
                    "E_data": level_stat["E_data"],
                    "E_total": level_stat["E_total"],
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values("run", inplace=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved summary CSV to {output_path}")

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.sort_values(["run", "level"], inplace=True)
        detail_df.to_csv(detail_output_path, index=False)
        print(f"[OK] Saved per-level detail CSV to {detail_output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-level SST coverage for CSV runs.")
    parser.add_argument(
        "--key-space", type=int, default=DEFAULT_KEY_SPACE,
        help=f"Total key space size (default: {DEFAULT_KEY_SPACE:,})."
    )
    parser.add_argument(
        "--pattern", default="sstables_*.csv",
        help="Glob pattern for CSV files (default: sstables_*.csv)."
    )
    parser.add_argument(
        "--output", default="level_coverage.png",
        help="Filename for the generated plot (default: level_coverage.png)."
    )
    parser.add_argument(
        "--csv-output",
        default="level_coverage_summary.csv",
        help="Filename for the summary CSV (default: level_coverage_summary.csv).",
    )
    parser.add_argument(
        "--detail-csv-output",
        default="level_coverage_details.csv",
        help="Filename for per-level detailed CSV (default: level_coverage_details.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    csv_files = sorted(Path(base_dir).glob(args.pattern))

    if not csv_files:
        print(f"[ERROR] No CSV files matched pattern '{args.pattern}' in {base_dir}")
        return 1

    data = collect_coverages(csv_files, args.key_space)
    output_path = base_dir / args.output
    csv_output_path = base_dir / args.csv_output
    detail_output_path = base_dir / args.detail_csv_output
    plot_coverages(data, output_path)
    write_summary_csv(data, csv_output_path, detail_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
