#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-SSTable entries from sstables benchmark output and write to individual CSVs.

- Input example:
    sstables_load_700GB_20251124_124704/
      ├─ sstables_001_20251124_124704/stdout.log
      ├─ sstables_002_20251124_124705/stdout.log
      └─ ...

- For each stdout.log, extract SST data and produce:
    output_dir/<run_name>_sstables.csv

CSV columns: level, id, size, entry_n, min_seq, max_seq, min_key, max_key
"""

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Any, Optional, Tuple

# Detect level headers like: "--- level 0 ---"
LEVEL_HDR_RE = re.compile(r"---\s*level\s+(\d+)\s*---", re.IGNORECASE)
# Detect SST lines like:
#   "123:456789:1000[111 .. 222][0011... .. ffff...](0)"
#   "128012:50710180[698558830 .. 699999841][hex .. hex](0)"
LINE_RE = re.compile(r"\s*(\d+):(\d+)(?::(\d+))?\[([^\]]+)\]\[([^\]]+)\]\(\d+\)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-SSTable data from sstables benchmark output into individual CSVs."
    )
    p.add_argument("root", help="Root directory (e.g., sstables_load_700GB_*)")
    p.add_argument(
        "--output-dir", "-o", default=None,
        help="Output directory for CSV files (default: <root_basename>_csv next to root)"
    )
    return p.parse_args()


def to_int_or_empty(s: str) -> Optional[int]:
    """Convert string to int, return None if not possible."""
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        return None


def hex16_prefix_to_int(s: str) -> Optional[int]:
    """Convert the first 16 hex chars of a key string to an int; return None if not possible."""
    try:
        return int(s.strip()[:16], 16)
    except Exception:
        return None


def parse_sst_log(path: str) -> List[Dict[str, Any]]:
    """
    Parse a single stdout.log and extract per-SSTable entries.
    Returns: list of {level, sst_id, size, min_seq, max_seq, min_key, max_key}
    """
    entries: List[Dict[str, Any]] = []
    current_level: Optional[int] = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m_level = LEVEL_HDR_RE.search(line)
            if m_level:
                current_level = int(m_level.group(1))
                continue

            m = LINE_RE.search(line)
            if not m or current_level is None:
                continue

            sst_id = int(m.group(1))
            size = int(m.group(2))
            entry_n_val = m.group(3)
            entry_n = int(entry_n_val) if entry_n_val is not None else None

            seq_parts = [p.strip() for p in m.group(4).split("..")]
            key_parts = [p.strip() for p in m.group(5).split("..")]

            min_seq = to_int_or_empty(seq_parts[0]) if len(seq_parts) >= 2 else None
            max_seq = to_int_or_empty(seq_parts[1]) if len(seq_parts) >= 2 else None
            min_key = hex16_prefix_to_int(key_parts[0]) if len(key_parts) >= 2 else None
            max_key = hex16_prefix_to_int(key_parts[1]) if len(key_parts) >= 2 else None

            entries.append({
                "level": current_level,
                "id": sst_id,
                "size": size,
                "entry_n": entry_n if entry_n is not None else "",
                "min_seq": min_seq if min_seq is not None else "",
                "max_seq": max_seq if max_seq is not None else "",
                "min_key": min_key if min_key is not None else "",
                "max_key": max_key if max_key is not None else "",
            })

    return entries


def find_run_logs(root: str) -> List[Tuple[str, str]]:
    """
    Find all stdout.log files under sstables_* subdirectories.
    Returns a list of (run_name, log_path).
    """
    pattern = os.path.join(os.path.abspath(root), "sstables_*", "stdout.log")
    paths = sorted(glob.glob(pattern))
    results: List[Tuple[str, str]] = []
    for p in paths:
        run_name = os.path.basename(os.path.dirname(p))
        results.append((run_name, p))
    return results


def get_output_dir(root: str, output: Optional[str]) -> str:
    """Determine output directory."""
    if output:
        os.makedirs(output, exist_ok=True)
        return output
    root_abs = os.path.abspath(root)
    root_base = os.path.basename(root_abs.rstrip(os.sep))
    out_dir = os.path.join(os.path.dirname(root_abs), f"{root_base}_csv")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_sst_csv(path: str, entries: List[Dict[str, Any]]) -> None:
    """Write per-SSTable entries to CSV file."""
    columns = ["level", "id", "size", "entry_n", "min_seq", "max_seq", "min_key", "max_key"]
    
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=columns)
        w.writeheader()
        for row in entries:
            w.writerow({k: row.get(k, "") for k in columns})


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root)
    
    if not os.path.isdir(root):
        print(f"[ERROR] Root directory does not exist: {root}")
        sys.exit(1)

    run_logs = find_run_logs(root)
    if not run_logs:
        print(f"[WARN] No stdout.log files found under '{root}/sstables_*/'")
        sys.exit(0)

    output_dir = get_output_dir(root, args.output_dir)
    print(f"[INFO] Found {len(run_logs)} runs")
    print(f"[INFO] Output directory: {output_dir}")

    processed = 0
    skipped = 0

    for run_name, log_path in run_logs:
        try:
            entries = parse_sst_log(log_path)
            if not entries:
                print(f"[WARN] No SST entries found: {log_path}")
                skipped += 1
                continue
            
            output_csv = os.path.join(output_dir, f"{run_name}.csv")
            write_sst_csv(output_csv, entries)
            
            print(f"[OK] {run_name}: {len(entries)} SSTables → {output_csv}")
            processed += 1
        except Exception as e:
            print(f"[ERROR] {run_name}: {e}")
            skipped += 1

    print(f"[DONE] Processed {processed} runs, skipped {skipped}")
    if processed > 0:
        print(f"       Output directory: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python util_sstables2csv.py <root_dir> [--output-dir output_dir]")
        sys.exit(1)
    main()
