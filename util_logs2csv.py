#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast aggregator for scan.sh readstat logs.

Input format (per line):
  <seq_no> <key_hex> <m1> <m2> <m3> <m4> <m5> <m6>

Output:
  One-line summary containing sums across all valid lines.
"""

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

READSTAT_RE = re.compile(r"^readstat(\.\d+)?$")
NUMERIC_SUFFIX_RE = re.compile(r"\.(\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate readlogs/readstat.* into one-line sums.")
    p.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Root directory to scan (default: auto-detect readlogs).",
    )
    p.add_argument(
        "--out-file",
        "-o",
        default=None,
        help="Output CSV file path (default: <input_dir>_csv/total_sum.csv).",
    )
    p.add_argument(
        "--print-only",
        action="store_true",
        help="Do not write CSV file. Print one summary line only.",
    )
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        help="Number of parallel workers (default: cpu_count). Use 1 for single process.",
    )
    return p.parse_args()


def detect_default_input_dir() -> Optional[str]:
    candidates = [
        os.path.join(os.getcwd(), "readlogs"),
        os.path.join(os.getcwd(), "eval", "readlogs"),
        "/home/smrc/TTLoad/readlogs",
        "/home/smrc/TTLoad/eval/readlogs",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def readstat_sort_key(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    m = NUMERIC_SUFFIX_RE.search(base)
    if m:
        return (int(m.group(1)), path)
    return (10**18, path)


def find_readstat_files(root: str) -> List[str]:
    files: List[str] = []
    root_abs = os.path.abspath(root)
    for dirpath, _, filenames in os.walk(root_abs):
        for name in filenames:
            if READSTAT_RE.match(name):
                files.append(os.path.join(dirpath, name))
    files.sort(key=readstat_sort_key)
    return files


def default_out_file(input_dir: str) -> str:
    base = os.path.abspath(input_dir.rstrip(os.sep))
    out_dir = os.path.join(os.path.dirname(base), f"{os.path.basename(base)}_csv")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "total_sum.csv")


def summarize_fast(readstat_files: List[str]) -> Tuple[int, int, int, List[int], int]:
    total_files = 0
    total_rows = 0
    malformed_rows = 0
    sums = [0, 0, 0, 0, 0, 0]

    for fpath in readstat_files:
        total_files += 1
        with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
            for raw in fp:
                parts = raw.split()
                if len(parts) != 8:
                    malformed_rows += 1
                    continue
                try:
                    # parts[0]=seq_no, parts[1]=key_hex
                    m1 = int(parts[2])
                    sums[0] += 1 if m1 != 0 else 0
                    sums[1] += int(parts[3])
                    sums[2] += int(parts[4])
                    sums[3] += int(parts[5])
                    sums[4] += int(parts[6])
                    sums[5] += int(parts[7])
                except ValueError:
                    malformed_rows += 1
                    continue
                total_rows += 1

    metrics_sum = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5]
    return total_files, total_rows, malformed_rows, sums, metrics_sum


def _summarize_one_file(fpath: str) -> Tuple[int, int, List[int]]:
    rows = 0
    malformed = 0
    sums = [0, 0, 0, 0, 0, 0]
    with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
        for raw in fp:
            parts = raw.split()
            if len(parts) != 8:
                malformed += 1
                continue
            try:
                m1 = int(parts[2])
                sums[0] += 1 if m1 != 0 else 0
                sums[1] += int(parts[3])
                sums[2] += int(parts[4])
                sums[3] += int(parts[5])
                sums[4] += int(parts[6])
                sums[5] += int(parts[7])
            except ValueError:
                malformed += 1
                continue
            rows += 1
    return rows, malformed, sums


def summarize_parallel(readstat_files: List[str], jobs: int) -> Tuple[int, int, int, List[int], int]:
    total_files = len(readstat_files)
    total_rows = 0
    malformed_rows = 0
    sums = [0, 0, 0, 0, 0, 0]

    if jobs <= 1:
        return summarize_fast(readstat_files)

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for rows, malformed, part_sums in ex.map(_summarize_one_file, readstat_files, chunksize=1):
            total_rows += rows
            malformed_rows += malformed
            sums[0] += part_sums[0]
            sums[1] += part_sums[1]
            sums[2] += part_sums[2]
            sums[3] += part_sums[3]
            sums[4] += part_sums[4]
            sums[5] += part_sums[5]

    metrics_sum = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5]
    return total_files, total_rows, malformed_rows, sums, metrics_sum


def write_one_line_csv(
    out_file: str,
    total_files: int,
    total_rows: int,
    malformed_rows: int,
    sums: List[int],
    metrics_sum: int,
) -> None:
    out_abs = os.path.abspath(out_file)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "input_files",
                "rows_valid",
                "rows_malformed",
                "m1_nonzero_count",
                "m2_sum",
                "m3_sum",
                "m4_sum",
                "m5_sum",
                "m6_sum",
                "metrics_sum",
            ]
        )
        writer.writerow([total_files, total_rows, malformed_rows] + sums + [metrics_sum])


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    if input_dir is None:
        input_dir = detect_default_input_dir()
        if input_dir is None:
            print("[ERROR] input_dir is missing and readlogs directory was not auto-detected.")
            print("Usage: python util_logs2csv.py <readlogs_dir> [--out-file OUTPUT_FILE]")
            sys.exit(1)

    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    files = find_readstat_files(input_dir)
    if not files:
        print(f"[WARN] No readstat files found under: {input_dir}")
        sys.exit(0)

    print(f"[INFO] Input directory : {input_dir}")
    print(f"[INFO] Found readstat files: {len(files)}")
    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    print(f"[INFO] Parallel jobs: {jobs}")

    total_files, total_rows, malformed_rows, sums, metrics_sum = summarize_parallel(files, jobs)

    # one-line output the user can copy/paste quickly
    print(
        "RESULT,{},{},{},{},{},{},{},{},{},{}".format(
            total_files,
            total_rows,
            malformed_rows,
            sums[0],
            sums[1],
            sums[2],
            sums[3],
            sums[4],
            sums[5],
            metrics_sum,
        )
    )

    if args.print_only:
        return

    out_file = args.out_file if args.out_file else default_out_file(input_dir)
    write_one_line_csv(out_file, total_files, total_rows, malformed_rows, sums, metrics_sum)
    print(f"[DONE] one-line csv written: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    main()
