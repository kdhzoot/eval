#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize read benchmark results from stdout.log files into a single CSV.

Scans a benchmark directory to collect read throughput (ops/sec), latency (micros/op),
cache hit/miss statistics, and bloom filter metrics from RocksDB db_bench stdout logs.

Input:
    - A directory containing run subfolders (e.g., runs_readrandom_...), 
      where each subfolder contains a stdout.log file.

Output:
    - A single CSV file summarizing performance and internal RocksDB statistics per run.

Usage:
    python summarize_read_stats.py <bench_dir> [-o <out.csv>]
"""

import argparse
import os
import re
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Summarize read logs into a CSV.")
    p.add_argument("bench_dir", help="Parent directory containing run subfolders.")
    p.add_argument("-o", "--out", default=None, help="Output CSV path.")
    return p.parse_args()

def parse_stdout_log(filepath):
    """Parses a single stdout.log and returns a dictionary of metrics."""
    res = {}
    with open(filepath, 'r', errors='ignore') as f:
        content = f.read()

    # Throughput and Latency
    # Example: readrandom   :     680.805 micros/op 1468 ops/sec 680.805 seconds 1000000 operations;    0.9 MB/s (632057 of 1000000 found)
    m = re.search(r"readrandom\s+:\s+([\d\.]+)\s+micros/op\s+([\d\.]+)\s+ops/sec", content)
    if m:
        res['micros_op'] = float(m.group(1))
        res['ops_sec'] = float(m.group(2))
    
    m = re.search(r"(\d+)\s+of\s+(\d+)\s+found\)", content)
    if m:
        res['found_keys'] = int(m.group(1))
        res['total_ops'] = int(m.group(2))
        res['found_ratio'] = res['found_keys'] / res['total_ops'] if res['total_ops'] > 0 else 0

    # Block Read Statistics
    m = re.search(r"Total Filter block reads:\s+(\d+)", content)
    if m: res['filter_read'] = int(m.group(1))
    
    m = re.search(r"Total Index block reads:\s+(\d+)", content)
    if m: res['index_read'] = int(m.group(1))
    
    m = re.search(r"Total Data block reads:\s+(\d+)", content)
    if m: res['data_read'] = int(m.group(1))
    
    m = re.search(r"Total All block reads:\s+(\d+)", content)
    if m: res['total_block_read'] = int(m.group(1))

    # RocksDB Statistics
    m = re.search(r"rocksdb\.block\.cache\.miss COUNT :\s+(\d+)", content)
    if m: res['cache_miss'] = int(m.group(1))
    
    m = re.search(r"rocksdb\.block\.cache\.hit COUNT :\s+(\d+)", content)
    if m: res['cache_hit'] = int(m.group(2)) if m.lastindex >= 2 else 0 # Fixed below
    
    # Re-doing stats extraction more carefully
    stats_patterns = {
        'cache_miss': r"rocksdb\.block\.cache\.miss COUNT :\s+(\d+)",
        'cache_hit': r"rocksdb\.block\.cache\.hit COUNT :\s+(\d+)",
        'index_cache_miss': r"rocksdb\.block\.cache\.index\.miss COUNT :\s+(\d+)",
        'index_cache_hit': r"rocksdb\.block\.cache\.index\.hit COUNT :\s+(\d+)",
        'filter_cache_miss': r"rocksdb\.block\.cache\.filter\.miss COUNT :\s+(\d+)",
        'filter_cache_hit': r"rocksdb\.block\.cache\.filter\.hit COUNT :\s+(\d+)",
        'data_cache_miss': r"rocksdb\.block\.cache\.data\.miss COUNT :\s+(\d+)",
        'data_cache_hit': r"rocksdb\.block\.cache\.data\.hit COUNT :\s+(\d+)",
        'bloom_useful': r"rocksdb\.bloom\.filter\.useful COUNT :\s+(\d+)",
        'bloom_full_positive': r"rocksdb\.bloom\.filter\.full\.positive COUNT :\s+(\d+)",
        'bloom_true_positive': r"rocksdb\.bloom\.filter\.full\.true\.positive COUNT :\s+(\d+)",
    }
    
    for key, pattern in stats_patterns.items():
        m = re.search(pattern, content)
        if m:
            res[key] = int(m.group(1))

    # Level Hit Statistics
    hit_matches = re.findall(r"Level (\d+) hits:\s+(\d+)", content)
    for lvl, count in hit_matches:
        res[f'L{lvl}_hit'] = int(count)

    return res

def main():
    args = parse_args()
    bench_dir = os.path.abspath(args.bench_dir)
    
    if not args.out:
        base = os.path.basename(bench_dir.rstrip(os.sep))
        args.out = os.path.join(os.path.dirname(bench_dir), f"{base}_summary.csv")

    rows = []
    
    # Walk through subdirectories to find stdout.log
    for root, dirs, files in os.walk(bench_dir):
        if "stdout.log" in files:
            log_path = os.path.join(root, "stdout.log")
            try:
                metrics = parse_stdout_log(log_path)
                if not metrics or 'ops_sec' not in metrics:
                    # If ops_sec is missing, benchmark might not have finished or it's a different type of log
                    continue
                
                # Derive run_id and run_index
                rel_path = os.path.relpath(root, bench_dir)
                metrics['path'] = rel_path
                
                # Try to extract run index from path (e.g., run_001)
                m = re.search(r"run_(\d+)", rel_path)
                if m:
                    metrics['run_index'] = int(m.group(1))
                else:
                    # Assign a dummy large index or use order of discovery
                    metrics['run_index'] = 999
                
                # Use the subdirectory name as run_id
                metrics['run_id'] = os.path.basename(root)
                
                rows.append(metrics)
                print(f"[OK] Parsed {log_path}")
            except Exception as e:
                print(f"[ERR] Failed to parse {log_path}: {e}")

    if not rows:
        print("[WARN] No results found or benchmark didn't complete (missing ops_sec).")
        return

    df = pd.DataFrame(rows)
    
    # Sort by run_index if present
    if 'run_index' in df.columns:
        df = df.sort_values(['run_index', 'run_id'])
    
    # Ensure columns mentioned in plot scripts are present
    required_cols = ['run_index', 'ops_sec', 'data_read', 'index_read', 'filter_read', 'total_block_read', 'cache_miss']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Identify all Level hit columns to keep them together
    hit_cols = sorted([c for c in df.columns if c.endswith('_hit') and c.startswith('L')])
    
    # Identify stats columns
    stats_cols = sorted([c for c in df.columns if c in [
        'cache_miss', 'cache_hit', 'index_cache_miss', 'index_cache_hit', 
        'filter_cache_miss', 'filter_cache_hit', 'data_cache_miss', 'data_cache_hit',
        'bloom_useful', 'bloom_full_positive', 'bloom_true_positive'
    ]])

    # Build final column list
    base_cols = ['run_index', 'run_id', 'ops_sec', 'micros_op', 'found_ratio']
    block_cols = ['data_read', 'index_read', 'filter_read', 'total_block_read']
    
    remaining = [c for c in df.columns if c not in base_cols + block_cols + hit_cols + stats_cols + ['path', 'found_keys', 'total_ops']]
    
    cols = base_cols + block_cols + stats_cols + hit_cols + remaining + ['path']
    df = df[cols]

    df.to_csv(args.out, index=False)
    print(f"[DONE] Wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
