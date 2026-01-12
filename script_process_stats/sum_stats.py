#!/usr/bin/env python3
"""
Parse RocksDB benchmark logs and aggregate cache operation statistics.

Input:
  --runs-dir <path> : Root directory containing run_* subfolders
  -o, --out-dir <path> : Output directory for results

Output:
  aggregate.csv : All parsed metrics in CSV format
  summary.txt   : Statistical summary of key metrics

Example:
  python parse_cache_ops.py --runs-dir /home/smrc/TTLoad/runs \
    -o /home/smrc/TTLoad/analysis_cache_ops
"""


import argparse
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple, Union

# 1) readrandom | mixgraph 모두 파싱
READ_OR_MIX_RE = re.compile(
    r"""(?mi)
    ^\s*(?:readrandom|mixgraph)\s*:\s*
    (?P<micros>\d+(?:\.\d+)?)\s*micros/op\s+
    (?P<ops>\d+(?:\.\d+)?)\s*ops/sec\s+
    (?P<secs>\d+(?:\.\d+)?)\s*seconds\b
    """,
    re.VERBOSE,
)

# 2) 기타 패턴
CACHE_MISS_RE = re.compile(r"(?mi)^\s*rocksdb\.block\.cache\.miss\s+COUNT\s*:\s*(?P<val>\d+)\s*$")
CACHE_HIT_RE  = re.compile(r"(?mi)^\s*rocksdb\.block\.cache\.hit\s+COUNT\s*:\s*(?P<val>\d+)\s*$")
CACHE_ADD_RE  = re.compile(r"(?mi)^\s*rocksdb\.block\.cache\.add\s+COUNT\s*:\s*(?P<val>\d+)\s*$")

HIT_LEVEL_RE = re.compile(r"(?mi)^\s*Level\s+(?P<lvl>[0-7])\s+hits\s*:\s*(?P<val>\d+)")

SKIPPED_DURING_RE = re.compile(r"(?mi)^\s*Skipped\s+Level\s+during\s+entire\s+benchmark\s*:\s*(?P<val>\d+)\s*$")
SKIPPED_PER_LEVEL_RE = re.compile(r"(?mi)^\s*L(?P<lvl>[0-7])\s*:\s*(?P<val>\d+)\s*$")
SKIPPED_L1_7_TOTAL_RE = re.compile(r"(?mi)^\s*Total\s+skipped\s*\(L1~L7\)\s*:\s*(?P<val>\d+)\s*$")

FILTER_READ_RE = re.compile(r"(?mi)^\s*Total\s+Filter\s+block\s+reads\s*:\s*(?P<val>\d+)\s*$")
INDEX_READ_RE  = re.compile(r"(?mi)^\s*Total\s+Index\s+block\s+reads\s*:\s*(?P<val>\d+)\s*$")
DATA_READ_RE   = re.compile(r"(?mi)^\s*Total\s+Data\s+block\s+reads\s*:\s*(?P<val>\d+)\s*$")

RUN_DIR_NAME_RE = re.compile(r"run_(\d+)_(.+)_(\d{8}_\d{6})", re.IGNORECASE)
CACHE_DIR_RE = re.compile(r"^cache_(?P<label>\d+(?:[KMG]B)?)$", re.IGNORECASE)

Number = Union[int, float]

def _int(m: Optional[re.Match]) -> Optional[int]:
    return int(m.group("val")) if m else None

def _size_label_to_bytes(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    m = re.fullmatch(r"(?i)(\d+)([KMG]?B)", label.strip())
    if not m:
        return None
    num, unit = int(m.group(1)), m.group(2).upper()
    scale = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3}[unit]
    return num * scale

def parse_one_stdout(log_path: Path) -> Optional[Dict[str, Optional[Number]]]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # A) readrandom | mixgraph
    ms = list(READ_OR_MIX_RE.finditer(text))
    if not ms:
        return None
    rr = ms[-1]
    micros = float(rr.group("micros"))
    ops    = float(rr.group("ops"))
    secs   = float(rr.group("secs"))

    # B) cache counters
    miss = _int(CACHE_MISS_RE.search(text))
    hit  = _int(CACHE_HIT_RE.search(text))
    add  = _int(CACHE_ADD_RE.search(text))
    hit_rate = (hit / (hit + miss)) if (hit is not None and miss is not None and (hit + miss) > 0) else None

    out: Dict[str, Optional[Number]] = {
        "micros_per_op": micros,
        "ops_sec": ops,
        "seconds": secs,
        "cache_miss": miss,
        "cache_hit": hit,
        "cache_add": add,
        "cache_hit_rate": hit_rate,
    }

    # C) per-level hits
    hit_levels: Dict[int, int] = {int(m.group("lvl")): int(m.group("val")) for m in HIT_LEVEL_RE.finditer(text)}
    if hit_levels:
        out["hit_level_total"] = sum(hit_levels.values())
    else:
        out["hit_level_total"] = None
    for i in range(8):
        out[f"hit_level_{i}"] = hit_levels.get(i)

    # D) skipped levels
    out["skipped_during_get"] = _int(SKIPPED_DURING_RE.search(text))
    skipped_levels: Dict[int, int] = {int(m.group("lvl")): int(m.group("val")) for m in SKIPPED_PER_LEVEL_RE.finditer(text)}
    out["skipped_total_L0_7"] = sum(skipped_levels.values()) if skipped_levels else None
    for i in range(8):
        out[f"skipped_L{i}"] = skipped_levels.get(i)
    out["skipped_total_L1_7_line"] = _int(SKIPPED_L1_7_TOTAL_RE.search(text))

    # E) block reads
    filter_read = _int(FILTER_READ_RE.search(text))
    index_read  = _int(INDEX_READ_RE.search(text))
    data_read   = _int(DATA_READ_RE.search(text))
    out["filter_read"], out["index_read"], out["data_read"] = filter_read, index_read, data_read
    out["total_block_read"] = (filter_read + index_read + data_read) if None not in (filter_read, index_read, data_read) else None

    return out

def find_logs(runs_dir: Path) -> List[Tuple[Path, Optional[str], Optional[int], Path]]:
    """[(run_dir, cache_size_label, cache_size_bytes, stdout_log)]"""
    results: List[Tuple[Path, Optional[str], Optional[int], Path]] = []
    for run_dir in sorted(runs_dir.glob("run_*")):
        # New layout
        for cache_dir in sorted(run_dir.glob("cache_*")):
            logp = cache_dir / "stdout.log"
            if logp.is_file():
                m = CACHE_DIR_RE.match(cache_dir.name)
                label = m.group("label") if m else None
                results.append((run_dir, label, _size_label_to_bytes(label), logp))
        # Legacy layout
        legacy = run_dir / "04_measure" / "stdout.log"
        if legacy.is_file():
            results.append((run_dir, None, None, legacy))
    return results

def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    if q <= 0: return vs[0]
    if q >= 1: return vs[-1]
    pos = q * (len(vs) - 1)
    lo = int(pos); hi = min(lo + 1, len(vs)-1)
    frac = pos - lo
    return vs[lo] * (1 - frac) + vs[hi] * frac

def _fmt_num(x: Optional[Number]) -> str:
    return "" if x is None else f"{float(x):.6f}"

def _fmt_int(x: Optional[Number]) -> str:
    return "" if x is None else str(int(x))

def write_csv(rows: List[Dict], out_csv: Path) -> None:
    """Legacy function - kept for compatibility"""
    pass

def write_csv_transposed(rows: List[Dict], out_csv: Path) -> None:
    """Write CSV in transposed format: metrics as rows, runs as columns"""
    if not rows:
        return
    
    # Sort rows by run_index and db_name for consistent ordering
    sorted_rows = sorted(rows, key=lambda r: (r.get("run_index") or 0, r.get("db_name") or ""))
    
    # Build header row - use db_name only (remove run_index prefix)
    headers = ["Metric"] + [r.get('db_name', 'unknown') for r in sorted_rows]
    
    rows_data = []
    
    # Performance metrics
    rows_data.append(_build_csv_row("ops_sec", sorted_rows, fmt_float=True))
    rows_data.append(_build_csv_row("seconds", sorted_rows, fmt_float=True))
    rows_data.append(_build_csv_row("micros_per_op", sorted_rows, fmt_float=True))
    
    # Cache metrics
    cache_total = []
    for r in sorted_rows:
        miss = r.get("cache_miss")
        hit = r.get("cache_hit")
        total = (miss + hit) if (miss is not None and hit is not None) else None
        cache_total.append(total)
    rows_data.append(_build_csv_row("cache_total", sorted_rows, values=cache_total, fmt_int=True))
    rows_data.append(_build_csv_row("cache_hit", sorted_rows, fmt_int=True))
    rows_data.append(_build_csv_row("cache_miss", sorted_rows, fmt_int=True))
    rows_data.append(_build_csv_row("hit_rate", sorted_rows, fmt_percent=True))
    
    # Block reads
    rows_data.append(_build_csv_row("index_read", sorted_rows, fmt_int=True))
    rows_data.append(_build_csv_row("data_read", sorted_rows, fmt_int=True))
    rows_data.append(_build_csv_row("filter_read", sorted_rows, fmt_int=True))
    rows_data.append(_build_csv_row("total_block_read", sorted_rows, fmt_int=True))
    
    # Hit level statistics
    for i in range(8):
        key = f"hit_level_{i}"
        vals = [r.get(key) for r in sorted_rows]
        if any(v is not None and v > 0 for v in vals):
            rows_data.append(_build_csv_row(f"hit_key_in_L{i}", sorted_rows, fmt_int=True, field=key))
    
    # Skipped level statistics
    rows_data.append(_build_csv_row("skipped_during_benchmark", sorted_rows, fmt_int=True, field="skipped_during_get"))
    for i in range(8):
        key = f"skipped_L{i}"
        vals = [r.get(key) for r in sorted_rows]
        if any(v is not None and v > 0 for v in vals):
            rows_data.append(_build_csv_row(f"skipped_L{i}", sorted_rows, fmt_int=True, field=key))
    
    # Write to CSV
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for row in rows_data:
            f.write(row + "\n")

def _build_csv_row(label: str, sorted_rows: List[Dict], values: List = None, 
                   fmt_int: bool = False, fmt_float: bool = False, fmt_percent: bool = False, 
                   field: str = None) -> str:
    """Build a CSV row with label and formatted values"""
    if values is None:
        values = [r.get(field or label) for r in sorted_rows]
    
    formatted_vals = []
    for v in values:
        if v is None:
            formatted_vals.append("")
        elif fmt_percent:
            formatted_vals.append(f"{v*100:.2f}%")
        elif fmt_int:
            formatted_vals.append(str(int(v)))
        elif fmt_float:
            formatted_vals.append(f"{float(v):.2f}")
        else:
            formatted_vals.append(str(v))
    
    return label + "," + ",".join(formatted_vals)

def write_summary(rows: List[Dict], out_txt: Path) -> None:
    """Legacy function - kept for compatibility"""
    pass

def main():
    ap = argparse.ArgumentParser(description="Parse RocksDB benchmark logs and generate cache statistics.")
    ap.add_argument("--runs-dir", required=True, help="Root directory containing run_* subfolders")
    ap.add_argument("-o","--out-file", default=None, help="Output CSV file (default: script_dir/<runs-dir-name>.csv)")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    
    # Default output file: same directory as sum_stats.py with runs-dir name
    if args.out_file is None:
        script_dir = Path(__file__).parent
        runs_dir_name = runs_dir.name
        out_file = script_dir / f"{runs_dir_name}.csv"
    else:
        out_file = Path(args.out_file)
    
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pairs = find_logs(runs_dir)
    if not pairs:
        print(f"[ERROR] No logs found under {runs_dir}/run_*/(cache_*|04_measure)/stdout.log")
        return 2

    rows: List[Dict[str, Union[str, Optional[Number]]]] = []
    for run_dir, cache_label, cache_bytes, log_path in pairs:
        parsed = parse_one_stdout(log_path)
        if parsed is None:
            print(f"[WARN] parse failed: {log_path}")
            continue
        m = RUN_DIR_NAME_RE.search(run_dir.name)
        if m:
            run_index = int(m.group(1))
            db_name = m.group(2)
        else:
            run_index = None
            db_name = None
        rows.append({
            "run_dir": str(run_dir),
            "run_index": run_index,
            "db_name": db_name,
            "cache_size_label": cache_label,
            "cache_size_bytes": cache_bytes,
            **parsed,
        })

    if not rows:
        print("[ERROR] Parsed 0 runs.")
        return 3

    write_csv_transposed(rows, out_file)
    print(f"[OK] wrote {out_file}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
