#!/usr/bin/env python3
"""Summarize fillrandom loading stats across run_* folders.

Scans a runs directory like:
  runs_fillrandom_.../run_100GB_001_.../fillrandom/

For each run, extracts:
- Load time from RocksDB db_bench output (fillrandom line)
- CPU usage deltas from /proc/stat snapshots (stat_BEFORE.txt / stat_AFTER.txt)
- Disk I/O deltas from /proc/diskstats snapshots (diskstats_BEFORE.txt / diskstats_AFTER.txt)

Outputs one CSV row per run.

Example:
  python summarize_load_stats.py \
    --runs-dir /home/smrc/TTLoad/eval/runs_fillrandom_100to500GB_10iter_same_20251226_101831

Optional:
  --disk-device md0
  --disk-device-regex '^(md0|dm-0)$'
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RUN_NAME_RE = re.compile(r"^run_(?P<size_gb>\d+)GB_(?P<iter>\d+)_?(?P<ts>\d{8}_\d{6})?$", re.IGNORECASE)

DB_PATH_RE = re.compile(r"(?mi)^\s*DB\s+path:\s*\[(?P<path>[^\]]+)\]\s*$")

FILLRANDOM_RE = re.compile(
    r"""(?mi)
    ^\s*fillrandom\s*:\s*
    (?P<micros>\d+(?:\.\d+)?)\s*micros/op\s+
    (?P<ops>\d+(?:\.\d+)?)\s*ops/sec\s+
    (?P<secs>\d+(?:\.\d+)?)\s*seconds\b
    (?:\s+(?P<operations>\d+)\s*operations)?
    (?:\s*;\s*(?P<mbps>\d+(?:\.\d+)?)\s*MB/s)?
    """,
    re.VERBOSE,
)

SYSSTATS_BEFORE_RE = re.compile(r"^===\s+/proc/(?:diskstats|stat)\s+BEFORE\s+(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})\s+===\s*$", re.M)
SYSSTATS_AFTER_RE = re.compile(r"^===\s+/proc/(?:diskstats|stat)\s+AFTER\s+(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})\s+===\s*$", re.M)


@dataclass(frozen=True)
class CpuTotals:
    user: int
    nice: int
    system: int
    idle: int
    iowait: int
    irq: int
    softirq: int
    steal: int

    @staticmethod
    def from_proc_stat_text(text: str) -> Optional["CpuTotals"]:
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("cpu "):
                continue
            parts = line.split()
            nums = []
            for x in parts[1:]:
                try:
                    nums.append(int(x))
                except ValueError:
                    break
            if len(nums) < 8:
                return None
            return CpuTotals(*nums[:8])
        return None


@dataclass(frozen=True)
class DiskStats:
    reads_completed: int
    reads_merged: int
    sectors_read: int
    ms_reading: int
    writes_completed: int
    writes_merged: int
    sectors_written: int
    ms_writing: int
    ios_in_progress: int
    ms_doing_ios: int
    ms_weighted: int
    discards_completed: Optional[int] = None
    discards_merged: Optional[int] = None
    sectors_discarded: Optional[int] = None
    ms_discarding: Optional[int] = None

    @staticmethod
    def from_values(values: List[int]) -> Optional["DiskStats"]:
        if len(values) < 11:
            return None
        base = values[:11]
        tail = values[11:15] if len(values) >= 15 else None
        if tail and len(tail) == 4:
            return DiskStats(*base, *tail)
        return DiskStats(*base)


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def parse_fillrandom_stdout(stdout_text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "load_seconds": None,
        "micros_per_op": None,
        "ops_sec": None,
        "mb_per_sec": None,
        "operations": None,
    }

    mdb = DB_PATH_RE.search(stdout_text)
    if mdb:
        out["db_path"] = mdb.group("path")
        out["db_name"] = Path(mdb.group("path")).name
    else:
        out["db_path"] = None
        out["db_name"] = None

    ms = list(FILLRANDOM_RE.finditer(stdout_text))
    if not ms:
        return out

    m = ms[-1]
    out["micros_per_op"] = float(m.group("micros"))
    out["ops_sec"] = float(m.group("ops"))
    out["load_seconds"] = float(m.group("secs"))
    out["operations"] = float(m.group("operations")) if m.group("operations") else None
    out["mb_per_sec"] = float(m.group("mbps")) if m.group("mbps") else None
    return out


def parse_sysstats_interval_seconds(sysstats_text: str) -> Optional[float]:
    if not sysstats_text:
        return None
    mb = SYSSTATS_BEFORE_RE.search(sysstats_text)
    ma = SYSSTATS_AFTER_RE.search(sysstats_text)
    if not mb or not ma:
        return None
    try:
        tb = datetime.strptime(mb.group("ts"), "%Y-%m-%dT%H:%M:%S%z")
        ta = datetime.strptime(ma.group("ts"), "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None
    return max(0.0, (ta - tb).total_seconds())


def parse_diskstats_text(text: str) -> Dict[str, DiskStats]:
    out: Dict[str, DiskStats] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        # major minor name ...
        name = parts[2]
        try:
            values = [int(x) for x in parts[3:]]
        except ValueError:
            continue
        ds = DiskStats.from_values(values)
        if ds is not None:
            out[name] = ds
    return out


def cpu_delta(before: CpuTotals, after: CpuTotals) -> CpuTotals:
    return CpuTotals(
        user=after.user - before.user,
        nice=after.nice - before.nice,
        system=after.system - before.system,
        idle=after.idle - before.idle,
        iowait=after.iowait - before.iowait,
        irq=after.irq - before.irq,
        softirq=after.softirq - before.softirq,
        steal=after.steal - before.steal,
    )


def cpu_pct(delta: CpuTotals) -> Dict[str, Optional[float]]:
    total = delta.user + delta.nice + delta.system + delta.idle + delta.iowait + delta.irq + delta.softirq + delta.steal
    if total <= 0:
        return {
            "cpu_total_ticks": float(total),
            "cpu_util_pct": None,
            "cpu_user_pct": None,
            "cpu_system_pct": None,
            "cpu_iowait_pct": None,
            "cpu_idle_pct": None,
        }

    busy = total - delta.idle - delta.iowait
    user = delta.user + delta.nice
    return {
        "cpu_total_ticks": float(total),
        "cpu_util_pct": busy / total * 100.0,
        "cpu_user_pct": user / total * 100.0,
        "cpu_system_pct": delta.system / total * 100.0,
        "cpu_iowait_pct": delta.iowait / total * 100.0,
        "cpu_idle_pct": delta.idle / total * 100.0,
    }


def _is_partition_device(name: str) -> bool:
    # nvme0n1p1 etc
    return bool(re.search(r"p\d+$", name))


def _is_loop_device(name: str) -> bool:
    return bool(re.fullmatch(r"loop\d+", name))


def choose_disk_device(
    before: Dict[str, DiskStats],
    after: Dict[str, DiskStats],
    preferred: Optional[str],
    device_regex: Optional[str],
) -> Optional[str]:
    common = set(before.keys()) & set(after.keys())
    if not common:
        return None

    if preferred:
        return preferred if preferred in common else None

    candidates = sorted(common)

    if device_regex:
        rx = re.compile(device_regex)
        candidates = [d for d in candidates if rx.search(d)]
        if not candidates:
            return None

    # Prefer md0 / dm-0 if present (very common in these logs)
    for preferred_name in ("md0", "dm-0"):
        if preferred_name in candidates:
            return preferred_name

    # Otherwise pick the device with the most (read+write) sectors, excluding loops and partitions.
    def score(dev: str) -> int:
        if _is_loop_device(dev) or _is_partition_device(dev):
            return -1
        b = before.get(dev)
        a = after.get(dev)
        if not b or not a:
            return -1
        return (a.sectors_read - b.sectors_read) + (a.sectors_written - b.sectors_written)

    best = max(candidates, key=score)
    return best if score(best) >= 0 else None


def disk_delta(before: DiskStats, after: DiskStats) -> Dict[str, float]:
    d_reads = after.reads_completed - before.reads_completed
    d_writes = after.writes_completed - before.writes_completed
    d_sectors_r = after.sectors_read - before.sectors_read
    d_sectors_w = after.sectors_written - before.sectors_written
    d_ms_doing = after.ms_doing_ios - before.ms_doing_ios

    bytes_r = d_sectors_r * 512
    bytes_w = d_sectors_w * 512

    return {
        "disk_read_iops": float(d_reads),
        "disk_write_iops": float(d_writes),
        "disk_read_mb": float(bytes_r) / 1024.0 / 1024.0,
        "disk_write_mb": float(bytes_w) / 1024.0 / 1024.0,
        "disk_ms_doing_ios": float(d_ms_doing),
    }


def iter_run_dirs(runs_dir: Path) -> Iterable[Path]:
    for p in sorted(runs_dir.glob("run_*")):
        if p.is_dir():
            yield p


def parse_run_dir(run_dir: Path, disk_device: Optional[str], disk_device_regex: Optional[str]) -> Optional[Dict[str, object]]:
    fill_dir = run_dir / "fillrandom"
    if not fill_dir.is_dir():
        return None

    stdout_path = fill_dir / "stdout.log"
    stat_before_path = fill_dir / "stat_BEFORE.txt"
    stat_after_path = fill_dir / "stat_AFTER.txt"
    disk_before_path = fill_dir / "diskstats_BEFORE.txt"
    disk_after_path = fill_dir / "diskstats_AFTER.txt"
    sysstats_path = fill_dir / "sysstats.log"

    stdout_text = _read_text(stdout_path) or ""
    stat_before_text = _read_text(stat_before_path) or ""
    stat_after_text = _read_text(stat_after_path) or ""
    disk_before_text = _read_text(disk_before_path) or ""
    disk_after_text = _read_text(disk_after_path) or ""
    sysstats_text = _read_text(sysstats_path) or ""

    fill = parse_fillrandom_stdout(stdout_text)

    cb = CpuTotals.from_proc_stat_text(stat_before_text)
    ca = CpuTotals.from_proc_stat_text(stat_after_text)
    cpu_metrics: Dict[str, Optional[float]] = {
        "cpu_total_ticks": None,
        "cpu_util_pct": None,
        "cpu_user_pct": None,
        "cpu_system_pct": None,
        "cpu_iowait_pct": None,
        "cpu_idle_pct": None,
    }
    if cb and ca:
        cpu_metrics = cpu_pct(cpu_delta(cb, ca))

    interval_sec = parse_sysstats_interval_seconds(sysstats_text)
    if interval_sec is None:
        interval_sec = fill.get("load_seconds")

    disk_metrics: Dict[str, Optional[float]] = {
        "disk_read_iops": None,
        "disk_write_iops": None,
        "disk_read_mb": None,
        "disk_write_mb": None,
        "disk_read_mbps": None,
        "disk_write_mbps": None,
        "disk_busy_pct": None,
        "disk_ms_doing_ios": None,
    }

    before_disks = parse_diskstats_text(disk_before_text)
    after_disks = parse_diskstats_text(disk_after_text)

    chosen_dev = choose_disk_device(before_disks, after_disks, disk_device, disk_device_regex)
    if chosen_dev and chosen_dev in before_disks and chosen_dev in after_disks:
        dd = disk_delta(before_disks[chosen_dev], after_disks[chosen_dev])
        disk_metrics.update(dd)
        if interval_sec and interval_sec > 0:
            disk_metrics["disk_read_mbps"] = float(dd["disk_read_mb"]) / float(interval_sec)
            disk_metrics["disk_write_mbps"] = float(dd["disk_write_mb"]) / float(interval_sec)
            # Some virtual devices (e.g., md0) may report ms_doing_ios=0 even under load.
            # In that case, leave disk_busy_pct empty rather than reporting a misleading 0.0.
            if float(dd["disk_ms_doing_ios"]) > 0:
                disk_metrics["disk_busy_pct"] = float(dd["disk_ms_doing_ios"]) / (float(interval_sec) * 1000.0) * 100.0
            else:
                disk_metrics["disk_busy_pct"] = None

    mrun = RUN_NAME_RE.match(run_dir.name)
    size_gb = int(mrun.group("size_gb")) if mrun and mrun.group("size_gb") else None
    iterno = int(mrun.group("iter")) if mrun and mrun.group("iter") else None

    row: Dict[str, object] = {
        "run_dir": run_dir.name,
        "size_gb": size_gb,
        "iter": iterno,
        "db_name": fill.get("db_name"),
        "db_path": fill.get("db_path"),
        "load_seconds": fill.get("load_seconds"),
        "ops_sec": fill.get("ops_sec"),
        "mb_per_sec": fill.get("mb_per_sec"),
        "micros_per_op": fill.get("micros_per_op"),
        "operations": fill.get("operations"),
        "interval_seconds": interval_sec,
        **cpu_metrics,
        "disk_device": chosen_dev,
        **disk_metrics,
    }

    return row


def write_csv_rows(rows: List[Dict[str, object]], out_csv: Path) -> None:
    if not rows:
        return

    # Stable column ordering
    fieldnames = [
        "run_dir",
        "size_gb",
        "iter",
        "db_name",
        "db_path",
        "load_seconds",
        "ops_sec",
        "mb_per_sec",
        "micros_per_op",
        "operations",
        "interval_seconds",
        "cpu_total_ticks",
        "cpu_util_pct",
        "cpu_user_pct",
        "cpu_system_pct",
        "cpu_iowait_pct",
        "cpu_idle_pct",
        "disk_device",
        "disk_read_iops",
        "disk_write_iops",
        "disk_read_mb",
        "disk_write_mb",
        "disk_read_mbps",
        "disk_write_mbps",
        "disk_busy_pct",
        "disk_ms_doing_ios",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize fillrandom loading time + cpu/disk stats into a CSV")
    ap.add_argument("--runs-dir", required=True, help="Root directory containing run_* subfolders")
    ap.add_argument(
        "-o",
        "--out-file",
        default=None,
        help="Output CSV file (default: script_dir/<runs-dir-name>_load_stats.csv)",
    )
    ap.add_argument(
        "--disk-device",
        default=None,
        help="Exact disk device name to summarize (e.g., md0, dm-0, nvme1n1). If unset, auto-select.",
    )
    ap.add_argument(
        "--disk-device-regex",
        default=None,
        help="Regex to filter disk devices before auto-select (e.g., '^(md0|dm-0)$').",
    )

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.out_file is None:
        script_dir = Path(__file__).parent
        out_csv = script_dir / f"{runs_dir.name}_load_stats.csv"
    else:
        out_csv = Path(args.out_file)

    rows: List[Dict[str, object]] = []
    for run_dir in iter_run_dirs(runs_dir):
        row = parse_run_dir(run_dir, args.disk_device, args.disk_device_regex)
        if row is not None:
            rows.append(row)

    # Sort rows for stability
    rows.sort(key=lambda r: (r.get("size_gb") or 0, r.get("iter") or 0, str(r.get("run_dir") or "")))

    write_csv_rows(rows, out_csv)
    print(f"Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
