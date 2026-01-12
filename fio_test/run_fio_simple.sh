#!/bin/bash
set -euo pipefail

# Simple FIO Test for Sequential Read/Write Bandwidth
# Measures maximum sequential read and write throughput on /dev/md0 or filesystem

TEST_DIR="${TEST_DIR:-/work}"
TEST_FILE="${TEST_FILE:-$TEST_DIR/fio_test_file}"
RAW_DEVICE="${RAW_DEVICE:-/dev/md0}"
RESULTS_DIR="${RESULTS_DIR:-./results_$(date +%Y%m%d_%H%M%S)}"

# Tuning knobs (override via env)
MODE="${MODE:-fs}"                 # fs or raw
BLOCKSIZE="${BLOCKSIZE:-1M}"
IODEPTH="${IODEPTH:-128}"
NUMJOBS="${NUMJOBS:-300}"
SIZE="${SIZE:-10GB}"              # total IO size per job (fio semantics)
MODE_MIXED="${MODE_MIXED:-0}"      # 1 to run mixed rw test
RWMIXREAD="${RWMIXREAD:-30}"       # read percentage in mixed rw

if [ "$MODE" = "raw" ]; then
    TARGET="$RAW_DEVICE"
else
    TARGET="$TEST_FILE"
fi
SIZE_ARG="--size=$SIZE"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "FIO Sequential Read/Write Bandwidth Test"
echo "=========================================="
echo "Device: $RAW_DEVICE (mounted at $TEST_DIR)"
echo "Mode: $MODE"
echo "Target: $TARGET"
if [ "$MODE" = "raw" ]; then
    echo "WARNING: raw mode writes directly to the block device and destroys data."
fi
echo "Blocksize: $BLOCKSIZE, iodepth: $IODEPTH, numjobs: $NUMJOBS, size: $SIZE"
echo "Mixed rw: $MODE_MIXED (rwmixread=$RWMIXREAD)"
echo "Results saved to: $RESULTS_DIR"
echo ""

# Ensure directory exists in fs mode
if [ "$MODE" = "fs" ]; then
    mkdir -p "$TEST_DIR"
fi

# ------------------------------------------------------------
# [1/2] Sequential WRITE first (precondition + create same file)
# ------------------------------------------------------------
echo "[1/2] Running Sequential Write Test (precondition)..."

fio \
    --name=sequential_write \
    --filename="$TARGET" \
    $SIZE_ARG \
    --blocksize="$BLOCKSIZE" \
    --ioengine=libaio \
    --iodepth="$IODEPTH" \
    --numjobs="$NUMJOBS" \
    --rw=write \
    --direct=1 \
    --group_reporting \
    --output-format=json > "$RESULTS_DIR/seq_write.json" 2>&1

echo ""

# # ------------------------------------------------------------
# # [2/2] Sequential READ on the same target
# # ------------------------------------------------------------
echo "[2/2] Running Sequential Read Test (same target)..."

fio \
    --name=sequential_read \
    --filename="$TARGET" \
    $SIZE_ARG \
    --blocksize="$BLOCKSIZE" \
    --ioengine=libaio \
    --iodepth="$IODEPTH" \
    --numjobs="$NUMJOBS" \
    --rw=read \
    --direct=1 \
    --group_reporting \
    --output-format=json > "$RESULTS_DIR/seq_read.json" 2>&1

echo ""

# ------------------------------------------------------------
# Optional: Mixed RW test (concurrent read+write)
# ------------------------------------------------------------
if [ "$MODE_MIXED" = "1" ]; then
    echo "[optional] Running Mixed RW Test (rw, rwmixread=$RWMIXREAD)..."
    fio \
        --name=mixed_rw \
        --filename="$TARGET" \
        $SIZE_ARG \
        --blocksize="$BLOCKSIZE" \
        --ioengine=libaio \
        --iodepth="$IODEPTH" \
        --numjobs="$NUMJOBS" \
        --rw=rw \
        --rwmixread="$RWMIXREAD" \
        --direct=1 \
        --group_reporting \
        --output-format=json > "$RESULTS_DIR/mixed_rw.json" 2>&1
    echo ""
fi

echo "=========================================="
echo "Results Summary"
echo "=========================================="

echo ""
echo "Sequential Write Bandwidth:"
jq '.jobs[0].write | "  Throughput: \(.bw_mean/1024 | round) MB/s\n  IOPS: \(.iops_mean | round)\n  Latency: \(.lat_ns.mean/1000 | round) μs"' "$RESULTS_DIR/seq_write.json"

echo ""
echo "Sequential Read Bandwidth:"
jq '.jobs[0].read | "  Throughput: \(.bw_mean/1024 | round) MB/s\n  IOPS: \(.iops_mean | round)\n  Latency: \(.lat_ns.mean/1000 | round) μs"' "$RESULTS_DIR/seq_read.json"

if [ "$MODE_MIXED" = "1" ]; then
    echo ""
    echo "Mixed RW Bandwidth:"
    # Mixed는 read+write 둘 다 표시
    jq '.jobs[0] | "  READ  Throughput: \(.read.bw_mean/1024 | round) MB/s  IOPS: \(.read.iops_mean | round)\n  WRITE Throughput: \(.write.bw_mean/1024 | round) MB/s  IOPS: \(.write.iops_mean | round)"' \
       "$RESULTS_DIR/mixed_rw.json"
fi

echo ""
echo "=========================================="
echo "Test completed! Full results in: $RESULTS_DIR"
echo "=========================================="

# Cleanup (fs mode only)
if [ "$MODE" = "fs" ]; then
    rm -f "$TEST_FILE" 2>/dev/null || true
fi
