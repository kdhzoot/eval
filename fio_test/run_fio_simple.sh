#!/bin/bash
set -euo pipefail

# ==============================================================================
# FIO Performance Test Script
# Measures sequential read/write bandwidth and latency.
# Supports both Filesystem (fs) and Raw Block Device (raw) modes.
# ==============================================================================

# --- Configuration (Override via Environment Variables) ---
TEST_DIR="${TEST_DIR:-/work}"
RAW_DEVICE="${RAW_DEVICE:-/dev/md0}"
RESULTS_DIR="${RESULTS_DIR:-./results_$(date +%Y%m%d_%H%M%S)}"

MODE="${MODE:-fs}"                 # 'fs' (filesystem) or 'raw' (block device)
BLOCKSIZE="${BLOCKSIZE:-1M}"
IODEPTH="${IODEPTH:-128}"
NUMJOBS="${NUMJOBS:-48}"           # Reduced jobs to decrease contention
SIZE="${SIZE:-10GB}"              # Size per job
MODE_MIXED="${MODE_MIXED:-0}"      # 1 to run mixed RW test
RWMIXREAD="${RWMIXREAD:-30}"       # Read percentage for mixed test

# --- Setup & Target Selection ---
mkdir -p "$RESULTS_DIR"

if [ "$MODE" = "raw" ]; then
    # Support multiple raw devices if provided as space-separated string
    # fio uses ':' as a separator for multiple filenames
    FILENAME_LIST=$(echo "$RAW_DEVICE" | tr ' ' ':')
    # For raw mode, we use --filename instead of --directory
    TARGET_ARGS="--filename=$FILENAME_LIST"
    TARGET_DESC="Raw Devices ($RAW_DEVICE)"
    echo "WARNING: Raw mode will overwrite data on $RAW_DEVICE"
else
    # Filesystem mode: Use a sub-directory and multiple files to avoid inode contention
    TARGET_DIR="$TEST_DIR/fio_files"
    mkdir -p "$TARGET_DIR"
    TARGET_ARGS="--directory=$TARGET_DIR --nrfiles=$NUMJOBS --file_service_type=roundrobin"
    TARGET_DESC="Directory ($TARGET_DIR)"
fi

# --- Helper Functions ---

log_header() {
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

run_fio() {
    local job_name="$1"
    local rw_type="$2"
    local extra_args="${3:-}"

    # Use a fixed name for the data layout to ensure file reuse across read/mixed tests
    local layout_name="fio_data_layout"

    echo ">>> Running $job_name ($rw_type)..."
    fio \
        --name="$layout_name" \
        $TARGET_ARGS \
        --size="$SIZE" \
        --blocksize="$BLOCKSIZE" \
        --ioengine=libaio \
        --iodepth="$IODEPTH" \
        --numjobs="$NUMJOBS" \
        --rw="$rw_type" \
        --direct=1 \
        --group_reporting \
        --output-format=json $extra_args > "$RESULTS_DIR/$job_name.json" 2>&1
}

parse_results() {
    local job_name="$1"
    local op_type="$2" # "read" or "write" or "mixed"
    local json_file="$RESULTS_DIR/$job_name.json"

    if [ "$op_type" = "mixed" ]; then
        jq -r '.jobs[0] | "  READ  Throughput: \(.read.bw_mean/1024/1024 | .*100 | round / 100) GB/s  IOPS: \(.read.iops_mean | round)\n  WRITE Throughput: \(.write.bw_mean/1024/1024 | .*100 | round / 100) GB/s  IOPS: \(.write.iops_mean | round)"' "$json_file"
    else
        jq -r ".jobs[0].$op_type | \"  Throughput: \(.bw_mean/1024/1024 | .*100 | round / 100) GB/s\n  IOPS: \(.iops_mean | round)\n  Latency: \(.lat_ns.mean/1000 | round) μs\"" "$json_file"
    fi
}

# --- Execution ---

log_header "FIO Sequential Performance Test"
echo "Mode:     $MODE"
echo "Target:   $TARGET_DESC"
echo "Jobs:     $NUMJOBS (QD=$IODEPTH, BS=$BLOCKSIZE)"
echo "Size:     $SIZE per job"
echo "Results:  $RESULTS_DIR"
echo ""

# 1. Sequential Write
run_fio "seq_write" "write"

# 2. Sequential Read
run_fio "seq_read" "read"

# 3. Mixed RW (Optional)
if [ "$MODE_MIXED" = "1" ]; then
    run_fio "mixed_rw" "rw" "--rwmixread=$RWMIXREAD"
fi

# --- Results Summary ---

log_header "Results Summary"

echo "Sequential Write Bandwidth:"
parse_results "seq_write" "write"
echo ""

echo "Sequential Read Bandwidth:"
parse_results "seq_read" "read"
echo ""

if [ "$MODE_MIXED" = "1" ]; then
    echo "Mixed RW Bandwidth ($RWMIXREAD% Read):"
    parse_results "mixed_rw" "mixed"
    echo ""
fi

log_header "Test Completed Successfully"
echo "Full JSON results available in: $RESULTS_DIR"

# --- Cleanup ---
if [ "$MODE" = "fs" ]; then
    rm -rf "$TARGET_DIR" 2>/dev/null || true
fi

