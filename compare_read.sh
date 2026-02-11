#!/usr/bin/env bash
set -euo pipefail

# ===== USAGE =====
usage() {
  cat <<EOF
Usage: $0 ttload_db=PATH fillrandom_db=PATH [options]

Required:
  ttload_db=PATH             Path to the TTLoad-generated database
  fillrandom_db=PATH         Path to the FillRandom-generated database

Options:
  workload=WORKLOAD          Workload type: readrandom (default), mixgraph
  reads=NUM                  Number of reads (default: 1000000)
  cache_sizes="SIZE1 SIZE2"  Cache sizes to test (default: "1")
                             e.g., "1 512MiB 1GiB"
  out_dir=DIR                Output base directory (default: auto-generated)

Example:
  $0 ttload_db=/work/tmp/ttload_1k_700GB_... fillrandom_db=/work/tmp/fillrandom_1k_700GB_...
EOF
  exit 1
}

# ===== DEFAULTS =====
TTLOAD_DB=""
FILLRANDOM_DB=""
WORKLOAD="readrandom"
READS="10000"
CACHE_SIZES="1"
OUT_ROOT=""
DB_BENCH_BIN="/home/smrc/TTLoad/rocksdb_perf/db_bench"

# ===== PARSE ARGS =====
for arg in "$@"; do
  case "$arg" in
    ttload_db=*)   TTLOAD_DB="${arg#ttload_db=}" ;;
    fillrandom_db=*) FILLRANDOM_DB="${arg#fillrandom_db=}" ;;
    workload=*)    WORKLOAD="${arg#workload=}" ;;
    reads=*)       READS="${arg#reads=}" ;;
    cache_sizes=*) CACHE_SIZES="${arg#cache_sizes=}" ;;
    out_dir=*)     OUT_ROOT="${arg#out_dir=}" ;;
    -h|--help)     usage ;;
    *) echo "Unknown: $arg"; usage ;;
  esac
done

if [[ -z "$TTLOAD_DB" || -z "$FILLRANDOM_DB" ]]; then
  echo "Error: ttload_db and fillrandom_db are required."
  usage
fi

# ===== PREPARE =====
TS="$(date +%Y%m%d_%H%M%S)"
[[ -z "$OUT_ROOT" ]] && OUT_ROOT="/home/smrc/TTLoad/eval/runs/compare_read_${TS}"
mkdir -p "$OUT_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Detect NUM from DB path
calculate_num() {
  local db_path="$1"
  local db_name="$(basename "$db_path")"
  if [[ "$db_name" =~ _([0-9]+)(GB|TB)_ ]]; then
    local size_val="${BASH_REMATCH[1]}"
    local size_unit="${BASH_REMATCH[2]}"
    if [[ "$size_unit" == "GB" ]]; then
      echo $((size_val * 1000000))
    elif [[ "$size_unit" == "TB" ]]; then
      echo $((size_val * 1000000 * 1000))
    fi
  else
    echo "280000000" # fallback
  fi
}

drop_caches() {
  sync
  if [[ -w /proc/sys/vm/drop_caches ]]; then
    echo 3 > /proc/sys/vm/drop_caches
  else
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || log "WARN: Failed to drop caches"
  fi
}

parse_bytes() {
  local s; s="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$s" =~ ^[0-9]+$ ]]; then echo "$s"; return 0; fi
  local num unit
  num="$(echo "$s" | sed -E 's/^([0-9]+).*/\1/')"
  unit="$(echo "$s" | sed -E 's/^[0-9]+(.*)$/\1/')"
  case "$unit" in
    b|"") echo "$num";;
    k|kb)  echo $(( num * 1000 ));;
    m|mb)  echo $(( num * 1000 * 1000 ));;
    g|gb)  echo $(( num * 1000 * 1000 * 1000 ));;
    kib)   echo $(( num * 1024 ));;
    mib)   echo $(( num * 1024 * 1024 ));;
    gib)   echo $(( num * 1024 * 1024 * 1024 ));;
    *)     echo "$1" ;; # pass through
  esac
}

# ===== BENCHMARK FUNCTION =====
run_bench() {
  local type="$1" # ttload or fillrandom
  local db_path="$2"
  local cache_bytes="$3"
  local cache_label="$4"
  local num="$5"
  
  local run_dir="${OUT_ROOT}/${type}/cache_${cache_label}"
  mkdir -p "$run_dir"
  
  log "Run $type | DB: $(basename "$db_path") | Cache: $cache_label"
  drop_caches
  
  # Command construction (same as read.sh)
  # Simplified to focus on comparison
  $DB_BENCH_BIN \
    --benchmarks="${WORKLOAD},stats" \
    --use_existing_db=true \
    --db="$db_path" \
    --num=$num \
    --reads=$READS \
    --key_size=24 \
    --value_size=1000 \
    --compression_type=none \
    --use_direct_io_for_flush_and_compaction=true \
    --use_direct_reads=true \
    --seed=1724 \
    --bloom_bits=10 \
    --statistics=1 \
    --cache_index_and_filter_blocks=true \
    --perf_level=2 \
    --cache_size=$cache_bytes \
    > "$run_dir/stdout.log" 2>&1
}

# ===== MAIN =====
NUM_TTLOAD=$(calculate_num "$TTLOAD_DB")
NUM_FILLRANDOM=$(calculate_num "$FILLRANDOM_DB")

log "Output directory: $OUT_ROOT"
log "TTLoad DB: $TTLOAD_DB (NUM=$NUM_TTLOAD)"
log "FillRandom DB: $FILLRANDOM_DB (NUM=$NUM_FILLRANDOM)"

for cache in $CACHE_SIZES; do
  BYTES=$(parse_bytes "$cache")
  run_bench "ttload" "$TTLOAD_DB" "$BYTES" "$cache" "$NUM_TTLOAD"
  run_bench "fillrandom" "$FILLRANDOM_DB" "$BYTES" "$cache" "$NUM_FILLRANDOM"
done

# ===== SUMMARY =====
log "Generating detailed comparison summary..."

for cache in $CACHE_SIZES; do
  echo ""
  echo "========================================================================================================================"
  echo " CACHE SIZE: $cache"
  echo "========================================================================================================================"

  # 1. Basic Performance & Found Keys
  echo "[1. Basic Performance & Results]"
  printf "%-12s | %-12s | %-12s | %-12s\n" "DB Type" "Throughput" "Latency" "Found Keys"
  printf "%-12s | %-12s | %-12s | %-12s\n" "" "(ops/sec)" "(us/op)" "(total)"
  echo "-------------+--------------+--------------+--------------"
  for type in ttload fillrandom; do
    log_file="${OUT_ROOT}/${type}/cache_${cache}/stdout.log"
    if [[ -f "$log_file" ]]; then
       perf=$(grep "$WORKLOAD" "$log_file" | head -n 1 || true)
       latency=$(echo "$perf" | awk '{print $3}' || echo "N/A")
       ops_sec=$(echo "$perf" | awk '{print $5}' || echo "N/A")
       found=$(grep "Total keys found:" "$log_file" | awk -F': ' '{print $2}' | xargs || echo "N/A")
       printf "%-12s | %-12s | %-12s | %-12s\n" "$type" "$ops_sec" "$latency" "$found"
    fi
  done

  # 2. Block Read Statistics (Totals)
  echo ""
  echo "[2. Total Block Read Statistics]"
  printf "%-12s | %-12s | %-12s | %-12s | %-12s\n" "DB Type" "Filter" "Index" "Data" "Total All"
  echo "-------------+--------------+--------------+--------------+--------------"
  for type in ttload fillrandom; do
    log_file="${OUT_ROOT}/${type}/cache_${cache}/stdout.log"
    if [[ -f "$log_file" ]]; then
       f_tot=$(grep "Total Filter block reads:" "$log_file" | awk -F': ' '{print $2}' | xargs || echo "0")
       i_tot=$(grep "Total Index block reads:" "$log_file" | awk -F': ' '{print $2}' | xargs || echo "0")
       d_tot=$(grep "Total Data block reads:" "$log_file" | awk -F': ' '{print $2}' | xargs || echo "0")
       a_tot=$(grep "Total All block reads:" "$log_file" | awk -F': ' '{print $2}' | xargs || echo "0")
       printf "%-12s | %-12s | %-12s | %-12s | %-12s\n" "$type" "$f_tot" "$i_tot" "$d_tot" "$a_tot"
    fi
  done

  # 3. Skip-Level Statistics (L1~L6)
  echo ""
  echo "[3. Skip-Level Statistics (Files skipped)]"
  printf "%-12s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s | %-8s\n" "DB Type" "L1" "L2" "L3" "L4" "L5" "L6" "Total"
  echo "-------------+--------+--------+--------+--------+--------+--------+----------"
  for type in ttload fillrandom; do
    log_file="${OUT_ROOT}/${type}/cache_${cache}/stdout.log"
    if [[ -f "$log_file" ]]; then
       l1_s=$(sed -n '/Skipped files per level:/,/Total skipped/p' "$log_file" | grep "L1:" | awk '{print $2}' || echo "0")
       l2_s=$(sed -n '/Skipped files per level:/,/Total skipped/p' "$log_file" | grep "L2:" | awk '{print $2}' || echo "0")
       l3_s=$(sed -n '/Skipped files per level:/,/Total skipped/p' "$log_file" | grep "L3:" | awk '{print $2}' || echo "0")
       l4_s=$(sed -n '/Skipped files per level:/,/Total skipped/p' "$log_file" | grep "L4:" | awk '{print $2}' || echo "0")
       l5_s=$(sed -n '/Skipped files per level:/,/Total skipped/p' "$log_file" | grep "L5:" | awk '{print $2}' || echo "0")
       l6_s=$(sed -n '/Skipped files per level:/,/Total skipped/p' "$log_file" | grep "L6:" | awk '{print $2}' || echo "0")
       t_s=$(grep "Total skipped (L1~L7):" "$log_file" | awk -F': ' '{print $2}' | xargs || echo "0")
       printf "%-12s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s | %-8s\n" "$type" "$l1_s" "$l2_s" "$l3_s" "$l4_s" "$l5_s" "$l6_s" "$t_s"
    fi
  done

  # 4. Level Hit Statistics (L1~L6)
  echo ""
  echo "[4. Level Hit Statistics (Keys found)]"
  printf "%-12s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s\n" "DB Type" "L1" "L2" "L3" "L4" "L5" "L6"
  echo "-------------+--------+--------+--------+--------+--------+--------"
  for type in ttload fillrandom; do
    log_file="${OUT_ROOT}/${type}/cache_${cache}/stdout.log"
    if [[ -f "$log_file" ]]; then
       l1_h=$(grep "Level 1 hits:" "$log_file" | awk '{print $4}' || echo "0")
       l2_h=$(grep "Level 2 hits:" "$log_file" | awk '{print $4}' || echo "0")
       l3_h=$(grep "Level 3 hits:" "$log_file" | awk '{print $4}' || echo "0")
       l4_h=$(grep "Level 4 hits:" "$log_file" | awk '{print $4}' || echo "0")
       l5_h=$(grep "Level 5 hits:" "$log_file" | awk '{print $4}' || echo "0")
       l6_h=$(grep "Level 6 hits:" "$log_file" | awk '{print $4}' || echo "0")
       printf "%-12s | %-6s | %-6s | %-6s | %-6s | %-6s | %-6s\n" "$type" "$l1_h" "$l2_h" "$l3_h" "$l4_h" "$l5_h" "$l6_h"
    fi
  done
done

log "All benchmarks completed. Detailed summary above. Results at $OUT_ROOT"
