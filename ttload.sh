#!/usr/bin/env bash
set -euo pipefail

# ===== 사용법 =====
usage() {
  cat <<EOF
Usage: $0 [REQUIRED OPTIONS] [OPTIONAL OPTIONS]

Required Options:
  size=SIZE                  Database size: 100GB, 280GB, 500GB, 700GB, 1TB, etc.
  db_dir=DIR                 Database root directory

Optional Options:
  workload=WORKLOAD          Workload type (default: ttload)
  seed_type=TYPE             Seed type: same (use same seed) or diff (increment seed each iteration)
                             (default: same)
  base_seed=SEED             Base seed (default: 1722070161)
  csv_dir=DIR                Directory for prediction CSVs (default: /home/smrc/TTLoad/ICL)
  perf=true|false            Enable perf profiling and flamegraph (default: false)
  tt_threads=NUM             Internal thread pool size for ttload (default: 48)
  tt_ingest_batch=NUM        SST files per ingest batch for ttload (default: 16)
  plot_tree=true|false       Extract sstables.log and plot LSM tree (default: false)

Output directory is automatically generated as:
  runs/runs_<workload>_<size>_<timestamp>/

Examples:
  $0 size=700GB db_dir=/work/tmp/ttload perf=true tt_threads=48
EOF
  exit 1
}

# ===== 기본 설정 =====
WORKLOAD="ttload"
DB_SIZES=""
DB_ROOT=""
SEED_TYPE="same"
FILL_SEED_BASE="1722070161"
CSV_ROOT="/home/smrc/TTLoad/ICL"
USE_PERF="false"
TT_THREADS="48"
TT_INGEST_BATCH="16"
PLOT_TREE="false"

# Binary used from specified path
DB_BENCH_BIN="/home/smrc/TTLoad/TTLoad/db_bench"
# sstables extraction uses the performance-optimized binary for consistency with sstables.sh
DB_BENCH_SSTABLES_BIN="/home/smrc/TTLoad/rocksdb_perf/db_bench"
FLAMEGRAPH_DIR="${HOME}/FlameGraph"

# ===== 명령줄 인자 파싱 =====
for arg in "$@"; do
  case "$arg" in
    workload=*)
      WORKLOAD="${arg#workload=}"
      ;;
    size=*)
      DB_SIZES="${arg#size=}"
      ;;
    db_dir=*)
      DB_ROOT="${arg#db_dir=}"
      ;;
    seed_type=*)
      SEED_TYPE="${arg#seed_type=}"
      ;;
    base_seed=*)
      FILL_SEED_BASE="${arg#base_seed=}"
      ;;
    csv_dir=*)
      CSV_ROOT="${arg#csv_dir=}"
      ;;
    perf=*)
      USE_PERF="${arg#perf=}"
      ;;
    tt_threads=*)
      TT_THREADS="${arg#tt_threads=}"
      ;;
    tt_ingest_batch=*)
      TT_INGEST_BATCH="${arg#tt_ingest_batch=}"
      ;;
    plot_tree=*)
      PLOT_TREE="${arg#plot_tree=}"
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      ;;
  esac
done

# ===== 필수 옵션 검증 =====
MISSING_OPTS=""
[[ -z "$WORKLOAD" ]] && MISSING_OPTS="$MISSING_OPTS workload"
[[ -z "$DB_SIZES" ]] && MISSING_OPTS="$MISSING_OPTS size"
[[ -z "$DB_ROOT" ]] && MISSING_OPTS="$MISSING_OPTS db_dir"

if [[ "$WORKLOAD" != "fillrandom" && "$WORKLOAD" != "fillseq" && "$WORKLOAD" != "ttload" ]]; then
  echo "Error: workload must be 'fillrandom', 'fillseq', or 'ttload'"
  exit 1
fi

if [[ "$PLOT_TREE" != "true" && "$PLOT_TREE" != "false" ]]; then
  echo "Error: plot_tree must be 'true' or 'false'"
  exit 1
fi

if ! [[ "$TT_INGEST_BATCH" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: tt_ingest_batch must be a positive integer"
  exit 1
fi

# ===== DB_NAME_PREFIX 설정 =====
if [[ "$WORKLOAD" == "fillseq" ]]; then
  DB_NAME_PREFIX="fillseq_1k"
elif [[ "$WORKLOAD" == "fillrandom" ]]; then
  DB_NAME_PREFIX="fillrandom_1k"
else
  DB_NAME_PREFIX="ttload_1k"
fi

# ===== 출력 디렉토리 설정 =====
TS="$(date +%Y%m%d_%H%M%S)"
# Get absolute path for output root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT="${SCRIPT_DIR}/runs"
mkdir -p "$OUT_ROOT"

# ===== 유틸리티 함수 =====
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

drop_caches() {
  log "Dropping page caches (requires sudo)"
  sync
  if command -v sudo >/dev/null 2>&1; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || log "WARN: sudo drop_caches failed"
  else
    sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || log "WARN: drop_caches failed (not root?)"
  fi
}

snap_proc_stats() {
  local outdir="$1" label="$2"
  cat /proc/diskstats > "$outdir/diskstats_${label}.txt" || true
  cat /proc/stat > "$outdir/stat_${label}.txt" || true
}

calc_num_for_size() {
  local size_str="${1^^}"
  if [[ "$size_str" =~ ^([0-9]+)GB$ ]]; then
    echo $(( ${BASH_REMATCH[1]} * 1000000 ))
  elif [[ "$size_str" =~ ^([0-9]+)TB$ ]]; then
    echo $(( ${BASH_REMATCH[1]} * 1000000 * 1000 ))
  else
    echo "ERR"; return 1
  fi
}

calc_stats() {
  local dir="$1"
  local start_sec="$2"
  local end_sec="$3"
  local target_db_dir="$4"

  # 1. Duration
  local duration=$(echo "$end_sec - $start_sec" | bc)

  # 2. CPU Util
  local cpu_before=($(grep '^cpu ' "$dir/stat_BEFORE.txt"))
  local cpu_after=($(grep '^cpu ' "$dir/stat_AFTER.txt"))
  
  local total_before=0
  for i in "${cpu_before[@]:1}"; do total_before=$((total_before + i)); done
  local total_after=0
  for i in "${cpu_after[@]:1}"; do total_after=$((total_after + i)); done

  local diff_total=$((total_after - total_before))
  local diff_idle=$((cpu_after[4] - cpu_before[4]))
  
  local cpu_util="0"
  if [ "$diff_total" -gt 0 ]; then
    cpu_util=$(echo "scale=2; 100 * ($diff_total - $diff_idle) / $diff_total" | bc)
  fi

  # 3. Disk Stats
  local real_db_path=$(readlink -f "$target_db_dir" || echo "$target_db_dir")
  while [[ ! -d "$real_db_path" && "$real_db_path" != "/" ]]; do
    real_db_path=$(dirname "$real_db_path")
  done
  local dev_name=$(df "$real_db_path" | tail -1 | awk '{print $1}')
  local device=$(basename "$dev_name")

  local disk_before=($(grep " $device " "$dir/diskstats_BEFORE.txt" || true))
  local disk_after=($(grep " $device " "$dir/diskstats_AFTER.txt" || true))

  local read_mb="0.00"
  local write_mb="0.00"
  if [[ ${#disk_before[@]} -gt 0 && ${#disk_after[@]} -gt 0 ]]; then
    local s_read=$((disk_after[5] - disk_before[5]))
    local s_write=$((disk_after[9] - disk_before[9]))
    read_mb=$(echo "scale=2; $s_read * 512 / 1024 / 1024" | bc)
    write_mb=$(echo "scale=2; $s_write * 512 / 1024 / 1024" | bc)
  fi

  {
    echo "======= Statistics Summary ======="
    echo "Workload          : $WORKLOAD"
    echo "Duration          : $duration seconds"
    echo "Average CPU Util  : $cpu_util %"
    echo "Disk ($device) Read  : $read_mb MB"
    echo "Disk ($device) Write : $write_mb MB"
    echo "=================================="
  } | tee "$dir/summary.txt"
}

mk_fill_cmd() {
  local db="$1" seed="$2" num="$3"
  local bench_type="$4"
  local size_tag="$5"
  
  local seed_param="--seed=$seed"
  local extra_params=""
  
  if [[ "$bench_type" == "ttload" ]]; then
    extra_params="--threads=1 --tt_threads=$TT_THREADS --tt_ingest_batch=$TT_INGEST_BATCH --max_background_jobs=1 --disable_auto_compactions=1 --csv_path=${CSV_ROOT}/prediction_${size_tag}_gpt5.csv"
  else
    extra_params="--threads=1 --max_background_jobs=64"
  fi

  cat <<CMD
$DB_BENCH_BIN \
  --benchmarks=${bench_type},stats \
  --db="$db" \
  --num=$num \
  $extra_params \
  --key_size=24 \
  --value_size=1000 \
  --compression_type=none \
  --use_direct_reads=1 \
  --use_direct_io_for_flush_and_compaction=1 \
  --target_file_size_base=67108864 \
  --bloom_bits=10 \
  --disable_wal=1 \
  --max_write_buffer_number=32 \
  --memtablerep=vector \
  --write_buffer_size=67108864 \
  $seed_param \
  --allow_concurrent_memtable_write=0
CMD
}

# ===== 메인 루프 =====
log "Starting Loading Workload: $WORKLOAD"
log "DB Sizes: $DB_SIZES"
log "Binary: $DB_BENCH_BIN"
log "Output Root: $OUT_ROOT"

for size in $DB_SIZES; do
  NUM=$(calc_num_for_size "$size")
  if [[ "$NUM" == "ERR" ]]; then
    log "ERROR: Invalid size: $size"; continue
  fi

  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="${OUT_ROOT}/runs_${WORKLOAD}_${size}_${ts}"
  mkdir -p "$run_dir"
  
  DB_PATH="${DB_ROOT}/${DB_NAME_PREFIX}_${size}_${ts}"
  
  # Seed calculation
  fill_seed=$FILL_SEED_BASE

  log ">>> Size: $size | DB: $DB_PATH"
  
  # 1. Pre-run: Drop caches & Stats
  drop_caches
  snap_proc_stats "$run_dir" "BEFORE"
  
  # 2. Run db_bench with background monitoring
  cmd=$(mk_fill_cmd "$DB_PATH" "$fill_seed" "$NUM" "$WORKLOAD" "$size")
  log "Executing: $WORKLOAD"
  
  # Start background monitoring
  # -dx 1 -y: 1s interval, skip boot-time stats
  iostat -dx 1 -y > "$run_dir/iostat.log" 2>&1 &
  IOSTAT_PID=$!
  mpstat -P ALL 1 > "$run_dir/mpstat.log" 2>&1 &
  MPSTAT_PID=$!
  
  # Wait for monitoring tools to stabilize
  sleep 0.2
  
  log "Command to execute:"
  printf '%s\n' "$cmd"

  start_sec=$(date +%s.%N)
  set +e
  
  if [[ "$USE_PERF" == "true" ]]; then
    log "Recording with perf (using sudo)..."
    sudo perf record -F 99 -g -o "$run_dir/perf.data" -- bash -c "$cmd" > "$run_dir/stdout.log" 2>&1
    sudo chown $(id -u):$(id -g) "$run_dir/perf.data"
  else
    bash -c "$cmd" > "$run_dir/stdout.log" 2>&1
  fi
  
  rc=$?
  end_sec=$(date +%s.%N)
  set -e
  
  # Stop background monitoring
  kill $IOSTAT_PID $MPSTAT_PID 2>/dev/null || true
  wait $IOSTAT_PID $MPSTAT_PID 2>/dev/null || true
  
  # Perf post-processing
  if [[ "$USE_PERF" == "true" && -f "$run_dir/perf.data" ]]; then
    log "Generating FlameGraph (including inlined functions)..."
    perf script --inline -i "$run_dir/perf.data" > "$run_dir/perf.script"
    "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" "$run_dir/perf.script" > "$run_dir/perf.folded"
    "${FLAMEGRAPH_DIR}/flamegraph.pl" "$run_dir/perf.folded" > "$run_dir/flamegraph.svg"
  fi

  if [[ $rc -ne 0 ]]; then
    log "WARN: db_bench failed with exit code $rc"
  fi

  # 3. Post-run: Stats & Summary
  snap_proc_stats "$run_dir" "AFTER"
  calc_stats "$run_dir" "$start_sec" "$end_sec" "$DB_PATH"
  
  # NEW: Run Python summary script for detailed analysis
  python3 "${SCRIPT_DIR}/script_ttload/ttload_summary.py" "$run_dir"

  if [[ "$PLOT_TREE" == "true" ]]; then
    # Extract SSTable info and Plot Tree
    log "Extracting SSTable information..."
    $DB_BENCH_SSTABLES_BIN \
      --benchmarks=sstables \
      --use_existing_db=true \
      --db="$DB_PATH" 2>&1 \
      | sed -u -e 's/\r/\n/g' \
               -e '/^[.][.][.] finished [0-9][0-9]* ops[[:space:]]*$/d' \
               -e '/^[[:space:]]*$/d' > "$run_dir/sstables.log"

    log "Plotting LSM Tree structure..."
    (
      cd "$run_dir" && python3 "${SCRIPT_DIR}/script_plot/plot_tree.py" "sstables.log"
    )
  fi
  
  log "✓ $size finished."
done

log "All tasks completed. Results are in $OUT_ROOT"
