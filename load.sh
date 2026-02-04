#!/usr/bin/env bash
set -euo pipefail

# ===== 사용법 =====
usage() {
  cat <<EOF
Usage: $0 [REQUIRED OPTIONS] [OPTIONAL OPTIONS]

Required Options:
  workload=WORKLOAD          Workload type: fillrandom, fillseq
  size=SIZE                  Database size: 100GB, 280GB, 500GB, 1TB, etc.
  iter=NUM                   Number of iterations
  db_dir=DIR                 Database directory
  seed_type=TYPE             Seed type: same (use same seed) or diff (increment seed each iteration)
                             (only applicable for fillrandom workload)

Optional Options:
  base_seed=SEED             Base seed for fillrandom (default: 1722070161)

Output directory is automatically generated as:
  runs_<workload>_<size>_<iter>iter_<seed_type>_<timestamp>/

Examples:
  $0 workload=fillrandom size=800GB iter=3 db_dir=/path/to/db seed_type=same
  $0 workload=fillseq size=280GB iter=5 db_dir=/db seed_type=same
  $0 workload=fillrandom size=280GB iter=5 db_dir=/db seed_type=diff base_seed=1234567890

EOF
  exit 1
}

# ===== 기본 설정 (선택적 옵션만 기본값 설정) =====
WORKLOAD=""
DB_SIZES=""
ITER=""
DB_ROOT=""
SEED_TYPE=""
FILL_SEED_BASE="1722070161"

FILL_DB_BENCH_BIN="${FILL_DB_BENCH_BIN:-/home/smrc/TTLoad/rocksdb_clean/db_bench}"
# Binary used for readrandom warmup (can be same binary)
READ_DB_BENCH_BIN="${READ_DB_BENCH_BIN:-/home/smrc/TTLoad/rocksdb_clean/db_bench}"
# Binary used specifically for sstables benchmark (default to S3-LOAD/db_bench if present)
SST_DB_BENCH_BIN="${SST_DB_BENCH_BIN:-/home/smrc/TTLoad/rocksdb_perf/db_bench}"

READS_FIXED="10000000"
WARMUP_DURATION="300"

# ===== 명령줄 인자 파싱 =====
for arg in "$@"; do
  case "$arg" in
    workload=*)
      WORKLOAD="${arg#workload=}"
      ;;
    size=*)
      DB_SIZES="${arg#size=}"
      ;;
    iter=*)
      ITER="${arg#iter=}"
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
[[ -z "$ITER" ]] && MISSING_OPTS="$MISSING_OPTS iter"
[[ -z "$DB_ROOT" ]] && MISSING_OPTS="$MISSING_OPTS db_dir"
[[ -z "$SEED_TYPE" ]] && MISSING_OPTS="$MISSING_OPTS seed_type"

if [[ -n "$MISSING_OPTS" ]]; then
  echo "Error: Missing required options:$MISSING_OPTS"
  echo
  usage
fi

# ===== 워크로드 & Seed Type 검증 =====
if [[ "$WORKLOAD" != "fillrandom" && "$WORKLOAD" != "fillseq" ]]; then
  echo "Error: workload must be 'fillrandom' or 'fillseq'"
  exit 1
fi

if [[ "$SEED_TYPE" != "same" && "$SEED_TYPE" != "diff" ]]; then
  echo "Error: seed_type must be 'same' or 'diff'"
  exit 1
fi

# fillseq는 seed_type이 의미 없지만 필수 인자이므로 허용
if [[ "$WORKLOAD" == "fillseq" ]]; then
  echo "NOTE: fillseq workload ignores seed_type (sequential keys don't use seeds)"
fi

# ===== DB_NAME_PREFIX 설정 =====
if [[ "$WORKLOAD" == "fillseq" ]]; then
  DB_NAME_PREFIX="fillseq_1k"
else
  DB_NAME_PREFIX="fillrandom_1k"
fi

# ===== 출력 디렉토리 자동 생성 =====
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="/home/smrc/TTLoad/eval/runs/runs_${WORKLOAD}_${DB_SIZES}_${ITER}iter_${SEED_TYPE}_${TS}"

# ===== 유틸 =====
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

# /proc/diskstats, /proc/stat 스냅샷 저장
# 사용법: snap_proc_stats <out_dir> <LABEL>
snap_proc_stats() {
  local outdir="$1" label="$2" ts
  ts="$(date +%Y-%m-%dT%H:%M:%S%z)"

  mkdir -p "$outdir"

  {
    echo "=== /proc/diskstats ${label} ${ts} ==="
    cat /proc/diskstats || true
    echo
    echo "=== /proc/stat ${label} ${ts} ==="
    cat /proc/stat || true
    echo
  } >>"$outdir/sysstats.log"

  # 개별 파일(덮어쓰기)
  cat /proc/diskstats >"$outdir/diskstats_${label}.txt"  || true
  cat /proc/stat      >"$outdir/stat_${label}.txt"       || true
}

# GB/TB 문자열을 num으로 변환 (1KV = 1KB 가정)
calc_num_for_size() {
  local size_str="${1^^}"   # 대문자로 변환
  if [[ "$size_str" =~ ^([0-9]+)GB$ ]]; then
    echo $(( ${BASH_REMATCH[1]} * 1000000 ))
  elif [[ "$size_str" =~ ^([0-9]+)TB$ ]]; then
    echo $(( ${BASH_REMATCH[1]} * 1000000 * 1000 ))   # 1TB = 1000GB 가정
  else
    echo "ERR"; return 1
  fi
}

# 단계 실행: 원본 로그 저장 + 핵심 요약 생성 + /proc 스냅샷(BEFORE/AFTER)
run_and_save() {
  # $1 = step dir, $2 = name, $3 = command string, $4 = do_snapshot (yes/no, default yes)
  local dir="$1" name="$2" cmd="$3" do_snap="${4:-yes}"
  mkdir -p "$dir"

  log "RUN [$name]"

  if [[ "$do_snap" == "yes" ]]; then
    # ==== BEFORE: drop caches + snapshot ====
    log "Dropping page caches before snapshot (may require root)"
    drop_caches || log "WARN: drop_caches failed or permission denied"
    log "Snapshotting /proc stats BEFORE → $dir"
    snap_proc_stats "$dir" "BEFORE"
  fi

  set +e
  bash -lc "$cmd" 2>&1 \
    | sed -u -e 's/\r/\n/g' \
             -e '/^\.\.\. finished [0-9]\+ ops *$/d' \
             -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  local rc=$?
  set -e

  if [[ "$do_snap" == "yes" ]]; then
    # ==== AFTER snapshot ====
    log "Snapshotting /proc stats AFTER → $dir"
    snap_proc_stats "$dir" "AFTER"
  fi

  if [[ $rc -ne 0 ]]; then
    log "WARN [$name]: exit code $rc (stdout.log 확인)"
  fi
}

# Drop page caches helper. Tries with sudo first, then without.
drop_caches() {
  # keep in a subshell to avoid altering the caller's errexit state
  (
    set +e
    sync
    if command -v sudo >/dev/null 2>&1; then
      sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1
      rc=$?
      if [[ $rc -eq 0 ]]; then
        exit 0
      fi
    fi
    # fallback: try without sudo (will fail if not root)
    sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1
    exit $?
  )
}

# ===== 커맨드 빌더들 =====
# fillrandom (db, seed, num)
mk_fillrandom_cmd() {
  local db="$1" seed="$2" num="$3"
  cat <<CMD
$FILL_DB_BENCH_BIN \
  --benchmarks=fillrandom,stats \
  --db="$db" \
  --num=$num \
  --threads=1 \
  --max_background_jobs=64 \
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
  --seed=$seed \
  --allow_concurrent_memtable_write=0
CMD
}

# fillseq (db, num) - no seed needed
mk_fillseq_cmd() {
  local db="$1" num="$2"
  cat <<CMD
$FILL_DB_BENCH_BIN \
  --benchmarks=fillseq,stats \
  --db="$db" \
  --num=$num \
  --threads=1 \
  --max_background_jobs=64 \
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
  --allow_concurrent_memtable_write=0
CMD
}

# warmup (db, num)
mk_readrandom_cmd() {
  local db="$1" num="$2"
  cat <<CMD
${READ_DB_BENCH_BIN} \
    --benchmarks=readrandom,stats \
  --use_existing_db=true \
  --db="$db" \
  --num=$num \
  --reads=$READS_FIXED \
  --key_size=24 \
  --value_size=1000 \
  --compression_type=none \
  --use_direct_io_for_flush_and_compaction=true \
  --use_direct_reads=true \
  --bytes_per_sync=0 \
  --seed=1724572420 \
  --bloom_bits=10 \
  --cache_index_and_filter_blocks=true \
  --statistics=1 \
  --cache_size=1 \
  --duration=$WARMUP_DURATION
CMD
}

# sstables (db)
mk_sstables_cmd() {
  local db="$1"
  cat <<CMD
  $SST_DB_BENCH_BIN \
  --benchmarks=sstables \
  --use_existing_db=true \
  --db="$db"
CMD
}

# ===== 메인 루프 =====
log "Starting workload: $WORKLOAD"
log "Database sizes: $DB_SIZES"
log "Iterations: $ITER"
if [[ "$WORKLOAD" == "fillrandom" ]]; then
  log "Seed type: $SEED_TYPE (base_seed: $FILL_SEED_BASE)"
else
  log "Seed type: N/A (fillseq uses sequential keys)"
fi
log "Output directory: $OUT_ROOT"
log "Database directory: $DB_ROOT"
echo

for size in $DB_SIZES; do
  NUM=$(calc_num_for_size "$size")
  if [[ "$NUM" == "ERR" ]]; then
    log "ERROR: Invalid DB_SIZES value: $size"; exit 1
  fi
  log ">>> DB Size=$size → num=$NUM"

  for i_num in $(seq 1 "$ITER"); do
    # zero-pad iteration to 3 digits for directory names (001, 002, ...)
    i="$(printf "%03d" "$i_num")"
    ts="$(date +%Y%m%d_%H%M%S)"
    run_dir="${OUT_ROOT}/run_${size}_${i}_${ts}"
    DB_DIR_RUN="${DB_ROOT}/${DB_NAME_PREFIX}_${size}_${i}_${ts}"

    # Seed 계산: fillrandom일 때만 의미 있음
    if [[ "$WORKLOAD" == "fillrandom" ]]; then
      if [[ "$SEED_TYPE" == "same" ]]; then
        fill_seed=$FILL_SEED_BASE
      else
        iter_idx=$((i_num - 1))
        fill_seed=$((FILL_SEED_BASE + iter_idx))
      fi
      seed_info="Seed: $fill_seed (seed_type: $SEED_TYPE)"
    else
      fill_seed=0  # fillseq doesn't use seed, but we need a variable
      seed_info="N/A (sequential)"
    fi

    mkdir -p "$run_dir"
    log "=== Iteration $i | Size: $size | DB: $DB_DIR_RUN | $seed_info ==="

    # (1) fill workload (fillrandom or fillseq)
    if [[ "$WORKLOAD" == "fillrandom" ]]; then
      run_and_save "$run_dir/fillrandom" "fillrandom" \
        "$(mk_fillrandom_cmd "$DB_DIR_RUN" "$fill_seed" "$NUM")" yes
    else
      run_and_save "$run_dir/fillseq" "fillseq" \
        "$(mk_fillseq_cmd "$DB_DIR_RUN" "$NUM")" yes
    fi

    # (2) readrandom warmup (no snapshots)
    run_and_save "$run_dir/readrandom" "readrandom" \
      "$(mk_readrandom_cmd "$DB_DIR_RUN" "$NUM")" no

    # (3) sstables dump (no snapshots)
    run_and_save "$run_dir/sstables" "sstables" \
      "$(mk_sstables_cmd "$DB_DIR_RUN")" no

    log "✓ Completed iteration $i for size=$size"
  done
done

log "=========================================="
log "✓ All iterations completed successfully!"
log "  Workload: $WORKLOAD"
log "  Sizes: $DB_SIZES"
if [[ "$WORKLOAD" == "fillrandom" ]]; then
  log "  Seed type: $SEED_TYPE"
fi
log "  Output: $OUT_ROOT"
log "=========================================="
