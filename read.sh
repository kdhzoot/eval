#!/usr/bin/env bash
set -euo pipefail

# ===== 사용법 =====
usage() {
  cat <<EOF
Usage: $0 [REQUIRED OPTIONS] [OPTIONAL OPTIONS]

Required Options:
  db_dir=DIR                 Database root directory (searches for DB subdirectories)
  workload=WORKLOAD          Workload type: readrandom, mixgraph

Optional Options:
  iter=NUM                   Number of DB iterations to process (default: all matching DBs)
  cache_sizes="SIZE1 SIZE2"  Cache sizes (space-separated, default: "1")
                             Units: B, K/KB, M/MB, G/GB, KiB/MiB/GiB
                             Example: "1 16MiB 32MiB 512MiB"
  db_prefix=PREFIX           Database name prefix pattern (default: fillrandom_1k)
  reads=NUM                  Number of reads per benchmark (default: 1000000)
  num=NUM                    Total keys in database (default: auto-detect from prefix)
  cmd_prefix="PREFIX"        Command prefix for NUMA/taskset (default: "")
  out_dir=DIR                Output directory (default: auto-generated)

Output directory is automatically generated as:
  runs_read_<workload>_<timestamp>/

Examples:
  # Basic readrandom with 1B cache
  $0 db_dir=/work/tmp/test workload=readrandom iter=10

  # Multiple cache sizes
  $0 db_dir=/work/tmp/test workload=readrandom iter=5 cache_sizes="1 16MiB 512MiB"

  # Mixgraph workload with NUMA binding
  $0 db_dir=/work/tmp/test workload=mixgraph iter=3 cmd_prefix="numactl -N0 -m0"

  # Custom DB prefix and reads
  $0 db_dir=/work/tmp/test workload=readrandom iter=10 db_prefix=fillseq_1k reads=5000000

EOF
  exit 1
}

# ===== 기본 설정 (선택적 옵션만 기본값 설정) =====
DB_ROOT=""
ITER=""
WORKLOAD=""
CACHE_SIZES="1"
DB_NAME_PREFIX="fillrandom_1k"
READS_FIXED="1000000"
NUM=""
CMD_PREFIX=""
OUT_ROOT=""

READ_DB_BENCH_BIN="${READ_DB_BENCH_BIN:-/home/smrc/TTLoad/rocksdb_perf/db_bench}"

# ===== 명령줄 인자 파싱 =====
for arg in "$@"; do
  case "$arg" in
    db_dir=*)
      DB_ROOT="${arg#db_dir=}"
      ;;
    iter=*)
      ITER="${arg#iter=}"
      ;;
    workload=*)
      WORKLOAD="${arg#workload=}"
      ;;
    cache_sizes=*)
      CACHE_SIZES="${arg#cache_sizes=}"
      ;;
    db_prefix=*)
      DB_NAME_PREFIX="${arg#db_prefix=}"
      ;;
    reads=*)
      READS_FIXED="${arg#reads=}"
      ;;
    num=*)
      NUM="${arg#num=}"
      ;;
    cmd_prefix=*)
      CMD_PREFIX="${arg#cmd_prefix=}"
      ;;
    out_dir=*)
      OUT_ROOT="${arg#out_dir=}"
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

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ===== 필수 옵션 검증 =====
MISSING_OPTS=""
[[ -z "$DB_ROOT" ]] && MISSING_OPTS="$MISSING_OPTS db_dir"
[[ -z "$WORKLOAD" ]] && MISSING_OPTS="$MISSING_OPTS workload"

if [[ -n "$MISSING_OPTS" ]]; then
  echo "Error: Missing required options:$MISSING_OPTS"
  echo
  usage
fi

# ===== 워크로드 검증 =====
if [[ "$WORKLOAD" != "readrandom" && "$WORKLOAD" != "mixgraph" ]]; then
  echo "Error: workload must be 'readrandom' or 'mixgraph'"
  exit 1
fi

# WORKLOAD → 실제 --benchmarks 문자열로 변환
benchmarks_of() {
  case "$1" in
    readrandom) echo "readrandom,stats" ;;
    mixgraph)   echo "mixgraph,stats" ;;
    *)          echo "$1" ;;
  esac
}
WORKLOAD_BENCHMARKS="$(benchmarks_of "$WORKLOAD")"




# ===== 출력 디렉토리 자동 생성 =====
if [[ -z "$OUT_ROOT" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  # DB_ROOT에서 마지막 디렉토리 이름 추출
  DB_BASENAME="$(basename "$DB_ROOT")"
  OUT_ROOT="/home/smrc/TTLoad/eval/runs_${WORKLOAD}_${DB_BASENAME}_${TS}"
fi

# ===== 유틸 =====

# DB 경로에서 NUM 계산: /path/to/fillrandom_1k_100GB_001 → 100000000
calculate_num() {
  local db_path="$1"
  local db_name="$(basename "$db_path")"
  
  # 전역 NUM이 지정된 경우 그대로 사용
  if [[ -n "$NUM" ]]; then
    echo "$NUM"
    return 0
  fi
  
  # DB 이름에서 크기 추출: fillrandom_1k_100GB_001 → 100GB
  if [[ "$db_name" =~ _([0-9]+)(GB|TB)_ ]]; then
    local size_val="${BASH_REMATCH[1]}"
    local size_unit="${BASH_REMATCH[2]}"
    if [[ "$size_unit" == "GB" ]]; then
      echo $((size_val * 1000000))
    elif [[ "$size_unit" == "TB" ]]; then
      echo $((size_val * 1000000 * 1000))
    fi
  else
    # 기본값
    echo "280000000"
  fi
}

require_root_for_drop() {
  if [[ ! -w /proc/sys/vm/drop_caches ]]; then
    log "[ERROR] Need root to drop caches." >&2
    exit 1
  fi
}
drop_caches() { sync; echo 3 > /proc/sys/vm/drop_caches; }

# "16MB", "512MiB", "1G", "1048576" → 바이트 정수
parse_bytes() {
  local s
  s="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$s" =~ ^[0-9]+$ ]]; then echo "$s"; return 0; fi
  local num unit
  num="$(echo "$s" | sed -E 's/^([0-9]+).*/\1/')"
  unit="$(echo "$s" | sed -E 's/^[0-9]+(.*)$/\1/')"
  case "$unit" in
    b|"")  echo $(( num ));;
    k|kb)  echo $(( num * 1000 ));;
    m|mb)  echo $(( num * 1000 * 1000 ));;
    g|gb)  echo $(( num * 1000 * 1000 * 1000 ));;
    kib)   echo $(( num * 1024 ));;
    mib)   echo $(( num * 1024 * 1024 ));;
    gib)   echo $(( num * 1024 * 1024 * 1024 ));;
    *) echo "[ERROR] 알 수 없는 단위: '$1'" >&2; return 2;;
  esac
}

# 1048576 → "1MB", 16777216 → "16MB" (1은 "1B")
label_bytes() {
  local b="$1"
  if [[ "$b" -eq 1 ]]; then echo "1B"; else echo "$(( b / 1000 / 1000 ))MB"; fi
}

# /proc/diskstats, /proc/stat 스냅샷 저장
# 사용법: snap_proc_stats <out_dir> <LABEL>
snap_proc_stats() {
  local outdir="$1" label="$2" ts
  ts="$(date +%Y-%m-%dT%H:%M:%S%z)"

  mkdir -p "$outdir"

  # 통합 로그(append)
  {
    echo "=== /proc/diskstats ${label} ${ts} ==="
    cat /proc/diskstats || true
    echo
    echo "=== /proc/stat ${label} ${ts} ==="
    cat /proc/stat || true
    echo
  } >>"$outdir/sysstats.log"

  # 원본 파일도 개별 저장(덮어쓰기)
  cat /proc/diskstats >"$outdir/diskstats_${label}.txt"  || true
  cat /proc/stat      >"$outdir/stat_${label}.txt"       || true
}


run_and_save() {
  local dir="$1" name="$2" cmd="$3"
  mkdir -p "$dir"
  log "RUN [$name]"
  log "WORKLOAD=$WORKLOAD | benchmarks=$WORKLOAD_BENCHMARKS"
 
  # ==== BEFORE snapshot ====
  log "Snapshotting /proc stats BEFORE → $dir"
  snap_proc_stats "$dir" "BEFORE"

  set +e
  if [[ -n "$CMD_PREFIX" ]]; then
    bash -lc "$CMD_PREFIX $cmd" 2>&1 \
      | sed -u -e 's/\r/\n/g' \
               -e '/^[.][.][.] finished [0-9][0-9]* ops[[:space:]]*$/d' \
               -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  else
    bash -lc "$cmd" 2>&1 \
      | sed -u -e 's/\r/\n/g' \
               -e '/^[.][.][.] finished [0-9][0-9]* ops[[:space:]]*$/d' \
               -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  fi
  local rc=$?
  set -e

  # ==== AFTER snapshot ====
  log "Snapshotting /proc stats AFTER → $dir"
  snap_proc_stats "$dir" "AFTER"


  if [[ $rc -ne 0 ]]; then
    log "WARN [$name]: exit code $rc (stdout.log 확인)"
  fi
}

# ===== 측정 커맨드 구성 =====
mk_measure_cmd() {
  local db="$1" cache_bytes="$2" num="$3"
  case "$WORKLOAD" in
    mixgraph)
      cat <<CMD
$READ_DB_BENCH_BIN \
  --benchmarks=$WORKLOAD_BENCHMARKS \
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
  --statistics=1 \
  --cache_index_and_filter_blocks=true \
  --key_dist_a=0.002312 \
  --key_dist_b=0.3467 \
  --keyrange_dist_a=14.18 \
  --keyrange_dist_b=-2.917 \
  --keyrange_dist_c=0.0164 \
  --keyrange_dist_d=-0.08082 \
  --keyrange_num=30 \
  --value_k=0.2615 \
  --value_sigma=25.45 \
  --iter_k=2.517 \
  --iter_sigma=14.236 \
  --mix_get_ratio=1 \
  --mix_put_ratio=0 \
  --mix_seek_ratio=0 \
  --sine_mix_rate_interval_milliseconds=5000 \
  --sine_a=1000 \
  --sine_b=0.000073 \
  --sine_d=4500 \
  --perf_level=2 \
  --cache_size=$cache_bytes
CMD
      ;;
    *)
      # 기본: readrandom
      cat <<CMD
$READ_DB_BENCH_BIN \
  --benchmarks=$WORKLOAD_BENCHMARKS \
  --use_existing_db=true \
  --db="$db" \
  --num=$num \
  --reads=$READS_FIXED \
  --threads=1 \
  --key_size=24 \
  --value_size=1000 \
  --compression_type=none \
  --use_direct_io_for_flush_and_compaction=true \
  --use_direct_reads=true \
  --bytes_per_sync=0 \
  --seed=1724572420 \
  --bloom_bits=10 \
  --statistics=1 \
  --cache_index_and_filter_blocks=true \
  --perf_level=2 \
  --cache_size=$cache_bytes
CMD
      ;;
  esac
}

 


collect_db_dirs() {
  find "$DB_ROOT" -maxdepth 1 -type d -name "${DB_NAME_PREFIX}_*" | sort
}

# ===== 메인 =====
trap 'echo; log "Interrupted"; exit 130' INT
require_root_for_drop
mkdir -p "$OUT_ROOT"

log "=========================================="
log "Starting READ workload"
log "=========================================="
log "Workload: $WORKLOAD (benchmarks=$WORKLOAD_BENCHMARKS)"
log "Database root: $DB_ROOT"
log "DB prefix pattern: ${DB_NAME_PREFIX}_*"
log "Iterations: $ITER"
log "Reads per benchmark: $READS_FIXED"
log "Cache sizes: $CACHE_SIZES"
log "Output directory: $OUT_ROOT"
log "db_bench binary: $READ_DB_BENCH_BIN"
if [[ -n "$CMD_PREFIX" ]]; then
  log "Command prefix: '$CMD_PREFIX'"
fi
log "=========================================="
echo

mapfile -t ALL_DB_DIRS < <(collect_db_dirs)
if [[ "${#ALL_DB_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] 대상 DB 없음: ${DB_ROOT}/${DB_NAME_PREFIX}_*" >&2
  exit 2
fi

# ITER 기본값: 모든 DB
if [[ -z "$ITER" ]]; then
  ITER="${#ALL_DB_DIRS[@]}"
  log "ITER not specified, using all $ITER matching databases"
fi

SEL_DB_DIRS=("${ALL_DB_DIRS[@]:0:$ITER}")
log "선택된 DB 개수: ${#SEL_DB_DIRS[@]} / 전체 후보: ${#ALL_DB_DIRS[@]}"

declare -a CACHE_BYTES=()
for tok in $CACHE_SIZES; do
  b="$(parse_bytes "$tok")" || { echo "[ERROR] 캐시 크기 파싱 실패: $tok" >&2; exit 3; }
  CACHE_BYTES+=("$b")
done
log "캐시 크기(바이트): ${CACHE_BYTES[*]}"

for idx in "${!SEL_DB_DIRS[@]}"; do
  db="${SEL_DB_DIRS[$idx]}"
  
  # 이 DB에 대한 NUM 계산
  CURRENT_NUM="$(calculate_num "$db")"
  log "DB: $(basename "$db") → NUM=$CURRENT_NUM"
  
  i=$(printf "%03d" $((idx + 1)))
  ts="$(date +%Y%m%d_%H%M%S)"
  db_name="$(basename "$db")"
  run_dir_base="${OUT_ROOT}/run_${i}_${db_name}_${ts}"
  mkdir -p "$run_dir_base"
  log "=== Iter $i | DB: $db | logs: $run_dir_base ==="
  log "Iter $i WORKLOAD=$WORKLOAD | benchmarks=$WORKLOAD_BENCHMARKS"

  for b in "${CACHE_BYTES[@]}"; do
    label="$(label_bytes "$b")"
    run_dir="${run_dir_base}/cache_${label}"
    mkdir -p "$run_dir"

    log "Dropping caches… (cache_size=$label)"
    drop_caches

    cmd="$(mk_measure_cmd "$db" "$b" "$CURRENT_NUM")"
    log "EXEC CMD: $cmd"

    run_and_save "$run_dir" "measure_${WORKLOAD}_cache_${label}" "$cmd"
  done

  log "✓ Completed iteration $i → $run_dir_base"
done

log "=========================================="
log "✓ All iterations completed successfully!"
log "  Workload: $WORKLOAD"
log "  Total iterations: $ITER"
log "  Cache sizes tested: $CACHE_SIZES"
log "  Output: $OUT_ROOT"
log "=========================================="
