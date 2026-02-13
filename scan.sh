#!/usr/bin/env bash
# orchestrator_seqscan_move.sh (Updated to align with read.sh)
# - 단일 DB 또는 다수 DB 대상: DB_PATH 또는 디렉토리 패턴으로 실행
# - s3readseq 실행 전후로 /proc/stats 스냅샷 저장
# - 실행 후 /home/smrc/TTLoad/readlogs를 /home/smrc/TTLoad/pscan으로 이동

set -euo pipefail

# ===== 기본 설정 =====
CMD_PREFIX="${CMD_PREFIX:-}"
READ_DB_BENCH_BIN="${READ_DB_BENCH_BIN:-/home/smrc/TTLoad/S3-LOAD/db_bench}"
OUT_ROOT=""  # 하단에서 자동 생성

# s3readseq 고정 파라미터 (read.sh 스타일로 환경변수 지원)
NUM="${NUM:-}" # 비어있을 경우 calculate_num에서 자동 계산
READS="${READS:-7000000}"
THREADS="${THREADS:-100}"
KEY_SIZE="${KEY_SIZE:-24}"
VALUE_SIZE="${VALUE_SIZE:-1000}"
CACHE_SIZE_BYTES="${CACHE_SIZE_BYTES:-$((512*1024*1024*1024))}" # 512GB
SEED="${SEED:-1724572420}"

# ===== 유틸리티 함수 (read.sh에서 복사/참조) =====
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

calculate_num() {
  local db_path="$1"
  local db_name="$(basename "$db_path")"
  if [[ -n "$NUM" ]]; then echo "$NUM"; return 0; fi

  if [[ "$db_name" =~ _([0-9]+)(GB|TB)_ ]]; then
    local size_val="${BASH_REMATCH[1]}"
    local size_unit="${BASH_REMATCH[2]}"
    if [[ "$size_unit" == "GB" ]]; then
      echo $((size_val * 1000000))
    elif [[ "$size_unit" == "TB" ]]; then
      echo $((size_val * 1000000 * 1000))
    fi
  else
    echo "700000000" # 기본값
  fi
}

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
  cat /proc/diskstats >"$outdir/diskstats_${label}.txt" || true
  cat /proc/stat >"$outdir/stat_${label}.txt" || true
}

run_and_save() {
  local dir="$1" name="$2" cmd="$3"
  mkdir -p "$dir"
  log "RUN [$name]"
  
  snap_proc_stats "$dir" "BEFORE"
  set +e
  if [[ -n "$CMD_PREFIX" ]]; then
    bash -lc "$CMD_PREFIX $cmd" 2>&1 | sed -u -e 's/\r/\n/g' -e '/^[.][.][.] finished/d' -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  else
    bash -lc "$cmd" 2>&1 | sed -u -e 's/\r/\n/g' -e '/^[.][.][.] finished/d' -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  fi
  local rc=$?
  set -e
  snap_proc_stats "$dir" "AFTER"
  [[ $rc -ne 0 ]] && log "WARN [$name]: exit code $rc"
}

mk_seqscan_cmd() {
  local db="$1" n_val="$2"
  cat <<CMD
$READ_DB_BENCH_BIN \\
  --benchmarks=s3readseq,stats \\
  --use_existing_db=true \\
  --db="$db" \\
  --num=$n_val \\
  --reads=$READS \\
  --threads=$THREADS \\
  --key_size=$KEY_SIZE \\
  --value_size=$VALUE_SIZE \\
  --compression_type=none \\
  --use_direct_io_for_flush_and_compaction=true \\
  --use_direct_reads=true \\
  --bytes_per_sync=0 \\
  --seed=$SEED \\
  --bloom_bits=10 \\
  --cache_index_and_filter_blocks=true \\
  --statistics=1 \\
  --cache_size=$CACHE_SIZE_BYTES
CMD
}

# ===== 메인 스크립트 =====
trap 'echo; log "Interrupted"; exit 130' INT

# DB 경로 및 출력 설정
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  DB_PATH="$1"
else
  echo "[ERROR] DB_PATH is required." >&2; exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
DB_BASE="$(basename "$DB_PATH")"
OUT_ROOT="/home/smrc/TTLoad/eval/runs/runs_scan_${DB_BASE}_${TS}"
mkdir -p "$OUT_ROOT"

log "SEQ-SCAN | DB: $DB_PATH"
log "OUT_ROOT: $OUT_ROOT"

# scan.sh 특화된 경로 설정
SRC_BASE="/home/smrc/TTLoad/readlogs"
DEST_BASE="/home/smrc/TTLoad/pscan"
mkdir -p "$DEST_BASE"

# 실행 (단일 DB 처리 구조 유지하되 read.sh 명명 규칙 적용)
CURRENT_NUM="$(calculate_num "$DB_PATH")"
run_dir_base="${OUT_ROOT}/run_001_${DB_BASE}_${TS}"

seq_cmd="$(mk_seqscan_cmd "$DB_PATH" "$CURRENT_NUM")"
run_and_save "$run_dir_base" "s3readseq_${DB_BASE}" "$seq_cmd"

# 사용한 디렉토리 이동
if [[ -d "$SRC_BASE" ]]; then
  dst_dir="${DEST_BASE}/${DB_BASE}_${TS}"
  log "Moving scan logs: $SRC_BASE -> $dst_dir"
  cp -a "$SRC_BASE" "$dst_dir"
else
  log "WARN: No scan logs found at $SRC_BASE"
fi

log "Done. Result in $OUT_ROOT"