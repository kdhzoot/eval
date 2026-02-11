#!/usr/bin/env bash
# orchestrator_seqscan_move.sh
# - 단일 DB 전용: DB_PATH=<absolute_db_path> 또는 첫 번째 인자로 DB 경로 전달
# - 각 대상에 대해:
#   (1) 전체 range 스캔: s3readseq,stats 실행
#   (2) 사용한 디렉토리 이동: /home/smrc/TTLoad/scan/<DB_BASENAME> -> /home/smrc/TTLoad/S3-LOAD/readlogs
#
# 변경점:
# - 캐시 크기 스윕/워밍업/summary.txt/캐시 드롭 제거
# - 실행 단계는 2단계로 단순화

set -euo pipefail

# ===== 기본 설정(환경변수로 덮어쓰기) =====
CMD_PREFIX="${CMD_PREFIX:-}"            # 예: "numactl -N0 -m0 taskset -c 0-39"

READ_DB_BENCH_BIN="${READ_DB_BENCH_BIN:-/home/smrc/TTLoad/S3-LOAD/db_bench}"
OUT_ROOT="${OUT_ROOT:-/home/smrc/TTLoad/runs_seqscan}"   # 스캔 실행 로그 보관 위치

DB_PATH="${DB_PATH:-}"               # 필수. 예: /work/tmp/.../fillrandom_1k_280GB_...

# s3readseq 고정 파라미터(요청 예시값 기반). 필요시 환경변수로 덮어쓰기 가능
NUM="${NUM:-700000000}"
READS="${READS:-7000000}"
THREADS="${THREADS:-100}"
KEY_SIZE="${KEY_SIZE:-24}"
VALUE_SIZE="${VALUE_SIZE:-1000}"
CACHE_SIZE_BYTES="${CACHE_SIZE_BYTES:-$((50*1024*1024*1024))}"  # 50GiB
SEED="${SEED:-1724572420}"

# ===== 유틸 =====
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

run_and_save() {
  local dir="$1" name="$2" cmd="$3"
  mkdir -p "$dir"
  log "RUN [$name]"
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
  if [[ $rc -ne 0 ]]; then
    log "WARN [$name]: exit code $rc (stdout.log 확인)"
  fi
}

mk_seqscan_cmd() {
  local db="$1"
  cat <<CMD
$READ_DB_BENCH_BIN \
  --benchmarks=s3readseq,stats \
  --use_existing_db=true \
  --db="$db" \
  --num=$NUM \
  --reads=$READS \
  --threads=$THREADS \
  --key_size=$KEY_SIZE \
  --value_size=$VALUE_SIZE \
  --compression_type=none \
  --use_direct_io_for_flush_and_compaction=true \
  --use_direct_reads=true \
  --bytes_per_sync=0 \
  --seed=$SEED \
  --bloom_bits=10 \
  --cache_index_and_filter_blocks=true \
  --statistics=1 \
  --cache_size=$CACHE_SIZE_BYTES
CMD
}

# ===== 메인 =====
trap 'echo; log "Interrupted"; exit 130' INT
mkdir -p "$OUT_ROOT"

log "SEQ-SCAN | single DB | s3readseq 후 디렉토리 이동"
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  DB_PATH="$1"
fi

if [[ -z "$DB_PATH" ]]; then
  echo "[ERROR] DB_PATH is required. Pass as env(DB_PATH=/path/to/db) or first arg." >&2
  exit 2
fi
if [[ ! -d "$DB_PATH" ]]; then
  echo "[ERROR] DB_PATH not found: $DB_PATH" >&2
  exit 2
fi

log "DB_PATH=$DB_PATH"
log "OUT_ROOT=$OUT_ROOT"
log "READ_DB_BENCH_BIN=$READ_DB_BENCH_BIN"
log "NUM=$NUM READS=$READS THREADS=$THREADS"
log "CACHE_SIZE_BYTES=$CACHE_SIZE_BYTES CMD_PREFIX='${CMD_PREFIX}'"

SEL_DB_DIRS=("$DB_PATH")
log "선택된 DB 개수: 1"

SRC_BASE="/home/smrc/TTLoad/readlogs"
DEST_BASE="/home/smrc/TTLoad/pscan"
mkdir -p "$DEST_BASE"

for idx in "${!SEL_DB_DIRS[@]}"; do
  db="${SEL_DB_DIRS[$idx]}"
  i=$(printf "%03d" $((idx + 1)))
  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir_base="${OUT_ROOT}/run_${i}_${ts}"
  mkdir -p "$run_dir_base"

  db_base="$(basename "$db")"
  log "=== Iter $i | DB: $db (base: $db_base) | logs: $run_dir_base ==="

  # (1) 전체 range 스캔 실행
  seq_cmd="$(mk_seqscan_cmd "$db")"
  run_and_save "$run_dir_base" "s3readseq_${db_base}" "$seq_cmd"

  # (2) 사용한 디렉토리 이동
  src_dir="${SRC_BASE}"
  dst_dir="${DEST_BASE}/${db_base}"
  if [[ -d "$src_dir" ]]; then
    log "Moving scan dir → $dst_dir"
    # cp -r 요구사항: 메타데이터 보존 위해 -a 사용 (원요청은 -r, 호환 가능)
    cp -a "$src_dir" "$dst_dir"
  else
    log "WARN: 소스 디렉토리 없음: $src_dir (s3readseq가 로그/아웃풋을 생성했는지 확인)"
  fi

  log "Completed iteration $i → $run_dir_base"
done

log "All iterations done."
