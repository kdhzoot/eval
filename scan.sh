#!/usr/bin/env bash
# orchestrator_seqscan_move.sh
# - fillrandom_1k_280GB_* DB들을 스캔 대상 목록으로 수집/선정 (이전 스크립트 로직 유지)
# - 각 대상에 대해:
#   (1) 전체 range 스캔: s3readseq,stats 실행
#   (2) 사용한 디렉토리 이동: /home/smrc/TTLoad/scan/<DB_BASENAME> -> /home/smrc/TTLoad/S3-LOAD/readlogs
#
# 변경점:
# - 캐시 크기 스윕/워밍업/summary.txt/캐시 드롭 제거
# - 실행 단계는 2단계로 단순화

set -euo pipefail

# ===== 기본 설정(환경변수로 덮어쓰기) =====
ITER="${ITER:-30}"                      # 이번에 돌릴 DB 개수
CMD_PREFIX="${CMD_PREFIX:-}"            # 예: "numactl -N0 -m0 taskset -c 0-39"

READ_DB_BENCH_BIN="${READ_DB_BENCH_BIN:-/home/smrc/TTLoad/S3-LOAD/db_bench}"
OUT_ROOT="${OUT_ROOT:-/home/smrc/TTLoad/runs_seqscan}"   # 스캔 실행 로그 보관 위치

DB_ROOT="${DB_ROOT:-/work/tmp/godong/diff_seed}"
DB_NAME_PREFIX="${DB_NAME_PREFIX:-fillrandom_1k_280GB}"   # 280GB만 선택

# s3readseq 고정 파라미터(요청 예시값 기반). 필요시 환경변수로 덮어쓰기 가능
NUM="${NUM:-280000000}"
READS="${READS:-7000000}"
THREADS="${THREADS:-40}"
KEY_SIZE="${KEY_SIZE:-24}"
VALUE_SIZE="${VALUE_SIZE:-1000}"
CACHE_SIZE_BYTES="${CACHE_SIZE_BYTES:-$((25*1024*1024*1024))}"  # 25GiB
SEED="${SEED:-1724572420}"

ORDER="${ORDER:-name}"                 # name | mtime | mtime_desc
SHUFFLE="${SHUFFLE:-0}"                # 1 → 무작위

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

collect_db_dirs() {
  local -a arr=()
  case "$ORDER" in
    name)
      while IFS= read -r -d '' d; do arr+=("$d"); done \
        < <(find "$DB_ROOT" -maxdepth 1 -type d -name "${DB_NAME_PREFIX}_*" -print0 | sort -z)
      ;;
    mtime)
      while IFS= read -r line; do arr+=("${line#* }"); done \
        < <(find "$DB_ROOT" -maxdepth 1 -type d -name "${DB_NAME_PREFIX}_*" -printf '%T@ %p\n' | sort -n)
      ;;
    mtime_desc)
      while IFS= read -r line; do arr+=("${line#* }"); done \
        < <(find "$DB_ROOT" -maxdepth 1 -type d -name "${DB_NAME_PREFIX}_*" -printf '%T@ %p\n' | sort -nr)
      ;;
    *)
      echo "[WARN] Unknown ORDER='$ORDER' → name 정렬 사용"
      while IFS= read -r -d '' d; do arr+=("$d"); done \
        < <(find "$DB_ROOT" -maxdepth 1 -type d -name "${DB_NAME_PREFIX}_*" -print0 | sort -z)
      ;;
  esac
  if [[ "$SHUFFLE" == "1" && "${#arr[@]}" -gt 1 ]]; then
    mapfile -t arr < <(printf '%s\n' "${arr[@]}" | shuf)
  fi
  printf '%s\n' "${arr[@]}"
}

# ===== 메인 =====
trap 'echo; log "Interrupted"; exit 130' INT
mkdir -p "$OUT_ROOT"

log "SEQ-SCAN | 280GB 전용 | s3readseq 후 디렉토리 이동"
log "DB_ROOT=$DB_ROOT / PREFIX=$DB_NAME_PREFIX"
log "ITER=$ITER OUT_ROOT=$OUT_ROOT ORDER=$ORDER SHUFFLE=$SHUFFLE"
log "READ_DB_BENCH_BIN=$READ_DB_BENCH_BIN"
log "NUM=$NUM READS=$READS THREADS=$THREADS"
log "CACHE_SIZE_BYTES=$CACHE_SIZE_BYTES CMD_PREFIX='${CMD_PREFIX}'"

mapfile -t ALL_DB_DIRS < <(collect_db_dirs)
if [[ "${#ALL_DB_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] 대상 DB 없음: ${DB_ROOT}/${DB_NAME_PREFIX}_*" >&2
  exit 2
fi
SEL_DB_DIRS=("${ALL_DB_DIRS[@]:0:$ITER}")
log "선택된 DB 개수: ${#SEL_DB_DIRS[@]} / 전체 후보: ${#ALL_DB_DIRS[@]}"

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

