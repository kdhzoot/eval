#!/usr/bin/env bash
set -euo pipefail

# ===== 사용법 =====
usage() {
  cat <<EOF
Usage: $0 <db_dir> [OPTIONS]

Arguments:
  db_dir                     Directory containing databases (required)

Optional Options:
  db_bench_bin=BIN           Path to db_bench binary (default: /home/smrc/TTLoad/rocksdb_perf/db_bench)
  out_root=DIR               Output directory (auto-generated if not specified)

Examples:
  $0 /work/tmp/load_700GB
  $0 /work/tmp/load_700GB db_bench_bin=/path/to/db_bench

EOF
  exit 1
}

# ===== 기본 설정 =====
DB_ROOT=""
DB_BENCH_BIN="${DB_BENCH_BIN:-/home/smrc/TTLoad/rocksdb_perf/db_bench}"
OUT_ROOT=""

# ===== 명령줄 인자 파싱 =====
if [[ $# -eq 0 ]]; then
  usage
fi

DB_ROOT="$1"
shift

for arg in "$@"; do
  case "$arg" in
    db_bench_bin=*)
      DB_BENCH_BIN="${arg#db_bench_bin=}"
      ;;
    out_root=*)
      OUT_ROOT="${arg#out_root=}"
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

# ===== db_dir 검증 =====
if [[ ! -d "$DB_ROOT" ]]; then
  echo "Error: Database directory does not exist: $DB_ROOT" >&2
  exit 1
fi

# ===== 출력 디렉토리 자동 생성 (미지정시) =====
if [[ -z "$OUT_ROOT" ]]; then
  DB_ROOT_NAME="$(basename "$DB_ROOT")"
  OUT_ROOT="/home/smrc/TTLoad/eval/sstables_${DB_ROOT_NAME}"

  # Avoid clobbering an existing run directory without relying on timestamps.
  if [[ -e "$OUT_ROOT" ]]; then
    base_out_root="$OUT_ROOT"
    n=1
    while [[ -e "${base_out_root}_${n}" ]]; do
      n=$((n + 1))
    done
    OUT_ROOT="${base_out_root}_${n}"
  fi
fi

# ===== 유틸 =====
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }

run_and_save() {
  local dir="$1" name="$2" cmd="$3"
  mkdir -p "$dir"

  set +e
  bash -lc "$cmd" 2>&1 \
    | sed -u -e 's/\r/\n/g' \
             -e '/^[.][.][.] finished [0-9][0-9]* ops[[:space:]]*$/d' \
             -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    log "WARN [$name]: exit code $rc (see $dir/stdout.log)"
  fi
}

# ===== SSTable 벤치마크 커맨드 구성 =====
mk_sstables_cmd() {
  local db="$1"
  cat <<CMD
$DB_BENCH_BIN \
  --benchmarks=sstables \
  --use_existing_db=true \
  --db="$db"
CMD
}

# ===== DB 디렉토리 수집 (이름순 정렬) =====
collect_db_dirs() {
  find "$DB_ROOT" -maxdepth 1 -type d ! -name . -printf '%f\n' | sort
}

# ===== 메인 =====
trap 'echo; log "Interrupted"; exit 130' INT
mkdir -p "$OUT_ROOT"

log "SSTABLES benchmark"
log "Database directory: $DB_ROOT"
log "Output directory: $OUT_ROOT"
log "DB Bench binary: $DB_BENCH_BIN"
echo

mapfile -t ALL_DB_DIRS < <(collect_db_dirs)
if [[ "${#ALL_DB_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] No databases found in: $DB_ROOT" >&2
  exit 2
fi

log "Total databases found: ${#ALL_DB_DIRS[@]}"
echo

for idx in "${!ALL_DB_DIRS[@]}"; do
  db_name="${ALL_DB_DIRS[$idx]}"
  db="${DB_ROOT}/${db_name}"
  safe_db_name="${db_name//[^A-Za-z0-9._-]/_}"

  i=$(printf "%03d" $((idx + 1)))
  run_dir="${OUT_ROOT}/sstables_${i}_${safe_db_name}"

  # Avoid clobbering an existing per-DB directory.
  if [[ -e "$run_dir" ]]; then
    base_run_dir="$run_dir"
    n=1
    while [[ -e "${base_run_dir}_${n}" ]]; do
      n=$((n + 1))
    done
    run_dir="${base_run_dir}_${n}"
  fi
  mkdir -p "$run_dir"

  log "=== Iteration $i | DB: $db_name ==="
  log "Output: $run_dir"

  cmd="$(mk_sstables_cmd "$db")"
  log "EXEC CMD: $cmd"

  run_and_save "$run_dir" "sstables_${i}" "$cmd"

  log "✓ Completed iteration $i"
  echo
done

log "=========================================="
log "✓ All iterations completed successfully!"
log "  Total databases processed: ${#ALL_DB_DIRS[@]}"
log "  Output: $OUT_ROOT"
log "=========================================="
