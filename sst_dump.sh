#!/usr/bin/env bash
# sst_dump.sh
# - 각 DB_DIR 내 모든 .sst를 스캔해 하나의 CSV로 저장
# - 출력: ./sst_dump_res/<DB basename>.csv
# - 컬럼: sst_id,key_hex,seq
# - sst_dump: ./S3-LOAD/sst_dump
# - 여러 DB_DIR을 한 번에 인자로 받아 순차 처리
set -euo pipefail

SST_DUMP_BIN="./S3-LOAD/sst_dump"

# -------- 옵션 파서 --------
JOBS=1
DB_DIRS=()
while (( "$#" )); do
  case "$1" in
    -j|--jobs)
      shift
      JOBS="${1:-1}"
      ;;
    -*)
      echo "[e] 알 수 없는 옵션: $1" >&2
      exit 2
      ;;
    *)
      DB_DIRS+=("$1")
      ;;
  esac
  shift
done

# -------- 검증 --------
if (( ${#DB_DIRS[@]} == 0 )); then
  echo "사용법: $0 [-j N|--jobs N] <DB_DIR...>" >&2
  echo "예시  : $0 -j 8 /work/tmp/s3load/fillrandom_1k_DB_280GB_*" >&2
  exit 2
fi
# JOBS 숫자 검증
if ! [[ "$JOBS" =~ ^[0-9]+$ ]]; then
  echo "[w] --jobs 값이 숫자가 아님: '$JOBS' → 1로 설정" >&2
  JOBS=1
fi
if (( JOBS < 1 )); then JOBS=1; fi

# flock 여부 (병렬 append 안전성)
USE_FLOCK=0
if command -v flock >/dev/null 2>&1; then
  USE_FLOCK=1
elif (( JOBS > 1 )); then
  echo "[w] 'flock' 없음 → 병렬 append 안전하지 않아 JOBS=1로 강제" >&2
  JOBS=1
fi

# -------- 공용 함수 --------
process_one_sst_file() {
  local sst="$1" out_csv="$2"
  local base name id tmp lock

  base="$(basename "$sst")"
  name="${base%.*}"
  if [[ "$name" =~ ^[0-9]+$ ]]; then
    id="$name"
  else
    id="$(echo "$name" | tr -cd '0-9')"
    [[ -z "$id" ]] && id="$name"
  fi

  # 임시 파일에 먼저 쓴 뒤, flock으로 out_csv에 단일 append
  tmp="$(mktemp "${out_csv}.part.XXXXXX")"
  lock="${out_csv}.lock"

  # sst_dump → gawk: 임시 파일에만 기록 (한 프로세스당 독립 파일)
  # LC_ALL=C로 성능 및 정규식 일관성 확보
  LC_ALL=C "$SST_DUMP_BIN" --file="$sst" --command=scan --output_hex 2>/dev/null \
  | gawk -v fid="$id" '
      match($0, /\x27([0-9A-Fa-f]+)\x27[[:space:]]+seq:([0-9]+)/, m) {
        # sst_id,key_hex,seq
        printf "%s,%s,%s\n", fid, m[1], m[2]
      }
    ' > "$tmp"

  # 여러 프로세스가 동시에 들어와도 여기서 직렬화됨
  if (( USE_FLOCK == 1 && JOBS > 1 )); then
    flock -w 60 "$lock" -c "cat \"$tmp\" >> \"$out_csv\""
  else
    cat "$tmp" >> "$out_csv"
  fi

  rm -f "$tmp"
}

export -f process_one_sst_file
export SST_DUMP_BIN JOBS

# -------- DB_DIR들 순차 처리 --------
mkdir -p ./sst_dump_res
for DB_DIR in "${DB_DIRS[@]}"; do
  if ! [[ -d "$DB_DIR" ]]; then
    echo "[w] 건너뜀(디렉토리 아님): $DB_DIR" >&2
    continue
  fi

  DB_BASENAME="$(basename "$DB_DIR")"
  OUT_CSV="./sst_dump_res/${DB_BASENAME}.csv"

  echo "[i] DB_DIR   : $DB_DIR"
  echo "[i] OUT_CSV  : $OUT_CSV"
  echo "[i] JOBS     : $JOBS"

  # CSV 헤더 초기화(덮어씀)
  echo "sst_id,key_hex,seq" > "$OUT_CSV"

  # SST 목록
  mapfile -d '' sst_list < <(find "$DB_DIR" -type f -name '*.sst' -print0 | sort -z)
  echo "[i] SST files: ${#sst_list[@]}"

  # 실행
  if (( JOBS > 1 )); then
    if command -v parallel >/dev/null 2>&1; then
      printf '%s\0' "${sst_list[@]}" \
      | parallel -0 -P "$JOBS" process_one_sst_file {} "$OUT_CSV"
    else
      printf '%s\0' "${sst_list[@]}" \
      | xargs -0 -n1 -P "$JOBS" -I{} bash -c 'process_one_sst_file "$1" "$2"' _ {} "$OUT_CSV"
    fi
  else
    for sst in "${sst_list[@]}"; do
      process_one_sst_file "$sst" "$OUT_CSV"
    done
  fi

  echo "[i] Done → $OUT_CSV"
done
