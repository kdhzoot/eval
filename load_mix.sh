#!/usr/bin/env bash
set -euo pipefail

# ===== 기본 설정 =====
ITER="${ITER:-30}"
CMD_PREFIX="${CMD_PREFIX:-}"

FILL_DB_BENCH_BIN="${FILL_DB_BENCH_BIN:-/home/smrc/TTLoad/rocksdb_ori_9.6.1/db_bench}"
# FILL_DB_BENCH_BIN="${FILL_DB_BENCH_BIN:-/home/smrc/TTLoad/S3-LOAD/db_bench}"
READ_DB_BENCH_BIN="${READ_DB_BENCH_BIN:-/home/smrc/TTLoad/S3-LOAD/db_bench}"

OUT_ROOT="${OUT_ROOT:-/home/smrc/TTLoad/runs}"
DB_ROOT="${DB_ROOT:-${DB_DIR:-/work/tmp/fillandmix}}"
DB_NAME_PREFIX="${DB_NAME_PREFIX:-fillrandom_1k_DB}"

READS_FIXED="${READS_FIXED:-1000000}"
WARMUP_DURATION="${WARMUP_DURATION:-120}"
FILL_SEED_BASE="${FILL_SEED_BASE:-1722070161}"

# 여러 크기를 지정 가능 (공백 구분). 예: DB_SIZES="100GB 280GB 500GB 1TB"
# DB_SIZES="${DB_SIZES:-100GB 500GB 1TB}"
DB_SIZES="${DB_SIZES:-280GB}"

# ===== 유틸 =====
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
require_root_for_drop() {
  if [[ ! -w /proc/sys/vm/drop_caches ]]; then
    echo "[ERROR] Need root to drop caches." >&2
    exit 1
  fi
}
drop_caches() { sync; echo 3 > /proc/sys/vm/drop_caches; }
drop_caches_with_log() {
  local label="$1"
  log "Dropping caches… (cache_size=$label)"
  drop_caches
}


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
  # $1 = step dir, $2 = name, $3 = command string
  local dir="$1" name="$2" cmd="$3"
  mkdir -p "$dir"

  log "RUN [$name]"
  # ==== BEFORE snapshot ====
  log "Snapshotting /proc stats BEFORE → $dir"
  snap_proc_stats "$dir" "BEFORE"

  # ✅ 실행 직전 cache drop
  drop_caches_with_log "pre_${name}"

  set +e
  if [[ -n "$CMD_PREFIX" ]]; then
    bash -lc "$CMD_PREFIX $cmd" 2>&1 \
      | sed -u -e 's/\r/\n/g' \
               -e '/^\.\.\. finished [0-9]\+ ops *$/d' \
               -e '/^[[:space:]]*$/d' >"$dir/stdout.log"
  else
    bash -lc "$cmd" 2>&1 \
      | sed -u -e 's/\r/\n/g' \
               -e '/^\.\.\. finished [0-9]\+ ops *$/d' \
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

  save_summary "$dir"
}

# 원본 로그에서 핵심 라인만 추출 (read/latency 요약 위주 + sstables는 전체를 보통 확인)
save_summary() {
  local dir="$1"
  awk '/ops\/sec|Micros\/op|micros\/op|percentile|P50|P95|P99|MB\/s/ { print }' \
    "$dir/stdout.log" > "$dir/summary.txt" || true
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

mk_mixgraph_cmd() {
  local db="$1" num="$2"
  cat <<CMD
$FILL_DB_BENCH_BIN \
  --benchmarks="mixgraph" \
  --db="$db" \
  --use_existing_db=true \
  --num=$num \
  --key_size=24 \
  --value_size=1000 \
  --compression_type=none \
  --use_direct_io_for_flush_and_compaction=true \
  --use_direct_reads=true \
  --bytes_per_sync=0 \
  --seed=1724572420 \
  --bloom_bits=10 \
  --statistics=1 \
  --cache_size=34359738368 \
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
  --mix_get_ratio=0.83 \
  --mix_put_ratio=0.14 \
  --mix_seek_ratio=0.03 \
  --sine_mix_rate_interval_milliseconds=5000 \
  --sine_a=1000 \
  --sine_b=0.000073 \
  --sine_d=4500
CMD
}

# warmup (db, num)
mk_warmup_cmd() {
  local db="$1" num="$2"
  cat <<CMD
$FILL_DB_BENCH_BIN \
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
$READ_DB_BENCH_BIN \
  --benchmarks=sstables \
  --use_existing_db=true \
  --db="$db"
CMD
}

# ===== 메인 루프 =====
require_root_for_drop

for size in $DB_SIZES; do
  NUM=$(calc_num_for_size "$size")
  if [[ "$NUM" == "ERR" ]]; then
    log "잘못된 DB_SIZES 값: $size"; exit 1
  fi
  log ">>> DB Size=$size → num=$NUM"

  for i in $(seq -w 1 "$ITER"); do
    ts="$(date +%Y%m%d_%H%M%S)"
    run_dir="${OUT_ROOT}/run_${size}_${i}_${ts}"
    DB_DIR_RUN="${DB_ROOT}/${DB_NAME_PREFIX}_${size}_${i}_${ts}"

    iter_idx=$((10#$i - 1))

    # same seed
    fill_seed=$FILL_SEED_BASE

    # diff_seed
    # fill_seed=$((FILL_SEED_BASE + iter_idx))

    mkdir -p "$run_dir"
    log "=== Iter $i | DB: $DB_DIR_RUN | fill_seed: $fill_seed ==="

    # (1) fillrandom
    run_and_save "$run_dir/01_fillrandom" "fillrandom" \
      "$(mk_fillrandom_cmd "$DB_DIR_RUN" "$fill_seed" "$NUM")"

    # (2) mixgraph ✅ 추가
    run_and_save "$run_dir/02_mixgraph" "mixgraph" \
      "$(mk_mixgraph_cmd "$DB_DIR_RUN" "$NUM")"

    # (3) readrandom warmup
    run_and_save "$run_dir/03_warmup" "readrandom_warmup" \
      "$(mk_warmup_cmd "$DB_DIR_RUN" "$NUM")"

    # (5) sstables 출력
    run_and_save "$run_dir/05_sstables" "sstables_dump" \
      "$(mk_sstables_cmd "$DB_DIR_RUN")"

    log "Completed iteration $i for size=$size → $run_dir"
  done
done

log "All iterations done for sizes: $DB_SIZES (steps: fillrandom → warmup → sstables)."
