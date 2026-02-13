#!/usr/bin/env python3
"""
SSTable 데이터 분포 분석 및 최적 분포 피팅 도구.
SSTable 사이의 Gap, Key Density, Entry Count 데이터를 분석하여 최적의 통계 분포 모델을 제안합니다.

사용법:
  # 1. SSTable 사이의 Gap 분석 (기본값)
  python3 fit_distribution.py <csv_dir> --target gap

  # 2. Key Density 분석
  python3 fit_distribution.py <csv_dir> --target kd

  # 3. Entry Count 분석 (Spike/Tail 분리 분석 모드 활성)
  python3 fit_distribution.py <csv_dir> --target entry

주요 옵션:
  --target {gap, kd, entry} : 분석 대상 선택
  --pattern "*.csv"         : 파일 패턴 지정
  --output "result.csv"     : 결과 저장 파일명 지정
"""

import argparse
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
from scipy.stats import kstest
from typing import Dict, Any, List, Tuple, Optional
import warnings

# 경고 메시지 필터링 (불필요한 RuntimeWarning 숨김)
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 로깅 설정
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 1. 데이터 유틸리티 & 점수 계산 (Data Utils & Scoring)
# -----------------------------------------------------------------------------

def ensure_positive(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """데이터 양수 보정 (로그 분포 피팅 시 에러 방지)"""
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    return np.maximum(x, eps)

def trim_both_tails_percent(
    x: np.ndarray, tail_pct: float = 0.01
) -> Tuple[np.ndarray, float, float, int]:
    """Trim both tails by percentile (e.g., tail_pct=0.01 -> keep p01 <= x <= p99)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x, np.nan, np.nan, 0
    if tail_pct <= 0.0:
        return x, np.nan, np.nan, 0

    q_low = float(np.quantile(x, tail_pct))
    q_high = float(np.quantile(x, 1.0 - tail_pct))
    mask = (x >= q_low) & (x <= q_high)
    trimmed = x[mask]
    removed = int(x.size - trimmed.size)
    return trimmed, q_low, q_high, removed


def trim_upper_iqr(
    x: np.ndarray, iqr_mult: float = 1.5
) -> Tuple[np.ndarray, float, float, float, float, int]:
    """Auto-trim upper-tail outliers with Tukey fence: Q3 + k*IQR."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x, np.nan, np.nan, np.nan, np.nan, 0
    if x.size < 4:
        return x, np.nan, np.nan, np.nan, np.nan, 0

    q1 = float(np.quantile(x, 0.25))
    q3 = float(np.quantile(x, 0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return x, q1, q3, iqr, np.nan, 0

    upper = q3 + iqr_mult * iqr
    trimmed = x[x <= upper]
    removed = int(x.size - trimmed.size)
    return trimmed, q1, q3, iqr, upper, removed


def calculate_fitting_score(data: np.ndarray, dist_name: str, params: tuple) -> float:
    """
    [핵심 기능] 모델 적합도 점수 계산 (R-squared of CDF)
    
    * 설명: sklearn.metrics.r2_score와 수학적으로 동일한 로직입니다.
    * 리턴: 0 ~ 100 사이의 점수 (100점에 가까울수록 완벽하게 일치함)
    """
    n = len(data)
    if n < 2: return 0.0
    
    # 1. Empirical CDF (실제 데이터의 분포)
    # 데이터를 정렬하여 각 포인트가 전체의 몇 % 위치에 있는지 계산
    x_sorted = np.sort(data)
    y_empirical = np.arange(1, n + 1) / n
    
    # 2. Theoretical CDF (모델이 예측한 분포)
    # 같은 데이터 값(x)을 넣었을 때 모델은 몇 % 위치라고 예측하는지 계산
    dist = getattr(st, dist_name)
    y_theoretical = dist.cdf(x_sorted, *params)
    
    # 3. R-squared 계산 (1 - 잔차제곱합 / 총제곱합)
    ss_res = np.sum((y_empirical - y_theoretical) ** 2)
    ss_tot = np.sum((y_empirical - np.mean(y_empirical)) ** 2)
    
    if ss_tot == 0: return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    
    # 음수가 나오면(모델이 평균보다 못하면) 0점으로 처리, 백분율 변환
    final_score = max(r2, 0.0) * 100.0
    return final_score


# -----------------------------------------------------------------------------
# 2. 데이터 추출 및 계산 로직 (SSTable Gap / Key Density)
# -----------------------------------------------------------------------------

def compute_level_gaps(level_df: pd.DataFrame) -> np.ndarray:
    """SSTable 간 Gap 계산"""
    sub = level_df.copy()
    sub["min_key"] = pd.to_numeric(sub.get("min_key"), errors="coerce")
    sub["max_key"] = pd.to_numeric(sub.get("max_key"), errors="coerce")
    
    sub = sub.dropna(subset=["min_key", "max_key"])
    if len(sub) < 2: return np.asarray([], dtype=float)

    sub = sub.sort_values(["min_key", "max_key"], kind="mergesort")
    prev_max = sub["max_key"].to_numpy(dtype=float)[:-1]
    next_min = sub["min_key"].to_numpy(dtype=float)[1:]
    
    gaps = next_min - prev_max
    return np.maximum(gaps, 0.0)


def compute_key_density(level_df: pd.DataFrame) -> np.ndarray:
    """SSTable Key Density 계산"""
    sub = level_df.copy()
    cols = ["min_key", "max_key", "entry_n"]
    for c in cols:
        sub[c] = pd.to_numeric(sub.get(c), errors="coerce")

    sub = sub.dropna(subset=cols)
    sub = sub[sub["entry_n"] > 0] 
    
    if len(sub) == 0: return np.asarray([], dtype=float)

    key_range = (sub["max_key"].to_numpy(dtype=float) - sub["min_key"].to_numpy(dtype=float) + 1.0)
    entry_n = sub["entry_n"].to_numpy(dtype=float)

    kd = key_range / entry_n
    
    kd = kd[np.isfinite(kd)]
    kd = kd[kd >= 0]
    return kd.astype(float, copy=False)


def compute_entry_counts(level_df: pd.DataFrame) -> np.ndarray:
    """SSTable Entry Count 추출 (단순 entry_n 컬럼 값)"""
    vals = pd.to_numeric(level_df.get("entry_n"), errors="coerce").dropna().to_numpy()
    return vals[vals > 0]


def build_samples_from_dir(input_dir: str, pattern: str, target_mode: str) -> pd.DataFrame:
    """CSV 파일들로부터 분석 대상 데이터 추출"""
    in_dir = Path(input_dir)
    files = sorted(in_dir.glob(pattern))
    if not files: raise ValueError(f"파일 없음: {in_dir}/{pattern}")

    logger.info(f"Target Mode: {target_mode.upper()}")
    logger.info(f"{len(files)}개의 파일 로드 및 전처리 중...")
    
    rows = []
    # 타겟별 필수 컬럼 설정
    required_cols = {"level"}
    if target_mode == "gap":
        required_cols.update(["min_key", "max_key"])
    elif target_mode == "kd":
        required_cols.update(["min_key", "max_key", "entry_n"])
    elif target_mode == "entry":
        required_cols.add("entry_n")

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
            if not required_cols.issubset(df.columns): continue

            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level"])
            db_name = csv_path.stem

            for lvl, g in df.groupby("level"):
                if target_mode == "gap":
                    values = compute_level_gaps(g)
                elif target_mode == "kd":
                    values = compute_key_density(g)
                elif target_mode == "entry":
                    values = compute_entry_counts(g)
                else:
                    values = np.array([])

                if values.size > 0:
                    for val in values:
                        rows.append({"db": db_name, "level": int(lvl), "value": float(val)})
        except: continue

    if not rows: raise ValueError("데이터 추출 실패 (CSV 컬럼명 확인 필요)")
    out = pd.DataFrame(rows)
    logger.info(f"총 {len(out)}개 샘플 포인트 추출 완료.")
    return out


# -----------------------------------------------------------------------------
# 3. 모델 피팅 및 점수 계산 (Fitting Engine)
# -----------------------------------------------------------------------------

def fit_best_distribution_mle(
    x: np.ndarray, top_k: int = 1, allow_free_loc: bool = True
) -> List[Dict[str, Any]]:
    x_safe = ensure_positive(x)
    n = len(x_safe)
    if n < 20:
        return []

    candidate_dists = [
        # "beta",
        "expon",
        # "lognorm",
        # "skewnorm",
        # "gamma",
        # "weibull_min",
        # "weibull_max",
        # "pareto",
        # "fisk",
        # "uniform"
    ]

    results = []

    for dist_name in candidate_dists:
        try:
            dist = getattr(st, dist_name)

            # 1️⃣ loc 추정 정책:
            # - KD: loc 자유추정
            # - Gap/Entry: loc=0 고정
            if allow_free_loc:
                params = dist.fit(x_safe)
            else:
                params = dist.fit(x_safe, floc=0)

            # 2️⃣ log-likelihood 계산
            log_lik = np.sum(dist.logpdf(x_safe, *params))
            if not np.isfinite(log_lik):
                continue

            # 3️⃣ 정식 BIC 계산
            k = len(params)  # 추정된 파라미터 개수
            # bic = -2 * log_lik + k * np.log(n)
            bic = -2 * log_lik

            # 4️⃣ K-S Test
            D_stat, p_value = kstest(x_safe, dist_name, args=params)

            # 5️⃣ CDF 기반 R² score
            score = calculate_fitting_score(x_safe, dist_name, params)

            results.append({
                "dist": dist_name,
                "bic": bic,
                "params": params,
                "ks_stat": D_stat,
                "ks_p": p_value,
                "score": score,
                "num_params": k
            })

        except Exception:
            continue

    # BIC 기준 정렬 (낮을수록 좋음)
    results.sort(key=lambda r: r["bic"])
    return results[:top_k]


# -----------------------------------------------------------------------------
# 4. 실행 및 요약 리포트
# -----------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, target_mode: str, top_k: int = 3) -> pd.DataFrame:
    summary_rows = []
    
    # DB별, Level별로 그룹화하여 분석 수행
    for (db, lvl), group_df in df.groupby(["db", "level"], sort=True):
        x_all = group_df["value"].to_numpy(dtype=float)
        row = {"db": db, "level": int(lvl), "sample_count": len(x_all)}

        if len(x_all) < 10:
            summary_rows.append(row)
            continue
        
        # [Entry Mode 특화: Spike/Tail 분리 분석]
        if target_mode == "entry":
            # 최빈값(Mode) 근처에서 일정 너비(window_width) 내에 가장 많은 데이터가 있는 구간 탐색
            window_width = 1000 
            x_sorted = np.sort(x_all)
            total_count = len(x_all)
            
            max_count = 0
            best_lower = x_sorted[0]
            right = 0
            for left in range(total_count):
                while right < total_count and x_sorted[right] <= x_sorted[left] + window_width:
                    right += 1
                if (right - left) > max_count:
                    max_count = right - left
                    best_lower = x_sorted[left]
            
            lower_bound, upper_bound = best_lower, best_lower + window_width
            mask_spike = (x_all >= lower_bound) & (x_all <= upper_bound)
            spike_count = np.sum(mask_spike)
            
            row["spike_min"] = round(float(lower_bound), 2)
            row["spike_max"] = round(float(upper_bound), 2)
            row["spike_ratio"] = round(spike_count / total_count, 4)
            row["spike_ref"] = round(float((lower_bound + upper_bound) / 2), 2)
            
            # Tail 데이터(나머지)에 대해서만 분포 피팅 수행
            x_to_fit = x_all[~mask_spike]
            row["tail_count"] = len(x_to_fit)
        else:
            x_to_fit = x_all

        # KD only: trim lower/upper 1% outliers before fitting
        if target_mode == "kd":
            x_to_fit, kd_trim_p01, kd_trim_p99, kd_trim_removed = trim_both_tails_percent(
                x_to_fit, tail_pct=0.01
            )
            row["kd_trim_tail_pct"] = 0.01
            row["kd_trim_p01"] = round(float(kd_trim_p01), 6) if np.isfinite(kd_trim_p01) else np.nan
            row["kd_trim_p99"] = round(float(kd_trim_p99), 6) if np.isfinite(kd_trim_p99) else np.nan
            row["kd_trim_removed"] = int(kd_trim_removed)
            row["kd_trimmed_count"] = int(len(x_to_fit))
        
        # GAP only: auto-trim upper-tail outliers before fitting (Q3 + 1.5*IQR)
        if target_mode == "gap":
            x_to_fit, gap_trim_q1, gap_trim_q3, gap_trim_iqr, gap_trim_upper, gap_trim_removed = trim_upper_iqr(
                x_to_fit, iqr_mult=1.5
            )
            row["gap_trim_iqr_mult"] = 1.5
            row["gap_trim_q1"] = round(float(gap_trim_q1), 6) if np.isfinite(gap_trim_q1) else np.nan
            row["gap_trim_q3"] = round(float(gap_trim_q3), 6) if np.isfinite(gap_trim_q3) else np.nan
            row["gap_trim_iqr"] = round(float(gap_trim_iqr), 6) if np.isfinite(gap_trim_iqr) else np.nan
            row["gap_trim_upper"] = round(float(gap_trim_upper), 6) if np.isfinite(gap_trim_upper) else np.nan
            row["gap_trim_removed"] = int(gap_trim_removed)
            row["gap_trimmed_count"] = int(len(x_to_fit))

        # 피팅 수행
        if len(x_to_fit) >= 20:
            mle_results = fit_best_distribution_mle(
                x_to_fit,
                top_k=top_k,
                allow_free_loc=(target_mode == "kd" or target_mode == "gap"),
            )
            for i, res in enumerate(mle_results):
                prefix = f"top_{i+1}"
                row[prefix] = res["dist"]
                row[f"{prefix}_bic"] = round(res["bic"], 2)
                row[f"{prefix}_params"] = str(tuple(round(p, 6) for p in res["params"]))
                row[f"{prefix}_score"] = round(res["score"], 2)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_distribution_stats(df: pd.DataFrame, target_mode: str):
    """분석 결과 요약 출력"""
    if target_mode == "entry" and "spike_ratio" in df.columns:
        avg_spike_ratio = df["spike_ratio"].mean()
        print("\n" + "="*60)
        print(f"  📌  [{target_mode.upper()}] Spike Range Analysis (Concentration)")
        print(f"  • Average Spike Ratio: {avg_spike_ratio*100:.1f}%")
        print("  (데이터의 상당수가 특정 구간(예: 64K)에 집중되어 있음)")
        print("="*60)

    if "top_1" not in df.columns: return
    valid_df = df.dropna(subset=["top_1"])
    total = len(valid_df)
    if total == 0: return

    print(f"\n[🎯 Best Model Counts for {'Tail Data' if target_mode=='entry' else 'Data'}]")
    for name, count in valid_df["top_1"].value_counts().items():
        print(f"  • {name:<15} : {(count/total)*100:5.1f}%  ({count} cases)")

    # Score 통계 출력
    if "top_1_score" in valid_df.columns:
        scores = valid_df["top_1_score"]
        avg_score = scores.mean()
        high_score_ratio = (len(scores[scores >= 90]) / total) * 100
        mid_score_ratio = (len(scores[(scores >= 70) & (scores < 90)]) / total) * 100
        
        print("\n[⭐ Goodness-of-Fit Score (0~100)]")
        print(f"  • Average Score    : {avg_score:.2f}점")
        print(f"  • Excellent (90+)  : {high_score_ratio:.1f}%")
        print(f"  • Good (70~90)     : {mid_score_ratio:.1f}%")
        print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SSTable distributions with Fitting Score.")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument("--pattern", default="*.csv", help="File pattern (default: *.csv)")
    parser.add_argument("--output", default=None, help="Output CSV filename")
    parser.add_argument("--target", choices=["gap", "kd", "entry"], default="gap", 
                        help="Analysis target: 'gap' (Gaps), 'kd' (Key Density), 'entry' (Entry Count)")
    
    args = parser.parse_args()

    if args.output is None:
        args.output = f"model_summary_{args.target}.csv"

    try:
        # 1. Load Data
        df_data = build_samples_from_dir(args.input_dir, args.pattern, args.target)
        
        # 2. Run Analysis
        logger.info(f"[{args.target.upper()}] Calculating Best Fit & Scores...")
        summary = run_analysis(df_data, args.target)
        
        # 3. Save & Print
        summary.to_csv(args.output, index=False)
        logger.info(f"결과 저장 완료: {args.output}")
        print_distribution_stats(summary, args.target)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
