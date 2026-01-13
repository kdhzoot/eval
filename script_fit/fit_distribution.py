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


def build_samples_from_dir(input_dir: str, pattern: str, target_mode: str) -> pd.DataFrame:
    """CSV 파일들로부터 분석 대상 데이터 추출"""
    in_dir = Path(input_dir)
    files = sorted(in_dir.glob(pattern))
    if not files: raise ValueError(f"파일 없음: {in_dir}/{pattern}")

    logger.info(f"Target Mode: {target_mode.upper()}")
    logger.info(f"{len(files)}개의 파일 로드 및 전처리 중...")
    
    rows = []
    required_cols = {"level", "min_key", "max_key"}
    if target_mode == "kd":
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

def fit_best_distribution_mle(x: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
    x_safe = ensure_positive(x)
    if len(x_safe) < 20: return [] # 데이터가 너무 적으면 스킵

    candidate_dists = [
        "expon",        # 지수 분포
        "lognorm",      # 로그 정규 분포
        "gamma",        # 감마 분포
        "weibull_min",  # 와이블 분포
        "pareto",       # 파레토 분포
        "fisk",         # 로그-로지스틱 분포
        "uniform"       # 균등 분포
    ]    
    results = []
    
    for dist_name in candidate_dists:
        try:
            dist = getattr(st, dist_name)
            
            # 1. Fit (MLE: 최대우도추정)
            params = dist.fit(x_safe, floc=0)
            
            # 2. BIC 계산 (모델 선택 기준)
            log_lik = np.sum(dist.logpdf(x_safe, *params))
            if not np.isfinite(log_lik): continue
            bic =  - 2 * log_lik

            # 3. K-S Test (참고용 통계)
            D_stat, p_value = kstest(x_safe, dist_name, args=params)
            
            # 4. Score 계산 (0~100점) - 직관적 지표
            score = calculate_fitting_score(x_safe, dist_name, params)

            results.append({
                "dist": dist_name, 
                "bic": bic, 
                "params": params,
                "ks_stat": D_stat,
                "ks_p": p_value,
                "score": score
            })
        except: continue

    # BIC가 낮은 순서대로 정렬 (가장 적합한 모델이 0번 인덱스)
    results.sort(key=lambda r: r["bic"])
    return results[:top_k]


# -----------------------------------------------------------------------------
# 4. 실행 및 요약 리포트
# -----------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    summary_rows = []
    
    # DB별, Level별로 그룹화하여 분석 수행
    for (db, lvl), group_df in df.groupby(["db", "level"], sort=True):
        x = group_df["value"].to_numpy(dtype=float)
        row = {"db": db, "level": int(lvl), "sample_count": len(x)}

        if len(x) < 30:
            summary_rows.append(row)
            continue

        # 피팅 수행
        mle_results = fit_best_distribution_mle(x, top_k=top_k)
        
        for i, res in enumerate(mle_results):
            prefix = f"top_{i+1}"
            row[prefix] = res["dist"]
            row[f"{prefix}_bic"] = round(res["bic"], 2)
            # row[f"{prefix}_params"] = str(tuple(round(p, 6) for p in res["params"]))
            # row[f"{prefix}_ks"] = round(res["ks_stat"], 4)
            # row[f"{prefix}_p"] = round(res["ks_p"], 6)
            
            # [Score 저장] 소수점 2자리까지
            # row[f"{prefix}_score"] = round(res["score"], 2)

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_distribution_stats(df: pd.DataFrame, target_mode: str):
    """분석 결과 요약 출력"""
    if "best_dist_1" not in df.columns: return
    valid_df = df.dropna(subset=["best_dist_1"])
    total = len(valid_df)
    if total == 0: return

    print("\n" + "="*60)
    print(f"  📊  [{target_mode.upper()}] Best Distribution Summary (Total: {total})")
    print("="*60)

    print("\n[🎯 Best Model Counts]")
    for name, count in valid_df["best_dist_1"].value_counts().items():
        print(f"  • {name:<15} : {(count/total)*100:5.1f}%  ({count} cases)")

    # Score 통계 출력
    if "best_dist_1_score" in valid_df.columns:
        scores = valid_df["best_dist_1_score"]
        avg_score = scores.mean()
        high_score_ratio = (len(scores[scores >= 90]) / total) * 100
        mid_score_ratio = (len(scores[(scores >= 70) & (scores < 90)]) / total) * 100
        
        print("\n[⭐ Goodness-of-Fit Score (0~100)]")
        print(f"  • Average Score    : {avg_score:.2f}점")
        print(f"  • Excellent (90+)  : {high_score_ratio:.1f}%")
        print(f"  • Good (70~90)     : {mid_score_ratio:.1f}%")
        print(f"    (점수가 높을수록 모델이 실제 데이터 분포를 완벽하게 설명함)")

    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SSTable distributions with Fitting Score.")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument("--pattern", default="*.csv", help="File pattern (default: *.csv)")
    parser.add_argument("--output", default=None, help="Output CSV filename")
    parser.add_argument("--target", choices=["gap", "kd"], default="gap", 
                        help="Analysis target: 'gap' (Gaps) or 'kd' (Key Density)")
    
    args = parser.parse_args()

    if args.output is None:
        args.output = f"model_summary_{args.target}.csv"

    try:
        # 1. Load Data
        df_data = build_samples_from_dir(args.input_dir, args.pattern, args.target)
        
        # 2. Run Analysis
        logger.info(f"[{args.target.upper()}] Calculating Best Fit & Scores...")
        summary = run_analysis(df_data)
        
        # 3. Save & Print
        summary.to_csv(args.output, index=False)
        logger.info(f"결과 저장 완료: {args.output}")
        print_distribution_stats(summary, args.target)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")