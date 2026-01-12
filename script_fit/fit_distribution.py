import argparse
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import kstest
from typing import Dict, Any, List, Tuple, Optional
import warnings

# 경고 메시지 필터링
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
# 1. 데이터 유틸리티 (Data Utils)
# -----------------------------------------------------------------------------

def ensure_positive(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """데이터 양수 보정"""
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    return np.maximum(x, eps)


def make_hist_density(x: np.ndarray, bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """히스토그램 밀도 계산"""
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return centers, hist, widths


# [RMSE 관련 유틸 주석 처리]
# def rmse_score(y_true: np.ndarray, y_pred: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
#     """Weighted RMSE 계산"""
#     y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
#     diff2 = (y_true - y_pred) ** 2
#     if weights is None:
#         return float(np.sqrt(np.mean(diff2)))
#     w = np.asarray(weights, dtype=float)
#     w = w / np.sum(w)
#     return float(np.sqrt(np.sum(w * diff2)))


# -----------------------------------------------------------------------------
# 2. 데이터 계산 로직 (Gap & Key Density)
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
    # 컬럼 확인 및 변환
    cols = ["min_key", "max_key", "entry_n"]
    for c in cols:
        sub[c] = pd.to_numeric(sub.get(c), errors="coerce")

    sub = sub.dropna(subset=cols)
    sub = sub[sub["entry_n"] > 0] # entry가 0인 경우 제외
    
    if len(sub) == 0: return np.asarray([], dtype=float)

    # Key Density = Key Range / Entry Count
    key_range = (sub["max_key"].to_numpy(dtype=float) - sub["min_key"].to_numpy(dtype=float) + 1.0)
    entry_n = sub["entry_n"].to_numpy(dtype=float)

    kd = key_range / entry_n
    
    # 유효성 검사 (inf, nan, 음수 제거)
    kd = kd[np.isfinite(kd)]
    kd = kd[kd >= 0]
    return kd.astype(float, copy=False)


def build_samples_from_dir(input_dir: str, pattern: str, target_mode: str) -> pd.DataFrame:
    """디렉토리 내 파일들로부터 데이터(Gap or KD) 추출"""
    in_dir = Path(input_dir)
    files = sorted(in_dir.glob(pattern))
    if not files: raise ValueError(f"파일 없음: {in_dir}/{pattern}")

    logger.info(f"Target Mode: {target_mode.upper()}")
    logger.info(f"{len(files)}개의 파일 처리 시작...")
    
    rows = []
    
    # 모드에 따른 필수 컬럼 정의
    required_cols = {"level", "min_key", "max_key"}
    if target_mode == "kd":
        required_cols.add("entry_n")

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
            # 필수 컬럼 체크
            if not required_cols.issubset(df.columns): continue

            df["level"] = pd.to_numeric(df["level"], errors="coerce")
            df = df.dropna(subset=["level"])
            db_name = csv_path.stem

            for lvl, g in df.groupby("level"):
                # 모드에 따라 계산 함수 분기
                if target_mode == "gap":
                    values = compute_level_gaps(g)
                elif target_mode == "kd":
                    values = compute_key_density(g)
                else:
                    values = np.array([])

                if values.size > 0:
                    for val in values:
                        # 분석 함수 통일을 위해 값 컬럼명을 'value'로 통일
                        rows.append({"db": db_name, "level": int(lvl), "value": float(val)})
        except: continue

    if not rows: raise ValueError("데이터 추출 실패 (컬럼명 확인 필요)")
    out = pd.DataFrame(rows)
    logger.info(f"총 {len(out)}개 샘플 추출 완료.")
    return out


# -----------------------------------------------------------------------------
# 3. 모델링: MLE + K-S Test
# -----------------------------------------------------------------------------

def fit_best_distribution_mle(x: np.ndarray, top_k: int = 1) -> List[Dict[str, Any]]:
    x_safe = ensure_positive(x)
    if len(x_safe) < 20: return []

    candidate_dists = [
        "expon",        # 지수 분포
        "lognorm",      # 로그 정규
        "gamma",        # 감마
        "weibull_min",  # 와이블
        "pareto",       # 파레토
        "fisk",         # 로그-로지스틱
        "uniform"       # 균등 분포
    ]    
    results = []
    
    for dist_name in candidate_dists:
        try:
            dist = getattr(st, dist_name)
            
            # 1. Fit (MLE)
            params = dist.fit(x_safe, floc=0)
            
            # 2. BIC
            log_lik = np.sum(dist.logpdf(x_safe, *params))
            if not np.isfinite(log_lik): continue
            
            bic = len(params) * np.log(len(x_safe)) - 2 * log_lik

            # 3. K-S Test (Goodness-of-Fit)
            D_stat, p_value = kstest(x_safe, dist_name, args=params)

            results.append({
                "dist": dist_name, 
                "bic": bic, 
                "params": params,
                "ks_stat": D_stat,
                "ks_p": p_value
            })
        except: continue

    results.sort(key=lambda r: r["bic"])
    return results[:top_k]


# -----------------------------------------------------------------------------
# 4. 모델링: RMSE (주석 처리됨)
# -----------------------------------------------------------------------------

# [RMSE 모델 함수 및 피팅 로직 주석 처리]
# def model_power_law(x, a, b): return a * np.power(x, b)
# def model_exponential(x, a, b): return a * np.exp(b * x)
# ... (생략) ...
# def fit_rmse_models_visual(x: np.ndarray, bins: int = 60):
#     ... (생략) ...
#     return results[:1]


# -----------------------------------------------------------------------------
# 5. 메인 실행 및 통계
# -----------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, bins: int = 60, top_k: int = 1) -> pd.DataFrame:
    summary_rows = []
    
    for (db, lvl), group_df in df.groupby(["db", "level"], sort=True):
        # 'value' 컬럼을 가져와서 분석 (Gap 또는 KD)
        x = group_df["value"].to_numpy(dtype=float)
        row = {"db": db, "level": int(lvl), "sample_count": len(x)}

        if len(x) < 30:
            summary_rows.append(row)
            continue

        # MLE (Generation + K-S Test)
        mle_results = fit_best_distribution_mle(x, top_k=top_k)
        for i, res in enumerate(mle_results):
            row[f"best_dist_{i+1}"] = res["dist"]
            row[f"best_dist_{i+1}_bic"] = round(res["bic"], 2)
            row[f"best_dist_{i+1}_params"] = str(tuple(round(p, 6) for p in res["params"]))
            row[f"best_dist_{i+1}_ks"] = round(res["ks_stat"], 4)

        # [RMSE 관련 실행 주석 처리]
        # rmse_results = fit_rmse_models_visual(x, bins=bins)
        # if rmse_results:
        #     row["visual_model"] = rmse_results[0]["model"]
        #     row["visual_rmse"] = round(rmse_results[0]["rmse"], 6)
        
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

def print_distribution_stats(df: pd.DataFrame, target_mode: str):
    if "best_dist_1" not in df.columns: return
    valid_df = df.dropna(subset=["best_dist_1"])
    total = len(valid_df)
    if total == 0: return

    print("\n" + "="*60)
    print(f"  📊  [{target_mode.upper()}] Best Distribution Summary (Total: {total})")
    print("="*60)

    print("\n[🎯 Best Model (MLE/BIC)]")
    for name, count in valid_df["best_dist_1"].value_counts().items():
        print(f"  • {name:<15} : {(count/total)*100:5.1f}%  ({count}건)")

    if "best_dist_1_ks" in valid_df.columns:
        avg_ks = valid_df["best_dist_1_ks"].mean()
        print("\n[📏 Goodness-of-Fit Summary (K-S Stat)]")
        print(f"  • Average D-statistic: {avg_ks:.4f}")
        print(f"    (Lower is better. <0.05: Excellent, >0.1: Poor)")

    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SSTable Gap or Key Density distributions.")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument("--pattern", default="*.csv", help="File pattern (default: *.csv)")
    parser.add_argument("--output", default=None, help="Output CSV filename")
    parser.add_argument("--target", choices=["gap", "kd"], default="gap", 
                        help="Analysis target: 'gap' (SSTable gaps) or 'kd' (Key Density)")
    
    args = parser.parse_args()

    # 출력 파일명 자동 설정
    if args.output is None:
        args.output = f"model_summary_{args.target}.csv"

    try:
        # 1. 데이터 로드 (모드에 따라 분기)
        df_data = build_samples_from_dir(args.input_dir, args.pattern, args.target)
        
        # 2. 분석 수행
        logger.info(f"[{args.target.upper()}] 모델 피팅 수행 중 (BIC & K-S Test)...")
        summary = run_analysis(df_data)
        
        # 3. 저장 및 출력
        summary.to_csv(args.output, index=False)
        logger.info(f"결과 저장 완료: {args.output}")
        print_distribution_stats(summary, args.target)
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")