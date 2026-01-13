import argparse
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
from pathlib import Path
from scipy.stats import kstest
from typing import List, Dict, Any  # <--- 이 부분이 빠져 있었습니다. 수정했습니다.
import warnings

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 분석할 후보 분포 리스트
CANDIDATE_DISTS = [
    "expon",        # 지수 분포
    "lognorm",      # 로그 정규 분포
    "gamma",        # 감마 분포
    "weibull_min",  # 와이블 분포
    "pareto",       # 파레토 분포
    "uniform",      # 균등 분포
]

# -----------------------------------------------------------------------------
# 1. 유틸리티 함수
# -----------------------------------------------------------------------------

def ensure_positive(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """데이터 양수 보정"""
    x = np.asarray(x, dtype=float)
    x = x[x >= 0]
    return np.maximum(x, eps)

def calculate_score(data: np.ndarray, dist_name: str, params: tuple) -> float:
    """
    적합도 점수 계산 (R-squared 기반, 0~100점)
    """
    n = len(data)
    if n < 2: return 0.0
    
    x_sorted = np.sort(data)
    y_empirical = np.arange(1, n + 1) / n
    
    dist = getattr(st, dist_name)
    try:
        y_theoretical = dist.cdf(x_sorted, *params)
    except:
        return 0.0
    
    ss_res = np.sum((y_empirical - y_theoretical) ** 2)
    ss_tot = np.sum((y_empirical - np.mean(y_empirical)) ** 2)
    
    if ss_tot == 0: return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return max(r2, 0.0) * 100.0

# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------

def compute_level_gaps(level_df: pd.DataFrame) -> np.ndarray:
    sub = level_df.dropna(subset=["min_key", "max_key"])
    if len(sub) < 2: return np.array([])
    
    sub["min_key"] = pd.to_numeric(sub["min_key"], errors="coerce")
    sub["max_key"] = pd.to_numeric(sub["max_key"], errors="coerce")
    sub = sub.dropna().sort_values(["min_key", "max_key"])
    
    gaps = sub["min_key"].values[1:] - sub["max_key"].values[:-1]
    return np.maximum(gaps, 0.0)

def compute_key_density(level_df: pd.DataFrame) -> np.ndarray:
    sub = level_df.dropna(subset=["min_key", "max_key", "entry_n"])
    cols = ["min_key", "max_key", "entry_n"]
    for c in cols: sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    sub = sub[sub["entry_n"] > 0]
    if len(sub) == 0: return np.array([])
    
    kd = (sub["max_key"].values - sub["min_key"].values + 1.0) / sub["entry_n"].values
    return kd[kd >= 0]

def load_grouped_data(input_dir: str, pattern: str, target_mode: str) -> List[dict]:
    """CSV 파일들을 읽어 (DB, Level)별 데이터 그룹 리스트 생성"""
    in_dir = Path(input_dir)
    files = sorted(in_dir.glob(pattern))
    
    groups = []
    
    logger.info(f"Target: {target_mode.upper()} | {len(files)}개 파일 로드 중...")
    
    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
            if not {"level", "min_key", "max_key"}.issubset(df.columns): continue
            
            db_name = csv_path.stem
            
            for lvl, g in df.groupby("level"):
                if target_mode == "gap":
                    vals = compute_level_gaps(g)
                elif target_mode == "kd":
                    vals = compute_key_density(g)
                else:
                    vals = np.array([])
                
                # 데이터가 너무 적으면(예: 10개 미만) Fitting 신뢰도가 낮으므로 제외할 수도 있음
                if len(vals) >= 10:
                    groups.append({
                        "id": f"{db_name}_L{int(lvl)}",
                        "data": vals
                    })
        except: continue
        
    return groups

# -----------------------------------------------------------------------------
# 3. 핵심 로직: 모든 그룹에 대해 모든 분포 평가
# -----------------------------------------------------------------------------

def evaluate_all_groups(groups: List[dict]) -> pd.DataFrame:
    """
    각 그룹별로 모든 후보 분포를 Fitting하고 점수를 기록
    """
    if not groups:
        return pd.DataFrame()

    logger.info(f"총 {len(groups)}개의 그룹에 대해 개별 Fitting 수행 중...")
    
    # 결과를 저장할 리스트
    # 구조: [{'group': 'db_L1', 'expon_score': 90, 'gamma_score': 95, ...}, ...]
    all_scores = []

    for idx, grp in enumerate(groups):
        data = ensure_positive(grp['data'])
        row = {"group_id": grp['id'], "sample_count": len(data)}
        
        # 진행 상황 로깅 (너무 자주 찍지 않음)
        if idx % 10 == 0:
            logger.debug(f"Processing group {idx+1}/{len(groups)}...")

        best_score_in_group = -1
        best_dist_in_group = None

        for dist_name in CANDIDATE_DISTS:
            try:
                dist = getattr(st, dist_name)
                # Fit
                params = dist.fit(data)
                # Score
                score = calculate_score(data, dist_name, params)
                
                row[f"{dist_name}_score"] = score
                
                # 이 그룹에서의 승자 판별용
                if score > best_score_in_group:
                    best_score_in_group = score
                    best_dist_in_group = dist_name
                    
            except:
                row[f"{dist_name}_score"] = 0.0
        
        # 이 그룹의 Best 분포 기록
        row["winner_dist"] = best_dist_in_group
        all_scores.append(row)

    return pd.DataFrame(all_scores)


def print_global_leaderboard(df_scores: pd.DataFrame):
    """
    전체 그룹 결과를 집계하여 '최종 우승 분포' 랭킹 출력
    """
    if df_scores.empty:
        logger.error("분석할 데이터가 없습니다.")
        return

    total_groups = len(df_scores)
    
    # 집계용 리스트
    leaderboard = []
    
    for dist_name in CANDIDATE_DISTS:
        col_name = f"{dist_name}_score"
        if col_name not in df_scores.columns: continue
        
        scores = df_scores[col_name]
        
        # 1. 평균 점수 (Average Score)
        avg_score = scores.mean()
        
        # 2. 우승 횟수 (Win Count) - 해당 분포가 그룹 내 1등을 한 횟수
        win_count = len(df_scores[df_scores["winner_dist"] == dist_name])
        win_rate = (win_count / total_groups) * 100
        
        # 3. 안정성 (90점 이상인 비율)
        excellent_rate = (len(scores[scores >= 90]) / total_groups) * 100
        
        leaderboard.append({
            "Distribution": dist_name,
            "Avg Score": avg_score,
            "Win Rate (%)": win_rate,
            "Excellent Fit (>90pt)": excellent_rate
        })
    
    # 평균 점수 내림차순 정렬
    df_leaderboard = pd.DataFrame(leaderboard).sort_values(by="Avg Score", ascending=False)
    
    # --- 터미널 출력 ---
    print("\n" + "="*80)
    print(f" 🏆 FINAL RESULT: Global Best Distribution (Based on {total_groups} Groups)")
    print("="*80)
    print(f" * 해석: 'Avg Score'가 가장 높은 분포가 모든 그룹을 통틀어 가장 범용적인 모델입니다.")
    print("-" * 80)
    
    # 포맷팅하여 출력
    print(f"{'Rank':<5} {'Distribution':<15} {'Avg Score':<12} {'Win Rate(1st)':<15} {'Excellent Fit(%)':<15}")
    print("-" * 80)
    
    for rank, (idx, row) in enumerate(df_leaderboard.iterrows(), 1):
        name = row['Distribution']
        avg = row['Avg Score']
        win = row['Win Rate (%)']
        exc = row['Excellent Fit (>90pt)']
        
        # 1등은 강조 표시
        prefix = "⭐️" if rank == 1 else "  "
        print(f"{prefix:<4} {name:<15} {avg:6.2f} pt     {win:5.1f} %        {exc:5.1f} %")
        
    print("="*80 + "\n")


# -----------------------------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify the single best distribution across all groups.")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument("--pattern", default="*.csv", help="File pattern")
    parser.add_argument("--target", choices=["gap", "kd"], default="gap", help="Target metric")
    
    args = parser.parse_args()

    try:
        # 1. 데이터 로드 (그룹별 리스트)
        groups = load_grouped_data(args.input_dir, args.pattern, args.target)
        
        if not groups:
            logger.error("데이터 로드 실패: 유효한 그룹이 없습니다.")
            exit(1)
            
        # 2. 모든 그룹 개별 피팅 & 점수 계산
        df_results = evaluate_all_groups(groups)
        
        # 3. 집계 및 최종 랭킹 출력
        print_global_leaderboard(df_results)
        
        # (옵션) 상세 결과를 파일로 저장하고 싶다면 아래 주석 해제
        # df_results.to_csv(f"fitting_details_{args.target}.csv", index=False)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")