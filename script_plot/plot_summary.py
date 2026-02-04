#!/usr/bin/env python3
"""
Plot metrics from a summary CSV file for specific run indices.

Usage:
  python3 plot_summary.py <csv_file> --indices 1 2 3 --metrics L0_hit L1_hit L2_hit L3_hit L4_hit L5_hit
"""

import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from summary CSV.')
    parser.add_argument('csv_file', help='Path to the summary CSV file')
    parser.add_argument('--indices', nargs='+', type=int, help='List of run_index values to plot')
    parser.add_argument('--labels', nargs='+', help='List of legend labels to match indices 1:1')
    parser.add_argument('--metrics', nargs='+', help='List of columns/metrics to plot (e.g., L0_hit L1_hit ...)')
    parser.add_argument('--output', default='summary_plot.png', help='Output filename')
    
    args = parser.parse_args()

    # Load data
    path = Path(args.csv_file)
    if not path.exists():
        print(f"Error: {args.csv_file} not found.")
        sys.exit(1)
        
    df = pd.read_csv(args.csv_file)
    
    # Filter by indices if provided
    if args.indices:
        # Re-index to ensure the order matches the input indices
        df = df.set_index('run_index').reindex(args.indices).reset_index()
        # Remove any indices that didn't exist in CSV (resulting in NaN rows)
        df = df.dropna(subset=['run_id'])
        if df.empty:
            print(f"Error: No data found for run_indices {args.indices}")
            sys.exit(1)
    
    # Check if labels match indices
    labels = args.labels
    if labels and len(labels) != len(df):
        print(f"Warning: Number of labels ({len(labels)}) does not match number of found indices ({len(df)}).")
        # Adjust labels if shorter, or use only what we need
        if len(labels) < len(df):
            labels.extend([f"Run {idx}" for idx in df['run_index'][len(labels):]])
        else:
            labels = labels[:len(df)]
    elif not labels:
        labels = [f"Run {idx} ({row['run_id']})" for _, row in df.iterrows()]

    # Default metrics if none provided (Level hits)
    metrics = args.metrics
    if not metrics:
        # Default to Level hits if nothing specified
        metrics = [col for col in df.columns if col.endswith('_hit') and col.startswith('L')]
        if not metrics:
            metrics = ['ops_sec', 'micros_op']
    elif len(metrics) == 1:
        # Helper keywords
        if metrics[0] == 'hits':
            metrics = [col for col in df.columns if col.endswith('_hit') and (col.startswith('L') or 'bloom' in col)]
        elif metrics[0] == 'cache':
            metrics = [col for col in df.columns if 'cache' in col]
        elif metrics[0] == 'bloom':
            metrics = [col for col in df.columns if 'bloom' in col]
        # Otherwise use it as a single column name or a pattern? 
        # For now, if it's in columns, use it.
        elif metrics[0] in df.columns:
            pass 
        else:
            # Try to match as a prefix/pattern
            matched = [col for col in df.columns if metrics[0] in col]
            if matched:
                metrics = matched
            else:
                print(f"Warning: Metric '{metrics[0]}' not found. Using as is.")
            
    # Prepare data for plotting
    plot_df = df.set_index('run_index')[metrics].fillna(0)
    
    # 데이터가 모든 Run에서 0인 메트릭(항항목) 제거
    non_zero_metrics = plot_df.columns[(plot_df != 0).any(axis=0)]
    if len(non_zero_metrics) < len(metrics):
        removed = set(metrics) - set(non_zero_metrics)
        print(f"Removing zero-value metrics: {removed}")
        metrics = list(non_zero_metrics)
        plot_df = plot_df[metrics]
    
    if plot_df.empty or len(metrics) == 0:
        print("Error: No non-zero metrics to plot.")
        sys.exit(1)
        
    # Prepare metrics display names for x-axis
    # L1_hit -> L1
    display_metrics = []
    for m in metrics:
        if 'L' in m and 'hit' in m:
            level = m.split('_')[0] # e.g., L1
            display_metrics.append(level)
        else:
            display_metrics.append(m.replace('_', ' '))

    # 폰트 설정 (Calibri가 없으면 Liberation Sans 또는 sans-serif 사용)
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    target_font = 'sans-serif'
    for candidate in ['Calibri', 'Liberation Sans', 'Arial', 'Ubuntu']:
        if candidate in available_fonts:
            target_font = candidate
            break
    
    plt.rcParams['font.family'] = target_font
    fig, ax = plt.subplots(figsize=(12, 7))
    # 네모 상자(plotting area)의 여백을 수동으로 조절하여 상자 크기를 줄임
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.2)
    
    ax.set_facecolor('#ffffff') # 명확한 대비를 위해 흰색 배경 사용
    ax.grid(True, axis='y', linestyle='--', color='#e0e0e0', zorder=0) # 점선 그리드
    
    n_metrics = len(metrics)
    n_runs = len(df)
    x_indices = np.arange(n_metrics)
    total_width = 0.8
    # bar_width는 실제 막대의 너비, slot_width는 각 막대에게 할당된 공간의 너비
    slot_width = total_width / n_runs
    bar_width = slot_width * 0.75 # 막대 사이에 공간을 주기 위해 너비 축소
    
    # 컬러 팔레트 설정 (1-5번 #8FAADC 통일, 6번 #F4B183 강조)
    if n_runs == 6:
        colors = ['#8FAADC'] * 5
        colors.append('#F4B183') # 6번째 강조 색상 수정
    else:
        # 6개가 아닐 경우 기본 tab10 팔레트 사용
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(n_runs)]
    
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        # 범례 그룹화 (6개 데이터일 때 1~5번은 하나로 합침)
        if n_runs == 6:
            if i == 0:
                current_label = "fillrandom #1~#5"
            elif i == 5:
                current_label = labels[5]
            else:
                current_label = "_nolegend_" # 범례에서 제외
        else:
            current_label = labels[i]
        
        # 각 막대를 옆으로 배치하기 위한 offset 계산
        # slot_width를 기준으로 중앙 정렬 유지
        offset = (i - (n_runs - 1) / 2) * slot_width
        ax.bar(x_indices + offset, row.values, width=bar_width, 
               edgecolor='black', linewidth=0.5, # 구분감을 위해 얇은 검은색 테두리
               color=colors[i], label=current_label, zorder=3)

    # Y축 상단 여유 공간 확보 (최댓값의 1.2배 정도로 설정)
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.25)

    # 제목 제거 (user request)
    # ax.set_title(...)

    # 축 제목 설정 (26pt) - labelpad를 통해 눈금과의 간격 확보
    ax.set_xlabel('Hit in L#', fontsize=26, labelpad=20)
    ax.set_ylabel('Count', fontsize=26, labelpad=25)
    
    # x축 눈금 및 라벨 설정 (18pt, 0도)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(display_metrics, rotation=0, fontsize=18)
    
    # y축 눈금 폰트 사이즈 조절 (18pt)
    ax.tick_params(axis='y', labelsize=18)
    
    # y축 데이터 포맷팅 (350000 -> 350k)
    def k_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:g}k'
        return f'{x:g}'
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))
    
    # 범례 설정 (기존 위치로 복구, 마커를 정사각형으로 설정)
    ax.legend(fontsize=14, handlelength=1.0, handleheight=1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # tight_layout 대신 수동 여백(subplots_adjust)을 사용하기 위해 제거/주석 처리
    # plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
