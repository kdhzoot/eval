import re
import math
import sys
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 24,        # 기본 글자 크기
    "axes.titlesize": 24,   # 타이틀 크기
    "axes.labelsize": 24,   # 축 레이블 크기
    "xtick.labelsize": 20,  # x축 눈금 크기
    "ytick.labelsize": 20,  # y축 눈금 크기
    "legend.fontsize": 14   # 범례 글자 크기
})


def parse_and_group_entries(filename):
    # 정규표현식 패턴:
    # (\d+) : id
    # :(\d+) : size
    # :(\d+) : entry_n
    # \[([^\]]+)\] : 시퀀스 범위 (min_seq .. max_seq)
    # \[([^\]]+)\] : 키 값 범위 (min_key .. max_key)
    pattern = re.compile(r"(\d+):(\d+):(\d+)\[([^\]]+)\]\[([^\]]+)\]")
    
    groups = {}       # 레벨별로 엔트리를 저장할 딕셔너리
    current_level = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 레벨 헤더 라인 처리: 예) --- level 0 --- version# 2 ---
            if line.startswith('--- level'):
                parts = line.split()
                if len(parts) >= 3:
                    current_level = parts[2]  # 예: "0", "1", ...
                    if current_level not in groups:
                        groups[current_level] = []
                continue

            # 데이터 라인 파싱
            m = pattern.search(line)
            if m:
                id = int(m.group(1))
                size = int(m.group(2))
                entry_n = int(m.group(3))
                seq_range = m.group(4)    # "min_seq .. max_seq"
                key_range = m.group(5)    # "min_key .. max_key"

                seq_parts = [part.strip() for part in seq_range.split("..")]
                key_parts = [part.strip() for part in key_range.split("..")]

                try:
                    min_seq = int(seq_parts[0])
                    max_seq = int(seq_parts[1])
                except ValueError:
                    min_seq = seq_parts[0]
                    max_seq = seq_parts[1]

                # key 값에서 앞의 8바이트 (16자리 16진수)만 추출하여 10진수로 변환
                try:
                    min_key_hex = key_parts[0][:16]
                    max_key_hex = key_parts[1][:16]
                    min_key = int(min_key_hex, 16)
                    max_key = int(max_key_hex, 16)
                except Exception as e:
                    print("키 파싱 오류:", e)
                    min_key = key_parts[0]
                    max_key = key_parts[1]

                entry = {
                    "id": id,
                    "size": size,
                    "entry_n": entry_n,
                    "min_seq": min_seq,
                    "max_seq": max_seq,
                    "min_key": min_key,
                    "max_key": max_key
                }

                if current_level is not None:
                    groups[current_level].append(entry)
                else:
                    print("경고: 레벨 헤더 없이 엔트리 발견:", line)
            else:
                pass
                # print("파싱 실패:", line)
    return groups

def calculate_key_density(groups):
    """
    레벨별로 그룹화된 엔트리 딕셔너리를 받아, 각 엔트리에서
    (max_key - min_key)를 entry_n으로 나눈 값을 key density로 계산하여
    엔트리에 "key_density" 항목으로 추가합니다.
    
    만약 entry_n이 0이면 key_density는 None으로 설정합니다.
    """
    for level, entries in groups.items():
        for entry in entries:
            entry_n = entry.get("entry_n", 0)
            if entry_n:
                key_range = entry["max_key"] - entry["min_key"]
                entry["key_density"] = key_range / entry_n
            else:
                entry["key_density"] = None
    return groups

def compute_key_density_stats(groups):
    """
    레벨별로 그룹화된 엔트리 딕셔너리를 받아, 각 레벨의 key density 평균, 표준편차와 sst 파일 갯수를 계산합니다.
    결과는 {레벨: {"mean": 평균, "stddev": 표준편차, "count": 파일갯수}} 형식의 딕셔너리로 반환합니다.
    """
    stats = {}
    for level, entries in groups.items():
        # sst 파일 갯수
        count = len(entries)
        
        # None이 아닌 key_density 값들만 모으기
        densities = [entry["key_density"] for entry in entries if entry["key_density"] is not None]
        
        if densities:
            mean_density = sum(densities) / len(densities)
            variance = sum((d - mean_density) ** 2 for d in densities) / len(densities)
            stddev_density = math.sqrt(variance)
        else:
            mean_density = None
            stddev_density = None
        
        stats[level] = {
            "mean": mean_density,
            "stddev": stddev_density,
            "count": count
        }
    return stats

def plot_key_span_broken_barh(groups, output_file: str):
    """
    broken_barh를 사용하여 각 SST의 (min_key, max_key) 범위를
    레벨별로 색상 블록으로 시각화.
    x축은 보기용으로 0~10000 범위로 정규화하여 표시.
    """
    # 전체 키 범위 계산 (정규화용)
    global_min = None
    global_max = None
    for entries in groups.values():
        for entry in entries:
            min_key = entry["min_key"]
            max_key = entry["max_key"]
            if isinstance(min_key, int) and isinstance(max_key, int):
                if global_min is None or min_key < global_min:
                    global_min = min_key
                if global_max is None or max_key > global_max:
                    global_max = max_key

    if global_min is None or global_max is None or global_max <= global_min:
        key_range = 1.0
        global_min = 0
    else:
        key_range = float(global_max - global_min)

    def norm(x):
        return 10000.0 * (x - global_min) / key_range

    plt.figure(figsize=(16, 6))
    height = 0.5  # 각 bar 높이 (한 눈에 보이도록 키움)

    for level_str in sorted(groups.keys(), key=lambda x: int(x)):
        entries = groups[level_str]
        if not entries:
            continue

        level = int(level_str)

        for i, entry in enumerate(entries):
            min_key = entry["min_key"]
            max_key = entry["max_key"]

            if isinstance(min_key, int) and isinstance(max_key, int):
                key_span = max_key - min_key
                if key_span <= 0:
                    continue  # 무효한 구간

                x_norm = norm(min_key)
                span_norm = norm(max_key) - x_norm
                if span_norm <= 0:
                    continue
                plt.broken_barh(
                    [(x_norm, span_norm)],
                    (level - height / 2, height),
                    facecolors='steelblue',
                    edgecolors='#3d3b3b',
                    alpha=0.9
                )

    plt.xlabel("Key space", labelpad=20)
    plt.ylabel("Level", labelpad=20)
    plt.title("Key Range Covered by SST", pad=25)

    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.xlim(0, 10000)
    plt.xticks([0, 2000, 4000, 6000, 8000, 10000])
    plt.yticks(sorted([int(k) for k in groups.keys() if groups[k]]))
    plt.tight_layout()
    plt.gca().invert_yaxis()

    plt.savefig(output_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python script.py <입력파일명>")
        sys.exit(1)

    filename = sys.argv[1]  # 명령행에서 입력받음

    p = Path(filename)
    # If input is a typical run directory log like .../sstables_xxx/stdout.log,
    # name output after the run directory.
    run_name = p.parent.name if p.name == "stdout.log" else p.stem
    output_png = f"{run_name}_key_range.png"

    grouped_entries = parse_and_group_entries(filename)
    grouped_entries = calculate_key_density(grouped_entries)
    remove_groups = [level for level, entries in grouped_entries.items() if not entries]

    for group in remove_groups:
        del grouped_entries[group]

    stats = compute_key_density_stats(grouped_entries)

    # 먼저 삭제할 레벨들을 수집
    for level, stat in stats.items():
        print(f"Level {level} - sst 파일 갯수: {stat['count']}, key_density 평균: {stat['mean']}, 표준편차: {stat['stddev']}")


    
    # for level, entries in grouped_entries.items():
    #     level = int(level)
    #     if(level==2):
    #         print(f"Level {level} - sst 파일 갯수: {len(entries)}")
    #         for entry in entries:
    #             print(f"  SST ID: {entry['id']}, Size: {entry['size']}, Entry Count: {entry['entry_n']}, "
    #                   f"Key Range: [{entry['min_key']} .. {entry['max_key']}], "
    #                   f"Key Density: {entry['key_density']}")
    

    plot_key_span_broken_barh(grouped_entries, output_png)
    print(f"Saved: {output_png}")


