#!/usr/bin/env python3
import sys
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def parse_iostat(file_path, device_name="md0"):
    if not os.path.exists(file_path):
        return None
    
    utils = []
    read_mb = []
    write_mb = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    headers = None
    for line in lines:
        line = line.strip()
        if not line or "Linux" in line or "avg-cpu" in line:
            continue
        if line.startswith("Device") or "Device" in line:
            headers = line.split()
            continue
        
        if headers:
            parts = line.split()
            if parts and parts[0] == device_name:
                try:
                    data = dict(zip(headers, parts))
                    util_val = float(data.get('%util', 0))
                    
                    # Read Rate
                    r_val = 0.0
                    if 'rMB/s' in data:
                        r_val = float(data['rMB/s'])
                    elif 'rkB/s' in data:
                        r_val = float(data['rkB/s']) / 1024.0
                    
                    # Write Rate
                    w_val = 0.0
                    if 'wMB/s' in data:
                        w_val = float(data['wMB/s'])
                    elif 'wkB/s' in data:
                        w_val = float(data['wkB/s']) / 1024.0
                    
                    utils.append(util_val)
                    read_mb.append(r_val)
                    write_mb.append(w_val)
                except (ValueError, KeyError):
                    continue
    
    if not utils:
        return None
        
    return {
        "avg_util": sum(utils) / len(utils),
        "max_util": max(utils),
        "avg_read_mb": sum(read_mb) / len(read_mb),
        "avg_write_mb": sum(write_mb) / len(write_mb),
        "series_util": utils,
        "series_read": read_mb,
        "series_write": write_mb
    }

def parse_mpstat(file_path):
    if not os.path.exists(file_path):
        return None
    
    usr = []
    sys_list = []
    iowait = []
    total_util = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if "all" in line and "%usr" not in line:
                parts = line.split()
                try:
                    all_idx = parts.index("all")
                    u = float(parts[all_idx + 1])
                    s = float(parts[all_idx + 3])
                    i = float(parts[all_idx + 4])
                    idle = float(parts[-1])
                    
                    usr.append(u)
                    sys_list.append(s)
                    iowait.append(i)
                    total_util.append(100.0 - idle)
                except (ValueError, IndexError):
                    continue
                
    if not usr:
        return None
        
    return {
        "avg_usr": sum(usr) / len(usr),
        "avg_sys": sum(sys_list) / len(sys_list),
        "avg_iowait": sum(iowait) / len(iowait),
        "avg_total_util": sum(total_util) / len(total_util),
        "series_total_util": total_util,
        "series_iowait": iowait
    }

def parse_stdout(file_path):
    if not os.path.exists(file_path):
        return None
    
    throughput = 0.0
    with open(file_path, 'r') as f:
        content = f.read()
        match = re.search(r';\s+([\d\.]+)\s+MB/s', content)
        if match:
            throughput = float(match.group(1))
            
    return throughput

def create_plots(run_dir, iostat_data, mpstat_data):
    if not iostat_data and not mpstat_data:
        return

    # --- Plot Style Configuration ---
    STYLE = {
        'fonts': ['Calibri', 'Liberation Sans', 'Arial', 'Ubuntu'],
        'fs_label': 22,        # Font size for axis labels
        'fs_title': 24,        # Font size for subplot titles
        'fs_tick': 18,         # Font size for axis ticks
        'fs_legend': 14,       # Font size for legend
        'label_pad': 10,       # Padding for axis labels
        'hspace': 0.4,         # Height space between subplots
        'fig_width': 12,       # Combined figure width
        'fig_height': 10,      # Combined figure height
        'grid_color': '#e0e0e0',
    }

    # Font setup
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    target_font = 'sans-serif'
    for candidate in STYLE['fonts']:
        if candidate in available_fonts:
            target_font = candidate
            break
    plt.rcParams['font.family'] = target_font

    fig = plt.figure(figsize=(STYLE['fig_width'], STYLE['fig_height']))
    fig.subplots_adjust(hspace=STYLE['hspace'])

    # Subplot 1: Disk Bandwidth (Read + Write)
    ax1 = fig.add_subplot(2, 1, 1)
    if iostat_data:
        ax1.set_facecolor('#ffffff')
        ax1.grid(True, axis='y', linestyle='--', color=STYLE['grid_color'], zorder=0)

        x = range(len(iostat_data['series_read']))
        read_gb = [r / 1024.0 for r in iostat_data['series_read']]
        write_gb = [w / 1024.0 for w in iostat_data['series_write']]
        
        ax1.plot(x, read_gb, label='Read GB/s', color='blue', alpha=0.7, zorder=3)
        ax1.plot(x, write_gb, label='Write GB/s', color='green', alpha=0.7, zorder=3)
        
        ax1.set_ylabel('Bandwidth (GB/s)', fontsize=STYLE['fs_label'], labelpad=STYLE['label_pad'])
        ax1.set_title('Disk Bandwidth Over Time (md0)', fontsize=STYLE['fs_title'], fontweight='bold')
        ax1.tick_params(axis='both', labelsize=STYLE['fs_tick'])
        ax1.legend(fontsize=STYLE['fs_legend'])
        
        # Set y-axis to start from 0
        curr_ylim = ax1.get_ylim()
        ax1.set_ylim(0, max(curr_ylim[1], 1.0))

    # Subplot 2: CPU Utilization
    ax2 = fig.add_subplot(2, 1, 2)
    if mpstat_data:
        ax2.set_facecolor('#ffffff')
        ax2.grid(True, axis='y', linestyle='--', color=STYLE['grid_color'], zorder=0)

        x = range(len(mpstat_data['series_total_util']))
        ax2.plot(x, mpstat_data['series_total_util'], label='Total CPU Util %', color='red', zorder=3)
        ax2.plot(x, mpstat_data['series_iowait'], label='CPU IOWait %', color='orange', linestyle='--', zorder=3)
        
        ax2.set_xlabel('Time (Seconds)', fontsize=STYLE['fs_label'], labelpad=STYLE['label_pad'])
        ax2.set_ylabel('Utilization (%)', fontsize=STYLE['fs_label'], labelpad=STYLE['label_pad'])
        ax2.set_title('CPU Utilization Over Time', fontsize=STYLE['fs_title'], fontweight='bold')
        ax2.tick_params(axis='both', labelsize=STYLE['fs_tick'])
        ax2.legend(fontsize=STYLE['fs_legend'])
        ax2.set_ylim(0, 110)

    # plt.tight_layout() # Using subplots_adjust instead
    plot_path = os.path.join(run_dir, "resource_utilization.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

def main(run_dir):
    if not os.path.isdir(run_dir):
        print(f"Error: {run_dir} is not a directory")
        return

    print(f"Results for: {os.path.abspath(run_dir)}")
    print("=" * 50)
    
    iostat_data = parse_iostat(os.path.join(run_dir, "iostat.log"), "md0")
    mpstat_data = parse_mpstat(os.path.join(run_dir, "mpstat.log"))
    throughput = parse_stdout(os.path.join(run_dir, "stdout.log"))
    
    if throughput:
        print(f"Benchmark Throughput       : {throughput:.2f} MB/s")
    
    if iostat_data:
        print(f"Disk (md0) Avg Read Rate   : {iostat_data['avg_read_mb']/1024.0:.2f} GB/s")
        print(f"Disk (md0) Avg Write Rate  : {iostat_data['avg_write_mb']/1024.0:.2f} GB/s")
    else:
        print("Disk data (md0) not found in iostat.log")

    if mpstat_data:
        print(f"CPU Avg Total Utilization  : {mpstat_data['avg_total_util']:.2f} %")
        print(f"CPU Avg User               : {mpstat_data['avg_usr']:.2f} %")
        print(f"CPU Avg System             : {mpstat_data['avg_sys']:.2f} %")
        print(f"CPU Avg IOWait             : {mpstat_data['avg_iowait']:.2f} %")
    else:
        print("CPU data not found in mpstat.log")
    
    create_plots(run_dir, iostat_data, mpstat_data)
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 ttload_summary.py <run_directory>")
        sys.exit(1)
    main(sys.argv[1])
