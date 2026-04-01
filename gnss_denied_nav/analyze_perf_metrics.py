import pandas as pd
import numpy as np
import sys


def analyze(csv_file):
    df = pd.read_csv(csv_file)

    # 1. Mean processing time and FPS
    avg_time_ms = df['proc_time_ms'].mean()
    avg_fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0

    # 2. Match success rate
    total_frames = len(df)
    matches = df[df['status'] == 'MATCH']
    match_count = len(matches)
    success_rate = (match_count / total_frames) * 100 if total_frames > 0 else 0

    # 3. Position error (MATCH frames only)
    rmse = np.sqrt((matches['error_m']**2).mean())
    max_error = matches['error_m'].max()

    print("-" * 40)
    print(f"ANALYSIS REPORT: {csv_file}")
    print("-" * 40)
    print(f"Total frames     : {total_frames}")
    print(f"Successful match : {match_count} ({success_rate:.2f}%)")
    print("-" * 40)
    print(f"Mean proc. time  : {avg_time_ms:.2f} ms")
    print(f"Mean FPS         : {avg_fps:.2f}")
    print("-" * 40)
    print(f"Position RMSE    : {rmse:.2f} m")
    print(f"Max error        : {max_error:.2f} m")
    print("-" * 40)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_perf_metrics.py <file.csv>")
    else:
        analyze(sys.argv[1])