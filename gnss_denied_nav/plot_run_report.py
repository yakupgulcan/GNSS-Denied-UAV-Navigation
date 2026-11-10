#!/usr/bin/env python3
"""
Flight report plots for a GNSS-denied navigation run.

Reads a perf_metrics_<timestamp>.csv written by perf_metrics_logger and
produces a set of PNG figures plus a text summary:

    01_route.png            planned line, flown path and estimated path
    02_error_along_route.png  position error over the route
    03_error_components.png   cross-track and along-track error
    04_match_rate.png       share of MATCH / NO_MATCH / OUTLIER per route segment
    05_error_distribution.png histogram of position error
    06_rejected_fixes.png   where rejected (outlier) fixes pointed
    07_processing_time.png  histogram of per-frame processing time
    summary.txt

Usage:
    python3 plot_run_report.py                       # newest perf_metrics_*.csv in ~ or .
    python3 plot_run_report.py perf_metrics_X.csv    # a specific run
    python3 plot_run_report.py run.csv --out report_dir --segment 100
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Style ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # ground truth / primary series
ORANGE = "#eb6834"    # estimate / outliers
AQUA = "#1baf7a"      # accepted matches
NEUTRAL = "#c3c2b7"   # "no match" (absence, not a series)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "text.color": INK, "font.size": 10, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "figure.dpi": 110, "savefig.dpi": 150,
})


def find_latest_csv():
    candidates = glob.glob(os.path.expanduser("~/perf_metrics_*.csv")) + glob.glob("perf_metrics_*.csv")
    candidates = [c for c in candidates if os.path.getsize(c) > 1000]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["t"] = df["timestamp"] - df["timestamp"].iloc[0]
    df["err_x"] = df["est_x"] - df["true_x"]      # cross-track (east) error
    df["err_y"] = df["est_y"] - df["true_y"]      # along-track (north) error
    df["err"] = np.hypot(df["err_x"], df["err_y"])
    return df


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def subtitle(ax, text):
    ax.text(0.0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def plot_route(df, matches, out_dir):
    """Flown vs estimated path. Axes are NOT equal-scale: the corridor is ~100x longer than wide."""
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axhline(0.0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="Planned line (x = 0)")
    ax.plot(df["true_y"], df["true_x"], color=BLUE, label="Flown path (ground truth)")
    ax.plot(matches["est_y"], matches["est_x"], color=ORANGE, linewidth=1.4,
            label="Estimated path (at accepted fixes)")
    ax.set_title("Flown path vs. estimated path", pad=22)
    subtitle(ax, "Top view, stretched: 1 m across the track is drawn much larger than 1 m along it.")
    ax.set_xlabel("Along-track distance from home, north (m)")
    ax.set_ylabel("Cross-track offset, east (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    return save(fig, out_dir, "01_route.png")


def plot_error_along_route(matches, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(matches["true_y"], matches["err"], color=BLUE, linewidth=1.2, label="Position error")
    med = matches["err"].median()
    p95 = matches["err"].quantile(0.95)
    ax.axhline(med, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.axhline(p95, color=INK_2, linewidth=1.2, linestyle=(0, (1, 2)))
    xmax = matches["true_y"].max()
    ax.set_xlim(right=xmax * 1.12)
    label_x = xmax * 1.115
    ax.text(label_x, med, f"median {med:.1f} m", va="bottom", ha="right", color=INK_2, fontsize=9)
    ax.text(label_x, p95, f"95% below {p95:.1f} m", va="bottom", ha="right", color=INK_2, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_title("Position error along the route", pad=22)
    subtitle(ax, "Distance between the estimated and the true position at each accepted fix.")
    ax.set_xlabel("Along-track distance from home (m)")
    ax.set_ylabel("Error (m)")
    return save(fig, out_dir, "02_error_along_route.png")


def plot_error_components(matches, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, sharey=True)
    parts = [("err_x", "Cross-track error (east): estimate minus truth"),
             ("err_y", "Along-track error (north): estimate minus truth")]
    for ax, (col, title) in zip(axes, parts):
        ax.axhline(0.0, color=AXIS, linewidth=1.0)
        ax.plot(matches["true_y"], matches[col], color=BLUE, linewidth=1.1)
        mean, std = matches[col].mean(), matches[col].std()
        ax.set_title(title, fontsize=11)
        ax.text(0.995, 0.93, f"mean {mean:+.1f} m, spread (std) {std:.1f} m",
                transform=ax.transAxes, ha="right", va="top", color=INK_2, fontsize=9)
        ax.set_ylabel("Error (m)")
    axes[-1].set_xlabel("Along-track distance from home (m)")
    return save(fig, out_dir, "03_error_components.png")


def plot_match_rate(df, out_dir, segment):
    seg = (df["true_y"] // segment) * segment
    status = df["status"].replace({"MATCH_INIT": "MATCH"})
    share = pd.crosstab(seg, status, normalize="index") * 100.0
    for col in ("MATCH", "OUTLIER", "NO_MATCH"):
        if col not in share:
            share[col] = 0.0
    x = share.index.values + segment / 2.0
    width = segment * 0.86
    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(share))
    for col, color, label in (("MATCH", AQUA, "Accepted fix"),
                              ("OUTLIER", ORANGE, "Rejected fix (outlier)"),
                              ("NO_MATCH", NEUTRAL, "No match found")):
        ax.bar(x, share[col].values, width=width, bottom=bottom, color=color,
               edgecolor=SURFACE, linewidth=1.0, label=label)
        bottom += share[col].values
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    overall = (status == "MATCH").mean() * 100.0
    ax.set_title("What happened to each camera frame, along the route", pad=22)
    subtitle(ax, f"Share of processed frames per {segment:.0f} m segment. Overall accepted: {overall:.0f}%.")
    ax.set_xlabel("Along-track distance from home (m)")
    ax.set_ylabel("Share of frames (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    return save(fig, out_dir, "04_match_rate.png")


def plot_error_distribution(matches, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    upper = max(5.0, np.ceil(matches["err"].quantile(0.995)))
    ax.hist(matches["err"].clip(upper=upper), bins=np.arange(0, upper + 1, 1.0),
            color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    med = matches["err"].median()
    p95 = matches["err"].quantile(0.95)
    top = ax.get_ylim()[1]
    for value, text in ((med, f"median {med:.1f} m"), (p95, f"95% below {p95:.1f} m")):
        ax.axvline(value, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(value, top, f" {text}", color=INK_2, fontsize=9, va="top", ha="left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Distribution of position error", pad=22)
    subtitle(ax, f"Accepted fixes only, 1 m bins. Largest error {matches['err'].max():.1f} m.")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Number of fixes")
    return save(fig, out_dir, "05_error_distribution.png")


def plot_rejected_fixes(df, out_dir):
    out = df[df["status"] == "OUTLIER"]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    lim = max(df["true_y"].max(), out["est_y"].max() if len(out) else 0.0) * 1.03
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            label="Correct place (claimed = true)")
    ax.scatter(out["true_y"], out["est_y"], s=14, color=ORANGE, edgecolor=SURFACE, linewidth=0.5,
               label="Rejected fix")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title("Where the rejected fixes pointed", pad=22)
    subtitle(ax, f"{len(out)} matches rejected by the outlier gate. Far from the dashed line = a look-alike place.")
    ax.set_xlabel("True along-track position (m)")
    ax.set_ylabel("Position the database match claimed (m)")
    return save(fig, out_dir, "06_rejected_fixes.png")


def plot_processing_time(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    upper = np.ceil(df["proc_time_ms"].quantile(0.99) / 50.0) * 50.0
    ax.hist(df["proc_time_ms"].clip(upper=upper), bins=np.arange(0, upper + 25, 25.0),
            color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    med = df["proc_time_ms"].median()
    ax.axvline(med, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(med, ax.get_ylim()[1], f" median {med:.0f} ms ({1000.0 / med:.1f} frames/s)",
            color=INK_2, fontsize=9, va="top", ha="left")
    ax.grid(axis="x", visible=False)
    ax.set_title("Processing time per camera frame", pad=22)
    subtitle(ax, "Feature extraction + database matching, 25 ms bins. Slowest 1% folded into the last bin.")
    ax.set_xlabel("Processing time (ms)")
    ax.set_ylabel("Number of frames")
    return save(fig, out_dir, "07_processing_time.png")


def longest_gap(df):
    """Longest stretch without an accepted fix: (seconds, metres travelled)."""
    m = df[df["status"].isin(["MATCH", "MATCH_INIT"])]
    if len(m) < 2:
        return float("nan"), float("nan")
    gaps = np.diff(m["t"].values)
    i = int(np.argmax(gaps))
    metres = np.hypot(m["true_x"].values[i + 1] - m["true_x"].values[i],
                      m["true_y"].values[i + 1] - m["true_y"].values[i])
    return float(gaps[i]), float(metres)


def summarize(df, matches, csv_path):
    status = df["status"].replace({"MATCH_INIT": "MATCH"})
    total = len(df)
    path_len = float(np.hypot(np.diff(df["true_x"]), np.diff(df["true_y"])).sum())
    net = float(np.hypot(df["true_x"].iloc[-1] - df["true_x"].iloc[0],
                         df["true_y"].iloc[-1] - df["true_y"].iloc[0]))
    gap_s, gap_m = longest_gap(df)
    lines = [
        f"Run:                 {os.path.basename(csv_path)}",
        f"Algorithm:           {df['algo_name'].iloc[0]}",
        f"Duration:            {df['t'].iloc[-1]:.0f} s",
        f"Distance flown:      {path_len:.0f} m  (mean speed {path_len / max(df['t'].iloc[-1], 1e-6):.1f} m/s)",
        f"Net progress:        {net:.0f} m  (flown/net = {path_len / max(net, 1e-6):.2f}; above 1 means weaving)",
        f"Final true position: x {df['true_x'].iloc[-1]:.1f} m, y {df['true_y'].iloc[-1]:.1f} m",
        "",
        f"Frames processed:    {total}",
        f"  accepted fix:      {(status == 'MATCH').sum()} ({(status == 'MATCH').mean() * 100:.1f}%)",
        f"  rejected outlier:  {(status == 'OUTLIER').sum()} ({(status == 'OUTLIER').mean() * 100:.1f}%)",
        f"  no match:          {(status == 'NO_MATCH').sum()} ({(status == 'NO_MATCH').mean() * 100:.1f}%)",
        f"Longest gap without an accepted fix: {gap_s:.1f} s ({gap_m:.0f} m)",
        "",
        "Position error at accepted fixes:",
        f"  median {matches['err'].median():.2f} m | RMSE {np.sqrt((matches['err'] ** 2).mean()):.2f} m | "
        f"95% below {matches['err'].quantile(0.95):.2f} m | max {matches['err'].max():.2f} m",
        f"  cross-track: mean {matches['err_x'].mean():+.2f} m, std {matches['err_x'].std():.2f} m",
        f"  along-track: mean {matches['err_y'].mean():+.2f} m, std {matches['err_y'].std():.2f} m",
        "",
        "Flown path vs. planned line (x = 0):",
        f"  mean offset {df['true_x'].mean():+.1f} m | range {df['true_x'].min():.1f} .. {df['true_x'].max():.1f} m",
        "",
        f"Processing time: median {df['proc_time_ms'].median():.0f} ms, "
        f"95% below {df['proc_time_ms'].quantile(0.95):.0f} ms",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plot a GNSS-denied navigation run report.")
    parser.add_argument("csv", nargs="?", help="perf_metrics_*.csv (default: newest in ~ or .)")
    parser.add_argument("--out", help="output directory (default: <csv name>_report next to the CSV)")
    parser.add_argument("--segment", type=float, default=100.0, help="route segment length in m for the match-rate chart")
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv()
    if not csv_path or not os.path.exists(csv_path):
        sys.exit("No perf_metrics CSV found. Pass the file path as an argument.")

    df = load(csv_path)
    matches = df[df["status"] == "MATCH"]
    if matches.empty:
        sys.exit(f"{csv_path} contains no accepted fixes (MATCH rows); nothing to plot.")

    out_dir = args.out or os.path.splitext(csv_path)[0] + "_report"
    os.makedirs(out_dir, exist_ok=True)

    plot_route(df, matches, out_dir)
    plot_error_along_route(matches, out_dir)
    plot_error_components(matches, out_dir)
    plot_match_rate(df, out_dir, args.segment)
    plot_error_distribution(matches, out_dir)
    plot_rejected_fixes(df, out_dir)
    plot_processing_time(df, out_dir)

    text = summarize(df, matches, csv_path)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()
