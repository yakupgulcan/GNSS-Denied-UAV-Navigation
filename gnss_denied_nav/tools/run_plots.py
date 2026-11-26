"""Figures for one analyzed run (called by analyze_run; written to <run>/plots/)."""
import os

import numpy as np

COLORS = {'truth': '#222222', 'ekf': '#1f77b4', 'est': '#ff7f0e', 'fix': '#2ca02c', 'route': '#888888'}


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def make_all(run_dir, s, f, series, diags, target, along_route, truth_cross, ekf_cross):
    out = os.path.join(run_dir, 'plots')
    os.makedirs(out, exist_ok=True)
    title = f"{s['run']}\n{s.get('method')} | window: take-off -> {s['window']['end']}"
    plt = _plt()
    for fn in (trajectory, errors, route_following, error_cdf, localization, speed_altitude):
        fig = fn(plt, s, f, series, diags, target, along_route, truth_cross, ekf_cross)
        if fig is None:
            continue
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'{fn.__name__}.png'), dpi=130)
        plt.close(fig)


def _window_mask(f, t):
    return (t >= f.t_takeoff) & (t < f.t_end)


def _fixes(f, diags):
    fx = [(d['stamp'], d['fix_x'], d['fix_y']) for d in diags if 'fix_x' in d]
    if not fx:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    t, x, y = map(np.array, zip(*fx))
    w = _window_mask(f, t)
    return t[w], x[w], y[w]


def trajectory(plt, s, f, series, diags, target, *_):
    fig, ax = plt.subplots(figsize=(6, 8))
    w = f.window
    ax.plot(f.x, f.y, color='#bbbbbb', lw=1, label='truth (outside window)')
    ax.plot(f.x[w], f.y[w], color=COLORS['truth'], lw=1.6, label='truth')
    for key, label in (('ekf', 'EKF'), ('est', 'published estimate')):
        t, x, y = series[key]
        m = _window_mask(f, t)
        ax.plot(x[m], y[m], color=COLORS[key], lw=1, label=label)
    if target is not None:
        ax.plot([0, target[0]], [0, target[1]], '--', color=COLORS['route'], lw=1, label='planned route')
        ax.plot(*target, '*', color='red', ms=12, label='target')
    i = min(np.searchsorted(f.grid, f.t_end), len(f.grid) - 1)
    ax.plot(f.x[i], f.y[i], 'o', mfc='none', mec='red', ms=9, label=f"window end ({s['window']['end']})")
    ax.set_xlabel('east (m)  [axis stretched]')
    ax.set_ylabel('north (m)')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    return fig


def errors(plt, s, f, series, diags, target, along_route, *_):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    xs = along_route if target is not None else f.grid - f.ts[0]
    xlabel = 'distance along route (m)' if target is not None else 'time (s)'
    for key, label in (('ekf', 'EKF'), ('est', 'published estimate')):
        t, x, y = series[key]
        m = _window_mask(f, t)
        tx, ty = f.truth(t[m])
        pos = np.interp(t[m], f.grid, xs)
        axes[0].plot(pos, np.hypot(x[m] - tx, y[m] - ty), color=COLORS[key], lw=0.9, label=label)
    t, x, y = _fixes(f, diags)
    if len(t):
        tx, ty = f.truth(t)
        axes[0].scatter(np.interp(t, f.grid, xs), np.hypot(x - tx, y - ty), s=3, color=COLORS['fix'],
                        alpha=0.4, label='raw map fix')
    axes[0].set_ylabel('error vs truth (m)')
    axes[0].legend(fontsize=7)
    t, x, y = series['ekf']
    m = _window_mask(f, t) & np.isin(np.round((t - f.grid[0]) / 0.1).astype(int).clip(0, len(f.grid) - 1),
                                     np.flatnonzero(f.cruise))
    if m.any():
        a, c = f.along_cross(t[m], x[m], y[m])
        pos = np.interp(t[m], f.grid, xs)
        axes[1].plot(pos, a, color=COLORS['ekf'], lw=0.9, label='EKF along-track')
        axes[1].plot(pos, c, color='#9467bd', lw=0.9, label='EKF cross-track')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('EKF error components (m)\n(cruise)')
    axes[1].set_xlabel(xlabel)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    return fig


def route_following(plt, s, f, series, diags, target, along_route, truth_cross, ekf_cross):
    if target is None:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    w = f.window
    ax.plot(along_route[w], truth_cross[w], color=COLORS['truth'], lw=1.2, label='true cross-track')
    ax.plot(along_route[w], ekf_cross[w], color=COLORS['ekf'], lw=0.9, label='EKF cross-track (believed)')
    ax.axhline(0, color=COLORS['route'], lw=0.8, ls='--')
    rf = s.get('route_following', {})
    rmse = rf.get('truth_cross_track', {}).get('rmse')
    ax.set_title(f'true cross-track RMSE {rmse} m', fontsize=9)
    ax.set_xlabel('distance along route (m)')
    ax.set_ylabel('distance from planned route (m)\n(+ = left)')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return fig


def error_cdf(plt, s, f, series, diags, *_):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (('ekf', 'EKF'), ('est', 'published estimate')):
        t, x, y = series[key]
        m = _window_mask(f, t)
        tx, ty = f.truth(t[m])
        e = np.sort(np.hypot(x[m] - tx, y[m] - ty))
        if len(e):
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=COLORS[key], label=label)
    t, x, y = _fixes(f, diags)
    if len(t):
        tx, ty = f.truth(t)
        e = np.sort(np.hypot(x - tx, y - ty))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=COLORS['fix'], label='raw map fix')
    ax.set_xlabel('error vs truth (m)')
    ax.set_ylabel('fraction of samples')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return fig


def localization(plt, s, f, series, diags, *_):
    frames = [d for d in diags if f.t_takeoff <= d['stamp'] < f.t_end]
    if not frames:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ms = np.array([d['ms_total'] for d in frames])
    axes[0].hist(ms, bins=40, color=COLORS['ekf'])
    axes[0].set_xlabel('processing time per frame (ms)')
    axes[0].set_ylabel('frames')
    axes[0].set_title(f'median {np.median(ms):.0f} ms, p90 {np.percentile(ms, 90):.0f} ms', fontsize=9)
    t = np.array([d['stamp'] for d in frames]) - f.ts[0]
    inl = np.array([d.get('inliers', 0) for d in frames])
    ok = np.array([d['status'] in ('MATCH', 'MATCH_INIT') for d in frames])
    axes[1].scatter(t[ok], inl[ok], s=3, color=COLORS['fix'], label='accepted')
    axes[1].scatter(t[~ok], inl[~ok], s=6, color='red', label='no fix / rejected')
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('inliers')
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    return fig


def speed_altitude(plt, s, f, *_):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    t = f.grid - f.ts[0]
    axes[0].plot(t, f.speed, color=COLORS['truth'], lw=0.8)
    axes[0].set_ylabel('true speed (m/s)')
    axes[1].plot(t, f.z, color=COLORS['truth'], lw=0.8)
    axes[1].set_ylabel('true height (m)')
    axes[1].set_xlabel('time (s)')
    for ax in axes:
        ax.axvline(f.t_takeoff - f.ts[0], color='green', lw=0.8, ls='--')
        ax.axvline(f.t_end - f.ts[0], color='red', lw=0.8, ls='--')
        ax.grid(alpha=0.3)
    axes[0].set_title('green: take-off, red: window end', fontsize=9)
    return fig
