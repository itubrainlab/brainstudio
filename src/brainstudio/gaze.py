"""Gaze analysis: blink / fixation / saccade detection, AOIs and heatmaps.

The detectors follow the algorithms of PyGaze's companion analysis toolbox
(PyGazeAnalyser, E. S. Dalmaijer): blinks are runs of missing samples,
fixations are dispersion-based (samples stay within ``maxdist`` of the
fixation start for at least ``mindur``), saccades are velocity-based.
They are reimplemented here vectorised and on NaN-coded data so they run
on long, high-rate recordings. Areas of interest use ``pygaze.plugins.aoi.AOI``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .model import GazeData, GazeEvents

try:  # PyGaze imports distutils at package import; keep it optional.
    from pygaze.plugins.aoi import AOI
except Exception:  # pragma: no cover - fallback when pygaze is unavailable
    AOI = None


@dataclass
class DetectionParams:
    blink_min_dur: float = 0.05  # s, shorter gaps are treated as dropouts
    blink_max_dur: float = 1.0  # s, longer gaps are tracking loss, not blinks
    fix_max_dist: float = 35.0  # px, dispersion radius from fixation start
    fix_min_dur: float = 0.06  # s
    sac_vel_thresh: float = 1500.0  # px/s
    sac_min_dur: float = 0.01  # s
    smooth_window: float = 0.02  # s, velocity smoothing window


def _runs(mask: np.ndarray) -> np.ndarray:
    """Return (start, stop) index pairs (stop exclusive) of True runs."""
    if not len(mask):
        return np.empty((0, 2), int)
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    return np.column_stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)])


def _sample_dt(g: GazeData) -> float:
    return 1.0 / g.sfreq if g.sfreq else float(np.median(np.diff(g.times)))


def detect_blinks(g: GazeData, p: DetectionParams = DetectionParams()) -> pd.DataFrame:
    dt = _sample_dt(g)
    missing = ~(np.isfinite(g.x) & np.isfinite(g.y))
    rows = []
    for a, b in _runs(missing):
        onset, offset = g.times[a], g.times[b - 1] + dt
        dur = offset - onset
        if p.blink_min_dur <= dur <= p.blink_max_dur:
            rows.append((onset, offset, dur))
    return pd.DataFrame(rows, columns=["onset", "offset", "duration"])


def detect_fixations(g: GazeData, p: DetectionParams = DetectionParams()) -> pd.DataFrame:
    """Dispersion-based fixation detection (PyGazeAnalyser algorithm)."""
    x, y, t = g.x, g.y, g.times
    n = len(t)
    dt = _sample_dt(g)
    chunk = max(int(5.0 / dt), 16)  # look ahead at most 5 s per fixation
    rows = []
    i = 0
    valid = np.isfinite(x) & np.isfinite(y)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        # Grow the look-ahead window geometrically: most candidates (saccade
        # samples) end within a few samples, so this keeps the loop cheap.
        look = 16
        while True:
            stop = min(n, i + 1 + look)
            d2 = (x[i + 1:stop] - x[i]) ** 2 + (y[i + 1:stop] - y[i]) ** 2
            out = ~(d2 <= p.fix_max_dist ** 2)  # NaN (missing) also ends it
            if out.any() or stop == n or look >= chunk:
                break
            look *= 2
        j = i + 1 + (int(np.argmax(out)) if out.any() else len(d2))
        onset, offset = t[i], t[j - 1] + dt
        if offset - onset >= p.fix_min_dur:
            rows.append((onset, offset, offset - onset,
                         float(np.nanmean(x[i:j])), float(np.nanmean(y[i:j]))))
            i = j
        else:
            i += 1
    return pd.DataFrame(rows, columns=["onset", "offset", "duration", "x", "y"])


def gaze_velocity(g: GazeData, p: DetectionParams = DetectionParams()) -> np.ndarray:
    """Smoothed 2D gaze speed in px/s (NaN where data is missing)."""
    dt = _sample_dt(g)
    w = max(int(round(p.smooth_window / dt)), 1)
    vx = np.gradient(g.x, g.times) if len(g.times) > 1 else np.zeros_like(g.x)
    vy = np.gradient(g.y, g.times) if len(g.times) > 1 else np.zeros_like(g.y)
    v = np.hypot(vx, vy)
    if w > 1:
        kernel = np.ones(w) / w
        finite = np.isfinite(v)
        num = np.convolve(np.where(finite, v, 0.0), kernel, "same")
        den = np.convolve(finite.astype(float), kernel, "same")
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(finite, num / den, np.nan)
    return v


def detect_saccades(g: GazeData, p: DetectionParams = DetectionParams(),
                    velocity: np.ndarray | None = None) -> pd.DataFrame:
    v = gaze_velocity(g, p) if velocity is None else velocity
    dt = _sample_dt(g)
    rows = []
    for a, b in _runs(np.nan_to_num(v, nan=0.0) > p.sac_vel_thresh):
        onset, offset = g.times[a], g.times[b - 1] + dt
        if offset - onset < p.sac_min_dur:
            continue
        a0, b1 = max(a - 1, 0), min(b, len(v) - 1)
        x0, y0, x1, y1 = g.x[a0], g.y[a0], g.x[b1], g.y[b1]
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue  # saccade-like transient around missing data (blink edge)
        rows.append((onset, offset, offset - onset, x0, y0, x1, y1,
                     float(np.hypot(x1 - x0, y1 - y0)), float(np.nanmax(v[a:b]))))
    return pd.DataFrame(rows, columns=["onset", "offset", "duration", "x0", "y0",
                                       "x1", "y1", "amplitude", "peak_velocity"])


def detect_events(g: GazeData, p: DetectionParams = DetectionParams()) -> GazeEvents:
    return GazeEvents(
        fixations=detect_fixations(g, p),
        saccades=detect_saccades(g, p),
        blinks=detect_blinks(g, p),
    )


def clean_pupil(g: GazeData, blinks: pd.DataFrame, pad: float = 0.05) -> np.ndarray | None:
    """Pupil trace with blinks (+/- pad) linearly interpolated."""
    if g.pupil is None:
        return None
    pupil = g.pupil.copy()
    bad = ~np.isfinite(pupil)
    for on, off in blinks[["onset", "offset"]].to_numpy():
        bad |= (g.times >= on - pad) & (g.times < off + pad)
    good = ~bad
    if good.sum() >= 2:
        pupil[bad] = np.interp(g.times[bad], g.times[good], pupil[good])
    return pupil


def heatmap(fixations: pd.DataFrame, screen_size: tuple[int, int], sigma: float = 40.0,
            downsample: int = 4) -> np.ndarray:
    """Duration-weighted fixation density map, shape (height, width) // downsample."""
    w, h = (int(np.ceil(s / downsample)) for s in screen_size)
    hm = np.zeros((h, w))
    if len(fixations):
        fx = np.clip((fixations["x"].to_numpy() / downsample).astype(int), 0, w - 1)
        fy = np.clip((fixations["y"].to_numpy() / downsample).astype(int), 0, h - 1)
        ok = np.isfinite(fixations["x"].to_numpy()) & np.isfinite(fixations["y"].to_numpy())
        np.add.at(hm, (fy[ok], fx[ok]), fixations["duration"].to_numpy()[ok])
        hm = gaussian_filter(hm, sigma / downsample)
    return hm


def make_aoi(kind: str, pos: tuple[float, float], size) -> object:
    """Create a PyGaze AOI ('rectangle', 'circle' or 'ellipse').

    For rectangles ``pos`` is the top-left corner; for circles/ellipses the
    centre (PyGaze convention).
    """
    if AOI is None:
        raise RuntimeError("PyGaze is not available")
    return AOI(kind, tuple(pos), size if isinstance(size, (int, float)) else list(size))


def aoi_statistics(fixations: pd.DataFrame, aois: dict[str, object]) -> pd.DataFrame:
    """Per-AOI fixation count, dwell time and time to first fixation."""
    rows = []
    for name, aoi in aois.items():
        hits = np.array([bool(aoi.contains((fx, fy)))
                         for fx, fy in fixations[["x", "y"]].to_numpy()], dtype=bool)
        sel = fixations[hits] if len(fixations) else fixations
        rows.append({
            "aoi": name,
            "n_fixations": int(hits.sum()),
            "dwell_time": float(sel["duration"].sum()) if len(sel) else 0.0,
            "first_fixation": float(sel["onset"].min()) if len(sel) else np.nan,
            "mean_fix_dur": float(sel["duration"].mean()) if len(sel) else np.nan,
        })
    return pd.DataFrame(rows)


def summary_table(ev: GazeEvents, duration: float) -> pd.DataFrame:
    f, s, b = ev.fixations, ev.saccades, ev.blinks
    dur = max(duration, 1e-9)
    return pd.DataFrame([
        ("Fixations", len(f)),
        ("Fixation rate (/s)", len(f) / dur),
        ("Mean fixation duration (ms)", 1000 * f["duration"].mean() if len(f) else np.nan),
        ("Saccades", len(s)),
        ("Mean saccade amplitude (px)", s["amplitude"].mean() if len(s) else np.nan),
        ("Mean peak velocity (px/s)", s["peak_velocity"].mean() if len(s) else np.nan),
        ("Blinks", len(b)),
        ("Blink rate (/min)", 60 * len(b) / dur),
    ], columns=["measure", "value"])
