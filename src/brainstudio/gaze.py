"""Gaze analysis with pymovements: event detection, AOIs, heatmaps and statistics.

Fixations, saccades and blinks are detected with `pymovements
<https://pymovements.readthedocs.io>`_ on the screen-pixel gaze:

* fixations: I-DT (dispersion threshold, ``pm.events.idt``);
* saccades: the Engbert & Kliegl velocity-threshold algorithm
  (``pm.events.microsaccades``, which detects saccades of any size);
* blinks: ``pm.events.blink`` on the pupil signal (or, without a pupil channel, on the
  samples where tracking is lost).

Velocities come from ``Gaze.pos2vel`` and the fixation location and saccade amplitude /
peak velocity from ``Gaze.compute_event_properties``. Areas of interest are rectangles
mapped with ``Events.map_to_aois`` (a ``pm.stimulus.TextStimulus`` table).
pymovements gets the sample number as its time axis, so durations are given in samples:
that handles any sampling rate, even an irregular one (its I-DT needs a constant interval
between timesteps). Velocities use the sampling rate (px/s), and event onsets and
offsets are mapped back to the exact sample times.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import pymovements as pm
from scipy.ndimage import gaussian_filter

from .model import GazeData, GazeEvents


@dataclass
class DetectionParams:
    """pymovements event detection on screen pixels (defaults: pymovements' own, with
    about 40 px for its 1° dispersion threshold)."""
    fix_dispersion: float = 40.0        # px, I-DT: (max − min x) + (max − min y)
    fix_min_dur: float = 0.1            # s
    sac_threshold_factor: float = 6.0   # λ: multiples of the median-based velocity SD
    sac_min_dur: float = 0.006          # s
    blink_min_dur: float = 0.05         # s
    blink_max_dur: float = 0.5          # s
    velocity_method: str = "smooth"     # Gaze.pos2vel: smooth (Savitzky-Golay),
    #                                     neighbors or preceding


def sampling_rate(g: GazeData) -> float:
    return float(g.sfreq or 1.0 / float(np.median(np.diff(g.times))))


def to_pymovements(g: GazeData, p: DetectionParams = DetectionParams()) -> pm.Gaze:
    """The gaze as a pymovements ``Gaze``: the sample number as time axis, pixel =
    position (so distances are in px and velocities in px/s), the pupil (or a validity
    stand-in) and the velocity."""
    xy = np.vstack([g.x, g.y])
    gz = pm.gaze.from_numpy(time=np.arange(len(g.times), dtype=np.int64), time_unit="ms",
                            pixel=xy, position=xy,
                            experiment=pm.Experiment(sampling_rate=sampling_rate(g)))
    pupil = g.pupil if g.pupil is not None else \
        np.where(np.isfinite(g.x) & np.isfinite(g.y), 1.0, np.nan)  # NaN = tracking lost
    gz.samples = gz.samples.with_columns(pl.Series("pupil", np.asarray(pupil, float)))
    gz.pos2vel(method=p.velocity_method)
    return gz


def _times(g: GazeData, at) -> np.ndarray:
    """Sample times (s) of pymovements timestamps (sample numbers)."""
    return g.times[np.clip(np.asarray(at, np.int64), 0, len(g.times) - 1)]


def detect_events(g: GazeData, p: DetectionParams = DetectionParams()) -> GazeEvents:
    """Fixations (onset, offset, duration, x, y), saccades (… amplitude px, peak_velocity
    px/s) and blinks (onset, offset, duration), in seconds, detected with pymovements."""
    sf = sampling_rate(g)
    n = lambda s, least=1: max(int(round(s * sf)), least)  # noqa: E731  seconds → samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no events were detected" and the like
        gz = to_pymovements(g, p)
        gz.detect("idt", dispersion_threshold=p.fix_dispersion,
                  minimum_duration=n(p.fix_min_dur, 2))
        gz.detect("microsaccades", threshold_factor=p.sac_threshold_factor,
                  minimum_duration=n(p.sac_min_dur), name="saccade")
        gz.detect("blink", minimum_duration=n(p.blink_min_dur),
                  maximum_duration=n(p.blink_max_dur), minimum_gap=n(0.005))
        names = set(gz.events.frame["name"].to_list())
        if "fixation" in names:
            gz.compute_event_properties("location", name="fixation")
        if "saccade" in names:
            gz.compute_event_properties(["amplitude", "peak_velocity"], name="saccade")
    ev = gz.events.frame

    def table(name, extra=()):
        e = ev.filter(pl.col("name") == name)
        on = _times(g, e["onset"].to_numpy()) if len(e) else np.empty(0)
        off = _times(g, e["offset"].to_numpy()) if len(e) else np.empty(0)
        df = pd.DataFrame({"onset": on, "offset": off, "duration": off - on})
        for col in extra:
            df[col] = e[col].to_numpy().astype(float) if len(e) else np.empty(0)
        return df

    fix = table("fixation")
    loc = ev.filter(pl.col("name") == "fixation")
    xy = np.array(loc["location"].to_list(), float).reshape(-1, 2) if len(loc) and \
        "location" in loc.columns else np.empty((0, 2))
    fix["x"], fix["y"] = xy[:, 0], xy[:, 1]
    sac = table("saccade", [c for c in ("amplitude", "peak_velocity")
                            if c in ev.columns])
    for c in ("amplitude", "peak_velocity"):
        if c not in sac:
            sac[c] = np.nan
    return GazeEvents(fixations=fix, saccades=sac, blinks=table("blink"))


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


def make_aois(specs: dict | None) -> pm.stimulus.TextStimulus | None:
    """Rectangular areas of interest ``{name: (left, top, width, height)}`` (px) as a
    pymovements AOI table."""
    if not specs:
        return None
    names = list(specs)
    box = np.array([specs[n] for n in names], float).reshape(-1, 4)
    return pm.stimulus.TextStimulus(
        pl.DataFrame({"aoi": names, "x": box[:, 0], "y": box[:, 1], "width": box[:, 2],
                      "height": box[:, 3]}),
        aoi_column="aoi", start_x_column="x", start_y_column="y", width_column="width",
        height_column="height")


def fixation_aois(fixations: pd.DataFrame, aois: pm.stimulus.TextStimulus | None
                  ) -> np.ndarray:
    """The AOI each fixation falls in ("" for none), with ``Events.map_to_aois``."""
    out = np.full(len(fixations), "", dtype=object)
    if aois is None or not len(fixations):
        return out
    ev = pm.Events(pl.DataFrame({
        "name": ["fixation"] * len(fixations),
        "onset": np.arange(len(fixations)), "offset": np.arange(len(fixations)) + 1,
        "location": fixations[["x", "y"]].to_numpy(float).tolist()}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # fixations without a location
        ev.map_to_aois(aois, verbose=False)
    hit = ev.frame["aoi"].to_list()
    out[:] = ["" if h is None else str(h) for h in hit]
    return out


def aoi_statistics(fixations: pd.DataFrame, aois: pm.stimulus.TextStimulus | None
                   ) -> pd.DataFrame:
    """Per-AOI fixation count, dwell time and time to first fixation."""
    names = fixation_aois(fixations, aois)
    rows = []
    for name in (aois.aois["aoi"].to_list() if aois is not None else []):
        sel = fixations[names == name]
        rows.append({
            "aoi": name,
            "n_fixations": int(len(sel)),
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
