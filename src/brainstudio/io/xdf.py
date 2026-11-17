"""Load Lab Streaming Layer XDF recordings into a :class:`Recording`.

Streams are classified as
* EEG     – type contains EEG/ExG, or a numeric regular-rate stream not
            recognised as gaze (largest one wins),
* gaze    – type contains gaze/eye/ET/pupil, or channel labels look like gaze,
* markers – string-valued (or irregular-rate single-channel) streams.

pyxdf synchronises all stream clocks; the recording time base starts at
the first EEG sample (or the first gaze sample when there is no EEG).
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyxdf

from ..eeg import set_standard_montage
from ..model import GazeData, Recording

GAZE_TYPE = re.compile(r"gaze|eye|^et$|pupil|tobii|eyelink", re.I)
EEG_TYPE = re.compile(r"eeg|exg", re.I)


# --------------------------------------------------------------------------- utils
def _info(stream: dict, key: str, default=""):
    v = stream["info"].get(key, [default])
    return v[0] if isinstance(v, list) and v else (v or default)


def _channel_meta(stream: dict) -> list[dict]:
    n = int(_info(stream, "channel_count", 0) or 0)
    chans = []
    try:
        desc = stream["info"]["desc"][0]
        for ch in desc["channels"][0]["channel"]:
            chans.append({k: (v[0] if v else "") for k, v in ch.items() if isinstance(v, list)})
    except (KeyError, IndexError, TypeError):
        pass
    if len(chans) != n:
        chans = [{"label": f"ch{i + 1}"} for i in range(n)]
    for i, c in enumerate(chans):
        c.setdefault("label", f"ch{i + 1}")
        if not c["label"]:
            c["label"] = f"ch{i + 1}"
    return chans


def _is_marker(stream: dict) -> bool:
    fmt = _info(stream, "channel_format", "")
    srate = float(_info(stream, "nominal_srate", 0) or 0)
    return fmt == "string" or (srate == 0 and int(_info(stream, "channel_count", 1)) == 1)


def _gaze_score(stream: dict) -> int:
    """How much a numeric stream looks like an eye tracker (0 = not at all)."""
    data = np.asarray(stream["time_series"])
    if data.size == 0 or not np.any(np.nan_to_num(data.astype(float))):
        return 0  # empty / all-zero stream (e.g. unused biometrics channels)
    score = 0
    if GAZE_TYPE.search(str(_info(stream, "type"))):
        score += 4
    if GAZE_TYPE.search(str(_info(stream, "name"))):
        score += 1
    labels = " ".join(c["label"].lower() for c in _channel_meta(stream))
    if re.search(r"gaze|pupil|por|pog", labels):
        score += 2
    return score


def describe_streams(path: str | Path) -> pd.DataFrame:
    """Lightweight table of streams in an XDF file (headers only)."""
    streams = pyxdf.resolve_streams(str(path))
    return pd.DataFrame([{
        "stream_id": s["stream_id"], "name": s["name"], "type": s["type"],
        "channels": s["channel_count"], "srate": s["nominal_srate"],
        "format": s["channel_format"],
    } for s in streams])


# --------------------------------------------------------------------------- gaze
def _pick(labels: list[str], *patterns: str) -> list[int]:
    for pat in patterns:
        idx = [i for i, l in enumerate(labels) if re.search(pat, l, re.I)]
        if idx:
            return idx
    return []


def gaze_from_matrix(times: np.ndarray, data: np.ndarray, labels: list[str],
                     screen_size=(1920, 1080), label="gaze",
                     sfreq: float | None = None) -> GazeData:
    """Build GazeData from a samples x channels matrix with channel labels.

    Handles per-eye channels (averaged), normalised [0, 1] coordinates
    (scaled to ``screen_size``) and 0-coded missing samples.
    """
    data = np.asarray(data, dtype=float)
    labels = [l.lower() for l in labels]

    def col(idx):
        if not idx:
            return None
        with np.errstate(all="ignore"):
            return np.nanmean(data[:, idx], axis=1) if len(idx) > 1 else data[:, idx[0]].copy()

    ix = _pick(labels, r"^(gaze|por)?[_ ]?x$", r"^bpogx$", r"pogx$", r"x_?pos|pos_?x", r"(gaze|por|pos).*x", r"x.*(gaze|por)",
               r"(^|_)x(_|$)", r"\bx\b", r"x")
    iy = _pick(labels, r"^(gaze|por)?[_ ]?y$", r"^bpogy$", r"pogy$", r"y_?pos|pos_?y", r"(gaze|por|pos).*y", r"y.*(gaze|por)",
               r"(^|_)y(_|$)", r"\by\b", r"y")
    ip = _pick(labels, r"pupil", r"diam", r"area")
    # avoid 3D / eye-position channels being taken as screen coordinates
    ix = [i for i in ix if not re.search(r"pupil|diam|3d|eye_?pos|_z", labels[i])] or ix
    iy = [i for i in iy if not re.search(r"pupil|diam|3d|eye_?pos|_z", labels[i])] or iy
    if not ix or not iy:
        if data.shape[1] >= 2:
            ix, iy = [0], [1]
        else:
            raise ValueError(f"Cannot identify gaze x/y channels in {labels}")
    x, y, pupil = col(ix), col(iy), col(ip)
    missing = ~np.isfinite(x) | ~np.isfinite(y) | ((x == 0) & (y == 0))
    # explicit validity flags (e.g. Gazepoint FPOGV/BPOGV, "valid" channels)
    iv = _pick(labels, r"^" + re.escape(labels[ix[0]][:-1]) + r"v$", r"valid")
    if iv:
        flag = data[:, iv[0]]
        if set(np.unique(flag[np.isfinite(flag)])) <= {0.0, 1.0}:
            missing |= flag != 1
    x[missing] = np.nan
    y[missing] = np.nan
    if pupil is not None:
        pupil[pupil <= 0] = np.nan
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() > 10 and np.percentile(np.abs(x[finite]), 95) <= 1.5 \
            and np.percentile(np.abs(y[finite]), 95) <= 1.5:
        # normalised coordinates (e.g. Gazepoint, Pupil Labs, Tobii ADCS);
        # judged on percentiles because trackers emit occasional garbage values
        x = x * screen_size[0]
        y = y * screen_size[1]
        if any("norm" in l for l in labels):  # Pupil Labs: origin bottom-left
            y = screen_size[1] - y
    # samples far outside the screen are tracker glitches
    w, h = screen_size
    off = (x < -0.5 * w) | (x > 1.5 * w) | (y < -0.5 * h) | (y > 1.5 * h)
    x[off] = np.nan
    y[off] = np.nan
    extra = {lab: data[:, i] for i, lab in enumerate(labels)}
    return GazeData(times, x, y, pupil, sfreq, tuple(screen_size), label, extra)


# --------------------------------------------------------------------------- EEG
_UNIT_SCALE = {"microvolts": 1e-6, "uv": 1e-6, "µv": 1e-6, "μv": 1e-6,
               "millivolts": 1e-3, "mv": 1e-3, "volts": 1.0, "v": 1.0, "nanovolts": 1e-9}


def _clean_timestamps(stream: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove foreign-clock samples and jitter from a stream's timestamps.

    Returns (timestamps, sample indices kept, info). ``info`` holds
    ``foreign_offset`` (foreign clock = own clock + offset) when some samples
    were stamped on another clock, the effective sampling rate and notes.
    """
    ts = np.asarray(stream["time_stamps"], float)
    n = len(ts)
    keep = np.ones(n, bool)
    info: dict = {"notes": []}
    nominal = float(_info(stream, "nominal_srate", 0) or 0)
    if n < 3:
        return ts, np.flatnonzero(keep), info

    # 1) samples stamped on a different clock show up as a jump of +D
    #    followed (within a few samples) by a jump of ~-D back
    d = np.diff(ts)
    big = 10.0  # s; no real inter-sample interval is this large *and* reversed
    offsets = []
    jumps = np.flatnonzero(np.abs(d) > big)
    used: set[int] = set()
    for j, a in enumerate(jumps):
        if a in used:  # already closed a previous excursion
            continue
        for b in jumps[j + 1: j + 50]:
            if b - a > 1000:
                break
            if np.sign(d[b]) != np.sign(d[a]) and abs(d[a] + d[b]) < 0.01 * abs(d[a]):
                keep[a + 1: b + 1] = False
                offsets.append((d[a] - d[b]) / 2)
                used.update((int(a), int(b)))
                break
    if offsets and np.std(offsets) < 1.0:
        info["foreign_offset"] = float(np.median(offsets))
        info["notes"].append(
            f"{(~keep).sum()} samples stamped on another clock removed "
            f"(clock offset {info['foreign_offset']:+.3f} s ± {np.std(offsets) * 1000:.1f} ms)")
    elif offsets:
        info["notes"].append(f"{(~keep).sum()} out-of-order samples removed")
    idx = np.flatnonzero(keep)
    ts = ts[idx]

    # 2) de-jitter regular streams per gap-free segment (linear fit)
    if nominal > 0 and len(ts) > 2:
        d = np.diff(ts)
        gap = max(1.0, 50.0 / nominal)
        cuts = np.flatnonzero((d > gap) | (d < -gap)) + 1
        rates = []
        out = ts.copy()
        for seg in np.split(np.arange(len(ts)), cuts):
            if len(seg) < 3:
                continue
            k = np.arange(len(seg))
            slope, icpt = np.polyfit(k, ts[seg], 1)
            out[seg] = icpt + slope * k
            if slope > 0:
                rates.append((len(seg), 1.0 / slope))
        ts = out
        if rates:
            eff = float(np.average([r for _, r in rates], weights=[w for w, _ in rates]))
            info["srate"] = eff
            if abs(eff - nominal) / nominal > 0.05:
                info["notes"].append(f"effective rate {eff:.2f} Hz (nominal {nominal:g} Hz)")
        if len(cuts):
            info["notes"].append(f"{len(cuts)} recording gap(s)")
            info["gaps"] = [(float(ts[c - 1]), float(ts[c])) for c in cuts]
    elif nominal > 0:
        info["srate"] = nominal
    return ts, idx, info


def _raw_from_stream(stream: dict, ts: np.ndarray, idx: np.ndarray, srate: float,
                     t0: float) -> mne.io.RawArray:
    chans = _channel_meta(stream)
    data = np.asarray(stream["time_series"], dtype=np.float64)[idx].T
    # Resample onto a regular grid starting at t0 so that jitter/dropouts do
    # not accumulate drift.
    grid = t0 + np.arange(int(np.floor((ts[-1] - t0) * srate)) + 1) / srate
    if len(grid) != len(ts) or np.abs(ts - grid).max() > 0.5 / srate:
        data = np.vstack([np.interp(grid, ts, ch) for ch in data])
    names, types, scales = [], [], []
    seen = set()
    for c in chans:
        name = c["label"]
        while name in seen:
            name += "_"
        seen.add(name)
        names.append(name)
        ctype = str(c.get("type", "")).lower()
        types.append({"eeg": "eeg", "eog": "eog", "ecg": "ecg", "emg": "emg",
                      "trigger": "stim", "stim": "stim", "aux": "misc",
                      "exg": "eeg"}.get(ctype, "eeg" if not ctype else "misc"))
        unit = str(c.get("unit", "")).lower()
        scales.append(_UNIT_SCALE.get(unit, 1e-6))  # LSL EEG is µV by convention
    data *= np.asarray(scales)[:, None]
    info = mne.create_info(names, srate, types)
    raw = mne.io.RawArray(data, info, verbose="error")
    set_standard_montage(raw)  # if channel names match 10-20/10-05
    return raw


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Overlap of two time ranges as a fraction of the shorter one."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    shortest = max(min(a[1] - a[0], b[1] - b[0]), 1e-9)
    return max(inter, 0.0) / shortest


# --------------------------------------------------------------------------- main
def load_xdf(path: str | Path, screen_size=(1920, 1080), eeg_stream=None,
             gaze_stream=None) -> Recording:
    """Load an XDF file. ``eeg_stream``/``gaze_stream`` force a stream by name or id.

    Clock handling: LSL clock offsets recorded in the file are applied by
    pyxdf. Streams that still do not overlap the EEG in time are mapped onto
    its clock using offsets recovered from samples stamped on the other clock
    (see :func:`_clean_timestamps`); failing that, stream starts are aligned
    and a warning is added to ``Recording.description``.
    """
    path = Path(path)
    # Jitter removal is done here (robust to stray timestamps), not by pyxdf.
    streams, _header = pyxdf.load_xdf(str(path), synchronize_clocks=True,
                                      dejitter_timestamps=False, verbose=False)
    streams = [s for s in streams if len(s["time_stamps"])]

    def matches(s, key):
        return key is not None and (str(key) == _info(s, "name") or
                                    str(key) == str(s["info"].get("stream_id")))

    markers = [s for s in streams if _is_marker(s)]
    numeric = [s for s in streams if not _is_marker(s)]
    gaze_s = next((s for s in numeric if matches(s, gaze_stream)), None)
    if gaze_s is None:
        scored = [(sc, k, s) for k, s in enumerate(numeric) if (sc := _gaze_score(s)) > 0]
        gaze_s = max(scored, key=lambda x: (x[0], -x[1]))[2] if scored else None
    eeg_cands = [s for s in numeric if s is not gaze_s and _gaze_score(s) < 4]
    eeg_s = next((s for s in eeg_cands if matches(s, eeg_stream)), None) or next(
        (s for s in eeg_cands if EEG_TYPE.search(str(_info(s, "type")))), None)
    if eeg_s is None and eeg_cands:
        eeg_s = max(eeg_cands, key=lambda s: int(_info(s, "channel_count", 0)))

    notes: list[str] = []
    clean = {}
    for s in [x for x in (eeg_s, gaze_s) if x is not None] + markers:
        ts, idx, info = _clean_timestamps(s)
        clean[id(s)] = (ts, idx, info)
        notes += [f"{_info(s, 'name')}: {m}" for m in info["notes"]]

    # ---- clock alignment onto the reference (EEG) stream
    ref = eeg_s or gaze_s
    shifts = {}
    if ref is not None:
        rts = clean[id(ref)][0]
        ref_range = (rts[0], rts[-1])
        known_offsets = [i["foreign_offset"] for _, _, i in clean.values()
                         if "foreign_offset" in i]
        for s in [x for x in (gaze_s,) if x is not None and x is not ref] + markers:
            ts = clean[id(s)][0]
            if not len(ts):
                continue
            rng = (ts[0], ts[-1])
            shift = 0.0
            if _overlap(rng, ref_range) == 0.0:
                cands = [o for o in known_offsets
                         if _overlap((rng[0] + o, rng[1] + o), ref_range) > 0]
                if cands:
                    shift = cands[0]
                    notes.append(f"{_info(s, 'name')}: mapped onto {_info(ref, 'name')} clock "
                                 f"using recovered offset {shift:+.3f} s")
                elif s is gaze_s:
                    shift = ref_range[0] - rng[0]
                    notes.append(f"WARNING {_info(s, 'name')}: no common clock with "
                                 f"{_info(ref, 'name')}; stream starts aligned — check sync")
            shifts[id(s)] = shift

    if eeg_s is not None:
        t0 = float(clean[id(eeg_s)][0][0])
    elif gaze_s is not None:
        t0 = float(clean[id(gaze_s)][0][0] + shifts.get(id(gaze_s), 0.0))
    else:
        t0 = 0.0

    rec = Recording(source=path)
    if eeg_s is not None:
        ts, idx, info = clean[id(eeg_s)]
        srate = info.get("srate") or float(1.0 / np.median(np.diff(ts)))
        rec.raw = _raw_from_stream(eeg_s, ts, idx, srate, t0)
        gaps = info.get("gaps", [])
        if gaps:
            rec.raw.set_annotations(mne.Annotations(
                [a - t0 for a, _ in gaps], [b - a for a, b in gaps], ["BAD_acq_gap"] * len(gaps)))
    if gaze_s is not None:
        ts, idx, info = clean[id(gaze_s)]
        labels = [c["label"] for c in _channel_meta(gaze_s)]
        data = np.asarray(gaze_s["time_series"], dtype=float)[idx]
        times = ts + shifts.get(id(gaze_s), 0.0) - t0
        rec.gaze = gaze_from_matrix(times, data, labels, screen_size,
                                    _info(gaze_s, "name") or "gaze", info.get("srate"))
    rows = []
    for s in markers:
        ts, idx, _ = clean[id(s)]
        shift = shifts.get(id(s), 0.0)
        series = s["time_series"]
        for t, k in zip(ts, idx):
            v = series[k]
            val = v[0] if isinstance(v, (list, tuple, np.ndarray)) and len(v) else v
            val = str(val).strip()
            if val:  # many "user data" streams are mostly empty strings
                rows.append((float(t + shift - t0), 0.0, val))
    events = pd.DataFrame(rows, columns=["onset", "duration", "description"])
    if rec.raw is not None and len(events):
        # Markers inside the EEG become MNE annotations; the rest stay in
        # rec.events so they are not lost (Recording.all_events merges both).
        inside = (events.onset >= 0) & (events.onset <= rec.raw.times[-1])
        rec.raw.set_annotations(rec.raw.annotations + mne.Annotations(
            events.onset[inside].to_numpy(), events.duration[inside].to_numpy(),
            events.description[inside].to_numpy()))
        events = events[~inside]
    rec.events = events.reset_index(drop=True)
    roles = {id(m): "markers" for m in markers}
    roles.update({id(eeg_s): "EEG", id(gaze_s): "gaze"})
    names = ", ".join(f"{_info(s, 'name')} [{roles.get(id(s), 'unused')}]" for s in streams)
    rec.description = "XDF streams: " + names + ("\n" + "\n".join(notes) if notes else "")
    return rec
