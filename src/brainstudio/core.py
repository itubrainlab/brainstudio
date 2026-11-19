"""brAInStudio's analysis, as plain functions.

Everything the app computes is done here, with standard libraries (MNE, MNE-BIDS, pyxdf,
pymovements, AutoReject, scipy, pandas). The app calls these functions, and an exported
script contains these same functions plus the settings it ran with — so the two give the
same results.

Data are an MNE ``Raw`` (EEG) and two pandas tables:

* ``events``: ``onset`` (s from the first EEG sample), ``duration``, ``description``
  (``NAME|key=value|…`` for markers that carry columns);
* ``gaze``: ``t`` (s), ``x``, ``y`` (screen px, NaN = lost), optionally ``pupil``, plus
  the tracker's own channels; ``gaze.attrs`` holds ``sfreq``, ``screen`` and ``label``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

mne.set_log_level("error")  # as the exported scripts: MNE's progress messages off

# =========================================================================== loading: XDF
GAZE_TYPE = re.compile(r"gaze|eye|^et$|pupil|tobii|eyelink", re.I)
EEG_TYPE = re.compile(r"eeg|exg", re.I)
#: channel types the XDF stream dialog offers
EEG_CHANNEL_TYPES = ("eeg", "eog", "ecg", "emg", "misc", "stim")
ROLES = ("EEG", "gaze", "markers", "ignore")
UNIT_SCALE = {"microvolts": 1e-6, "uv": 1e-6, "µv": 1e-6, "μv": 1e-6, "millivolts": 1e-3,
              "mv": 1e-3, "volts": 1.0, "v": 1.0, "nanovolts": 1e-9}


def stream_field(s: dict, key: str, default=""):
    """A field of an XDF stream header."""
    v = s["info"].get(key, [default])
    return v[0] if isinstance(v, list) and v else (v or default)


def stream_key(s: dict) -> str:
    """Stable identifier of a stream within its file (the XDF stream id)."""
    return str(s["info"].get("stream_id", stream_field(s, "name")))


def stream_channels(s: dict) -> list[dict]:
    """Channel descriptions (label, type, unit) of a stream, ch1… when it has none."""
    n = int(stream_field(s, "channel_count", 0) or 0)
    try:
        chans = [{k: (v[0] if v else "") for k, v in ch.items() if isinstance(v, list)}
                 for ch in s["info"]["desc"][0]["channels"][0]["channel"]]
    except (KeyError, IndexError, TypeError):
        chans = []
    if len(chans) != n:
        chans = [{} for _ in range(n)]
    for i, c in enumerate(chans):
        c["label"] = c.get("label") or f"ch{i + 1}"
    return chans


def is_marker_stream(s: dict) -> bool:
    srate = float(stream_field(s, "nominal_srate", 0) or 0)
    return stream_field(s, "channel_format") == "string" or (
        srate == 0 and int(stream_field(s, "channel_count", 1)) == 1)


def numbers_from_text(s: dict) -> np.ndarray | None:
    """Samples of a text stream as numbers (some apps, e.g. Gazepoint, send numbers as
    text). None when the stream is mostly empty (occasional codes are markers) or holds
    words."""
    rows = list(s["time_series"])
    if not rows:
        return None
    df = pd.DataFrame(rows).astype(str).apply(lambda c: c.str.strip())
    df = df.replace({"": np.nan, "None": np.nan})
    num = df.apply(pd.to_numeric, errors="coerce")
    filled = int(df.notna().to_numpy().sum())
    if not filled or num.notna().to_numpy().sum() < 0.9 * filled \
            or df.notna().any(axis=1).mean() < 0.5:
        return None
    n = int(stream_field(s, "channel_count", 0) or num.shape[1])
    arr = num.to_numpy(float)
    return arr[:, :n] if arr.shape[1] >= n else np.pad(arr, ((0, 0), (0, n - arr.shape[1])),
                                                        constant_values=np.nan)


def gaze_score(s: dict) -> int:
    """How much a numeric stream looks like an eye tracker (0 = not at all)."""
    data = np.asarray(s["time_series"])
    if data.size == 0 or not np.any(np.nan_to_num(data.astype(float))):
        return 0
    labels = " ".join(c["label"].lower() for c in stream_channels(s))
    return (4 * bool(GAZE_TYPE.search(str(stream_field(s, "type"))))
            + bool(GAZE_TYPE.search(str(stream_field(s, "name"))))
            + 2 * bool(re.search(r"gaze|pupil|por|pog", labels)))


def stream_roles(streams: list[dict], roles: dict | None = None) -> tuple[dict, dict]:
    """(automatic, final) role of every stream: the gaze stream is the most eye-tracker
    like numeric one, EEG the EEG-typed (else largest) other numeric one, text streams
    are markers. ``roles`` (stream key → role) overrides."""
    markers = [s for s in streams if is_marker_stream(s)]
    numeric = [s for s in streams if not is_marker_stream(s)]
    scored = [(gaze_score(s), -k, s) for k, s in enumerate(numeric) if gaze_score(s) > 0]
    gaze = max(scored, key=lambda x: x[:2])[2] if scored else None
    cands = [s for s in numeric if s is not gaze and gaze_score(s) < 4]
    eeg = next((s for s in cands if EEG_TYPE.search(str(stream_field(s, "type")))), None) \
        or (max(cands, key=lambda s: int(stream_field(s, "channel_count", 0))) if cands
            else None)
    auto = {stream_key(s): "ignore" for s in streams}
    auto.update({stream_key(m): "markers" for m in markers})
    for s, r in ((eeg, "EEG"), (gaze, "gaze")):
        if s is not None:
            auto[stream_key(s)] = r
    final = dict(auto)
    for k, r in (roles or {}).items():
        if k in final:
            if r not in ROLES:
                raise ValueError(f"Unknown stream role {r!r} (use {', '.join(ROLES)})")
            final[str(k)] = r
    for r in ("EEG", "gaze"):
        if sum(v == r for v in final.values()) > 1:
            raise ValueError(f"Only one stream can be {r}")
    return auto, final


def clean_timestamps(s: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Timestamps without jitter and without samples stamped on another clock.

    Returns (timestamps, indices of the samples kept, info). Some trackers stamp a few
    samples with another clock: a jump of +D followed by one of −D; those samples go,
    and D is kept as ``foreign_offset``. Regularly sampled streams get a straight line
    through their timestamps per gap-free stretch, plus the line's 5 s trend when the
    rate is not constant (so drift is followed, jitter removed)."""
    from scipy.ndimage import uniform_filter1d
    ts = np.asarray(s["time_stamps"], float)
    keep = np.ones(len(ts), bool)
    info: dict = {"notes": []}
    nominal = float(stream_field(s, "nominal_srate", 0) or 0)
    if len(ts) < 3:
        return ts, np.flatnonzero(keep), info
    d = np.diff(ts)
    jumps, offsets, used = np.flatnonzero(np.abs(d) > 10.0), [], set()
    for j, a in enumerate(jumps):
        for b in [] if a in used else jumps[j + 1: j + 50]:
            if b - a > 1000:
                break
            if np.sign(d[b]) != np.sign(d[a]) and abs(d[a] + d[b]) < 0.01 * abs(d[a]):
                keep[a + 1: b + 1] = False
                offsets.append((d[a] - d[b]) / 2)
                used.update((int(a), int(b)))
                break
    if offsets and np.std(offsets) < 1.0:
        info["foreign_offset"] = float(np.median(offsets))
        info["notes"].append(f"{(~keep).sum()} samples stamped on another clock removed "
                             f"(clock offset {info['foreign_offset']:+.3f} s ± "
                             f"{np.std(offsets) * 1000:.1f} ms)")
    elif offsets:
        info["notes"].append(f"{(~keep).sum()} out-of-order samples removed")
    idx = np.flatnonzero(keep)
    ts = ts[idx]
    if nominal > 0 and len(ts) > 2:
        gap = max(1.0, 50.0 / nominal)
        cuts = np.flatnonzero(np.abs(np.diff(ts)) > gap) + 1
        out, rates, drift = ts.copy(), [], 0.0
        for seg in np.split(np.arange(len(ts)), cuts):
            if len(seg) < 3:
                continue
            k = np.arange(len(seg))
            slope, start = np.polyfit(k, ts[seg], 1)
            line = start + slope * k
            if slope > 0:
                rates.append((len(seg), 1.0 / slope))
                trend = uniform_filter1d(ts[seg] - line, int(min(len(seg), max(3, round(
                    5.0 / slope)))), mode="nearest")
                if np.abs(trend).max() > max(0.01, 2 * slope):
                    drift = max(drift, float(np.abs(trend).max()))
                    line = np.maximum.accumulate(line + trend)
            out[seg] = line
        ts = out
        if drift:
            info["notes"].append("sampling rate not constant: timestamps followed (up to "
                                 f"{drift:.2f} s away from a constant rate)")
        if rates:
            info["srate"] = float(np.average([r for _, r in rates],
                                             weights=[w for w, _ in rates]))
            if abs(info["srate"] - nominal) / nominal > 0.05:
                info["notes"].append(f"effective rate {info['srate']:.2f} Hz (nominal "
                                     f"{nominal:g} Hz)")
        if len(cuts):
            info["notes"].append(f"{len(cuts)} recording gap(s)")
            info["gaps"] = [(float(ts[c - 1]), float(ts[c])) for c in cuts]
    elif nominal > 0:
        info["srate"] = nominal
    return ts, idx, info


def channel_type(c: dict) -> str:
    t = str(c.get("type", "")).lower()
    return {"eeg": "eeg", "eog": "eog", "ecg": "ecg", "emg": "emg", "trigger": "stim",
            "stim": "stim", "aux": "misc", "exg": "eeg"}.get(t, "eeg" if not t else "misc")


def eeg_from_stream(s: dict, ts, idx, srate: float, t0: float, opts=None) -> mne.io.RawArray:
    """The EEG stream as an MNE Raw on a regular time grid from ``t0`` (µV → V unless the
    channels say otherwise). ``opts``: include (bool per channel), names, types."""
    chans, opts = stream_channels(s), opts or {}
    data = np.asarray(s["time_series"], float)[idx].T
    sel = [k for k, on in enumerate(opts.get("include") or [True] * len(chans))
           if on and k < len(chans)]
    if not sel:
        raise ValueError("No EEG channels selected")
    names = opts.get("names") or [c["label"] for c in chans]
    types = opts.get("types") or [channel_type(c) for c in chans]
    grid = t0 + np.arange(int(np.round((ts[-1] - t0) * srate)) + 1) / srate
    data = data[sel]
    if len(grid) != len(ts) or np.abs(ts - grid).max() > 0.5 / srate:
        data = np.vstack([np.interp(grid, ts, ch) for ch in data])
    labels = []
    for k in sel:  # unique names
        name = names[k] or chans[k]["label"]
        while name in labels:
            name += "_"
        labels.append(name)
    scale = [UNIT_SCALE.get(str(chans[k].get("unit", "")).lower(), 1e-6) for k in sel]
    raw = mne.io.RawArray(data * np.asarray(scale)[:, None],
                          mne.create_info(labels, srate, [types[k] for k in sel]))
    standard_montage(raw)
    return raw


GAZE_COLUMN_ROLES = ("x", "y", "pupil", "valid", "extra", "ignore")


def gaze_columns(labels: list[str], n_cols: int | None = None) -> dict[str, list[int]]:
    """Which channels are gaze x, y, pupil and validity (indices)."""
    labels = [label.lower() for label in labels]

    def pick(*patterns):
        for pat in patterns:
            found = [i for i, lab in enumerate(labels) if re.search(pat, lab, re.I)]
            if found:
                return found
        return []
    ix = pick(r"^(gaze|por)?[_ ]?x$", r"^bpogx$", r"pogx$", r"x_?pos|pos_?x", r"(gaze|por|pos).*x",
              r"x.*(gaze|por)", r"(^|_)x(_|$)", r"\bx\b", r"x")
    iy = pick(r"^(gaze|por)?[_ ]?y$", r"^bpogy$", r"pogy$", r"y_?pos|pos_?y", r"(gaze|por|pos).*y",
              r"y.*(gaze|por)", r"(^|_)y(_|$)", r"\by\b", r"y")
    ip = pick(r"pupil", r"diam", r"^[lr]pmm$", r"^[lr]pd$", r"area")
    not_screen = r"pupil|diam|3d|eye_?pos|_z"
    ix = [i for i in ix if not re.search(not_screen, labels[i])] or ix
    iy = [i for i in iy if not re.search(not_screen, labels[i])] or iy
    if (not ix or not iy) and (n_cols if n_cols is not None else len(labels)) >= 2:
        ix, iy = [0], [1]
    iv = pick(r"^" + re.escape(labels[ix[0]][:-1]) + r"v$", r"valid") if ix else []
    return {"x": ix, "y": iy, "pupil": ip, "valid": iv[:1]}


def numeric_samples(series, idx) -> np.ndarray:
    """Rows idx of a stream's samples as floats; text that is not a number (a string-format
    stream's empty USER field, say) becomes NaN."""
    a = np.asarray(series)[idx]
    if a.dtype.kind in "biuf":
        return a.astype(float)
    return pd.DataFrame(a).apply(pd.to_numeric, errors="coerce").to_numpy(float)


def gaze_table(times, data, labels, screen=(1920, 1080), label="gaze", sfreq=None,
               mapping=None) -> pd.DataFrame:
    """Gaze samples as a table in screen pixels: x/y averaged over eyes, lost samples
    (NaN, 0/0, validity flag ≠ 1) and glitches far off screen set to NaN, 0–1
    coordinates scaled to the screen. The tracker's channels are kept (lower-case)."""
    data = np.asarray(data, float)
    labels = [lab.lower() for lab in labels]
    m = mapping or gaze_columns(labels, data.shape[1])
    if not m.get("x") or not m.get("y"):
        raise ValueError(f"Cannot identify gaze x/y channels in {labels}")

    def col(idx):
        with np.errstate(all="ignore"):
            return None if not idx else np.nanmean(data[:, idx], axis=1)
    x, y, pupil = col(m["x"]), col(m["y"]), col(m.get("pupil"))
    lost = ~np.isfinite(x) | ~np.isfinite(y) | ((x == 0) & (y == 0))
    if m.get("valid"):
        flag = data[:, m["valid"][0]]
        if set(np.unique(flag[np.isfinite(flag)])) <= {0.0, 1.0}:
            lost |= flag != 1
    x[lost] = y[lost] = np.nan
    ok = np.isfinite(x)
    if ok.sum() > 10 and max(np.percentile(np.abs(x[ok]), 95),
                             np.percentile(np.abs(y[ok]), 95)) <= 1.5:
        x, y = x * screen[0], y * screen[1]  # normalised coordinates
        if any("norm" in lab for lab in labels):  # Pupil Labs: origin bottom-left
            y = screen[1] - y
    w, h = screen
    off = (x < -0.5 * w) | (x > 1.5 * w) | (y < -0.5 * h) | (y > 1.5 * h)
    x[off] = y[off] = np.nan
    g = pd.DataFrame({"t": np.asarray(times, float), "x": x, "y": y})
    if pupil is not None:
        pupil[(pupil <= 0) | lost] = np.nan
        g["pupil"] = pupil
    for k, lab in enumerate(labels):
        g["ch_" + lab] = data[:, k]
    g.attrs.update(sfreq=sfreq or (1 / np.median(np.diff(times)) if len(times) > 1 else None),
                   screen=tuple(screen), label=label)
    return g


# ---- marker streams with several columns ('start', 'trials', 'sad', …)
MARKER_COLUMN_ROLES = ("name", "flag", "attribute", "ignore")
START_WORDS = {"start", "begin", "onset", "on", "open"}
STOP_WORDS = {"stop", "end", "offset", "off", "close"}


def marker_table(rows, labels) -> pd.DataFrame:
    n = max([len(r) for r in rows] + [len(labels)])
    cols = list(labels) + [f"ch{k + 1}" for k in range(len(labels), n)]
    return pd.DataFrame([[str(r[k]).strip() if k < len(r) and r[k] is not None else ""
                          for k in range(n)] for r in rows], columns=cols[:n])


def guess_marker_columns(table: pd.DataFrame, stream: str = "") -> tuple[list, list]:
    """(names, roles) of the columns: a start/stop column is the flag, the first short
    text column the name, the others attributes (empty ones ignored)."""
    n = table.shape[1]
    names = list(table.columns)
    generic = [bool(re.fullmatch(r"ch\d+|", str(c).lower())) for c in names]
    prefix = re.sub(r"\W+", "_", stream).strip("_").lower() or "col"
    names = [f"{prefix}_{k + 1}" if g else c for k, (c, g) in enumerate(zip(names, generic))]
    empty = [not (table.iloc[:, k] != "").any() for k in range(n)]
    roles = ["ignore" if e else "attribute" for e in empty]
    flag = next((k for k in range(n) if (v := set(table.iloc[:, k].str.lower()) - {""})
                 and len(v) >= 2 and v <= START_WORDS | STOP_WORDS and v & START_WORDS
                 and v & STOP_WORDS), None)
    if flag is not None:
        roles[flag] = "flag"
        names[flag] = "action" if generic[flag] else names[flag]
    short = [k for k in range(n) if k != flag and not empty[k]
             and 1 <= (v := table.iloc[:, k][table.iloc[:, k] != ""]).nunique() <= 40
             and v.str.len().median() <= 32]
    name_col = short[0] if short else next((k for k in range(n) if k != flag and not empty[k]),
                                           0)
    roles[name_col] = "name"
    names[name_col] = "event" if generic[name_col] else names[name_col]
    return names, roles


def marker_events(times, table, names, roles, end_events=True) -> pd.DataFrame:
    """Events from a marker table: a start row and the next stop row of the same name
    become one event with a duration (plus an instant ``<name>_end`` event)."""
    def desc(name, attrs):
        safe = lambda v: str(v).replace("|", "/").replace("=", ":")  # noqa: E731
        return "|".join([safe(name) or "marker"] + [f"{k}={safe(v)}" for k, v in attrs.items()
                                                    if v != ""])
    iname = [k for k, r in enumerate(roles) if r == "name"]
    iflag = next((k for k, r in enumerate(roles) if r == "flag"), None)
    iattr = [k for k, r in enumerate(roles) if r == "attribute"]
    out, open_ = [], {}
    for t, row in zip(times, table.to_numpy(dtype=object)):
        name = "_".join(row[k] for k in iname if row[k]) or "marker"
        attrs = {names[k]: row[k] for k in iattr}
        flag = row[iflag].lower() if iflag is not None else ""
        if flag in START_WORDS:
            open_.setdefault(name, []).append((float(t), attrs))
        elif flag in STOP_WORDS and open_.get(name):
            t0, merged = open_[name].pop(0)
            merged = dict(merged)
            for k, v in attrs.items():
                if v and not merged.get(k):
                    merged[k] = v                  # e.g. the response, known at the end
                elif v and merged.get(k) != v:
                    merged[f"{k}_end"] = v
            out.append((t0, float(t) - t0, desc(name, merged)))
            if end_events:
                out.append((float(t), 0.0, desc(f"{name}_end", merged)))
        else:
            out.append((float(t), 0.0, desc(f"{name}_{flag}" if flag else name, attrs)))
    out += [(t0, 0.0, desc(name, a)) for name, items in open_.items() for t0, a in items]
    return pd.DataFrame(out, columns=["onset", "duration", "description"]).sort_values(
        "onset", kind="stable", ignore_index=True)


def read_xdf(path, screen=(1920, 1080), roles=None, options=None):
    """EEG, events and gaze of an XDF file, in seconds from the first EEG sample.

    Returns ``(raw, events, gaze, info)``; ``info`` has the streams (with their roles
    and options, as the stream dialog shows them) and notes. ``roles``: stream key →
    EEG / gaze / markers / ignore; ``options``: stream key → EEG include/names/types,
    gaze columns, marker names/columns/end_events. LSL clock offsets are applied by
    pyxdf; a stream that still does not overlap the EEG is moved onto its clock with
    an offset recovered from samples stamped on both clocks."""
    import pyxdf
    streams = [s for s in pyxdf.load_xdf(str(path), synchronize_clocks=True,
                                         dejitter_timestamps=False, verbose=False)[0]
               if len(s["time_stamps"])]
    notes, options = [], options or {}
    for s in streams:  # numbers sent as text at a regular rate are data, not markers
        if stream_field(s, "channel_format") == "string" and \
                float(stream_field(s, "nominal_srate", 0) or 0) > 0:
            arr = numbers_from_text(s)
            if arr is not None:
                s["time_series"], s["info"]["channel_format"] = arr, ["double64"]
                notes.append(f"{stream_field(s, 'name')}: numeric samples sent as text, "
                             "read as numbers")
    auto, final = stream_roles(streams, roles)
    by_role = {r: [s for s in streams if final[stream_key(s)] == r] for r in ROLES}
    eeg = by_role["EEG"][0] if by_role["EEG"] else None
    gz = by_role["gaze"][0] if by_role["gaze"] else None
    for s in (eeg, gz):
        if s is not None and is_marker_stream(s):
            raise ValueError(f"{stream_field(s, 'name')} carries text markers, not numeric "
                             "samples, so it cannot be EEG or gaze — use the markers role")
    used = [s for s in (eeg, gz) if s is not None] + by_role["markers"]
    clean = {id(s): clean_timestamps(s) for s in used}
    for s in used:
        notes += [f"{stream_field(s, 'name')}: {m}" for m in clean[id(s)][2]["notes"]]

    # streams that do not overlap the reference stream: move them onto its clock
    ref, shift = eeg or gz, {}
    if ref is not None:
        r0, r1 = clean[id(ref)][0][[0, -1]]
        offsets = [c[2]["foreign_offset"] for c in clean.values() if "foreign_offset" in c[2]]
        overlaps = lambda a, b: min(b, r1) > max(a, r0)  # noqa: E731
        for s in ([gz] if gz is not None and gz is not ref else []) + by_role["markers"]:
            a, b = clean[id(s)][0][[0, -1]]
            if overlaps(a, b):
                continue
            o = next((o for o in offsets if overlaps(a + o, b + o)), None)
            if o is not None:
                shift[id(s)] = o
                notes.append(f"{stream_field(s, 'name')}: mapped onto "
                             f"{stream_field(ref, 'name')} clock using recovered offset "
                             f"{o:+.3f} s")
            elif s is gz:
                shift[id(s)] = r0 - a
                notes.append(f"WARNING {stream_field(s, 'name')}: no common clock with "
                             f"{stream_field(ref, 'name')}; stream starts aligned — check "
                             "sync")
    t0 = (clean[id(eeg)][0][0] if eeg is not None else
          clean[id(gz)][0][0] + shift.get(id(gz), 0.0) if gz is not None else 0.0)

    raw = gaze = None
    if eeg is not None:
        ts, idx, inf = clean[id(eeg)]
        raw = eeg_from_stream(eeg, ts, idx, inf.get("srate") or 1 / np.median(np.diff(ts)),
                              float(t0), options.get(stream_key(eeg)))
        if inf.get("gaps"):
            raw.set_annotations(mne.Annotations([a - t0 for a, _ in inf["gaps"]],
                                                [b - a for a, b in inf["gaps"]],
                                                ["BAD_acq_gap"] * len(inf["gaps"])))
    if gz is not None:
        ts, idx, inf = clean[id(gz)]
        labels = [c["label"] for c in stream_channels(gz)]
        cols = options.get(stream_key(gz), {}).get("columns")
        mapping = {r: [k for k, c in enumerate(cols) if c == r]
                   for r in ("x", "y", "pupil", "valid")} if cols else None
        gaze = gaze_table(ts + shift.get(id(gz), 0.0) - t0,
                          numeric_samples(gz["time_series"], idx), labels, screen,
                          stream_field(gz, "name") or "gaze", inf.get("srate"), mapping)
    frames, marker_meta = [], {}
    for s in by_role["markers"]:
        ts, idx, _ = clean[id(s)]
        times = ts + shift.get(id(s), 0.0) - t0
        rows = [list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r]
                for r in (s["time_series"][k] for k in idx)]
        if max((len(r) for r in rows), default=1) <= 1:  # one value: the description
            vals = [str(r[0]).strip() if r else "" for r in rows]
            frames.append(pd.DataFrame({"onset": times, "duration": 0.0,
                                        "description": vals})[[v != "" for v in vals]])
            continue
        table = marker_table(rows, [c["label"] for c in stream_channels(s)])
        names, croles = guess_marker_columns(table, stream_field(s, "name"))
        marker_meta[stream_key(s)] = {"names": names, "columns": croles, "end_events": True}
        mo = options.get(stream_key(s), {})
        keep = (table != "").any(axis=1).to_numpy()
        frames.append(marker_events(times[keep], table[keep], mo.get("names") or names,
                                    mo.get("columns") or croles, mo.get("end_events", True)))
    events = pd.concat([pd.DataFrame(columns=["onset", "duration", "description"])] + frames,
                       ignore_index=True).sort_values("onset", kind="stable",
                                                      ignore_index=True)
    events = events.astype({"onset": float, "duration": float})
    info = {"notes": notes, "streams": [stream_info(s, auto[stream_key(s)],
                                                    final[stream_key(s)], clean.get(id(s)),
                                                    marker_meta.get(stream_key(s)),
                                                    options.get(stream_key(s)))
                                        for s in streams],
            "description": "XDF streams: " + ", ".join(
                f"{stream_field(s, 'name')} [{final[stream_key(s)]}"
                f"{'' if final[stream_key(s)] == auto[stream_key(s)] else ', set by user'}]"
                for s in streams)}
    return raw, events, gaze, info


def stream_info(s, auto_role, role, clean, marker_meta, opts) -> dict:
    """Everything the stream dialog shows about one stream."""
    def preview(col):
        v = np.asarray(col)
        if v.dtype.kind in "fiu":
            f = v[np.isfinite(v)] if v.dtype.kind == "f" else v
            return f"{f.min():.4g} … {f.max():.4g}  (mean {f.mean():.4g})" if len(f) \
                else "all missing"
        u = pd.Series([str(x).strip() for x in v]).loc[lambda x: x != ""].value_counts()
        return (f"{len(u)} distinct: " + ", ".join(f"{k} ({n})" for k, n in u.head(4).items())
                + (" …" if len(u) > 4 else "")) if len(u) else "empty"
    chans, ts, series = stream_channels(s), s["time_stamps"], s["time_series"]
    n = len(chans)
    if isinstance(series, np.ndarray) and series.ndim == 2:
        step = max(1, len(series) // 20000)
        cols = [series[::step, k] for k in range(min(n, series.shape[1]))]
    else:
        rows = [list(r) if isinstance(r, (list, tuple, np.ndarray)) else [r]
                for r in list(series)[:20000]]
        width = max([len(r) for r in rows] + [n])
        cols = [[r[k] if k < len(r) else "" for r in rows] for k in range(width)]
    channels = [{"label": (chans[k] if k < n else {}).get("label", f"ch{k + 1}"),
                 "type": (chans[k] if k < n else {}).get("type", ""),
                 "unit": (chans[k] if k < n else {}).get("unit", ""),
                 "preview": preview(c)} for k, c in enumerate(cols)]
    labels = [c["label"] for c in channels]
    auto_opts = {}
    if role == "EEG" or auto_role == "EEG" or (not is_marker_stream(s) and role != "gaze"):
        auto_opts["EEG"] = {"include": [True] * len(channels), "names": labels,
                            "types": [channel_type(c) for c in chans]
                            + ["eeg"] * (len(channels) - n)}
    if not is_marker_stream(s):
        g, cr = gaze_columns(labels, len(channels)), ["extra"] * len(channels)
        for r in ("valid", "pupil", "y", "x"):
            for k in g.get(r, []):
                cr[k] = r
        auto_opts["gaze"] = {"columns": cr}
    if marker_meta:
        auto_opts["markers"] = marker_meta
    elif is_marker_stream(s):
        auto_opts["markers"] = {"names": labels[:1] or ["value"], "columns": ["name"],
                                "end_events": True, "single": True}
    try:
        manufacturer = s["info"]["desc"][0].get("manufacturer", [""])[0]
    except (KeyError, IndexError, TypeError, AttributeError):
        manufacturer = ""
    return {"key": stream_key(s), "name": stream_field(s, "name"),
            "type": stream_field(s, "type"), "channels": n,
            "srate": float(stream_field(s, "nominal_srate", 0) or 0),
            "effective_srate": clean[2].get("srate") if clean else None,
            "format": stream_field(s, "channel_format"), "samples": len(ts),
            "duration": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
            "first": float(ts[0]) if len(ts) else None,
            "source_id": stream_field(s, "source_id"), "hostname": stream_field(s, "hostname"),
            "created_at": stream_field(s, "created_at"), "manufacturer": manufacturer or "",
            "notes": list(clean[2].get("notes", [])) if clean else [], "labels": labels,
            "channel_info": channels, "auto_role": auto_role, "role": role,
            "auto_options": auto_opts, "options": opts or {}}


def describe_streams(path) -> pd.DataFrame:
    """Streams of an XDF file (headers only)."""
    import pyxdf
    return pd.DataFrame([{"stream_id": s["stream_id"], "name": s["name"], "type": s["type"],
                          "channels": s["channel_count"], "srate": s["nominal_srate"],
                          "format": s["channel_format"]}
                         for s in pyxdf.resolve_streams(str(path))])


# =========================================================================== loading: other files
def standard_montage(raw) -> None:
    """Positions from MNE's 10-05 template for channels whose names match."""
    name = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() \
        else "standard_1005"
    try:
        raw.set_montage(name, on_missing="ignore", match_case=False)
    except Exception:  # noqa: BLE001  (e.g. duplicate positions)
        pass


def guess_channel_type(name: str) -> str:
    n = name.upper()
    for key, kind in (("EOG", "eog"), ("ECG", "ecg"), ("EKG", "ecg"), ("EMG", "emg"),
                      ("TRIG", "stim"), ("STI", "stim"), ("STATUS", "stim"), ("ACC", "misc"),
                      ("GYR", "misc"), ("BATT", "misc"), ("COUNTER", "misc"),
                      ("VALID", "misc")):
        if key in n:
            return kind
    return "eeg"


def type_channels_by_name(raw) -> dict:
    """Re-type channels a reader typed as EEG whose names say otherwise (EOG, TRIG, …)."""
    changed = {c: guess_channel_type(c) for c, t in zip(raw.ch_names, raw.get_channel_types())
               if t == "eeg" and guess_channel_type(c) != "eeg"}
    if changed:
        raw.set_channel_types(changed)
    return changed


def gaze_from_raw(raw, screen=(1920, 1080), drop=True) -> pd.DataFrame | None:
    """MNE ``eyegaze``/``pupil`` channels (EyeLink, BIDS) as a gaze table."""
    picks = mne.pick_types(raw.info, eyetrack=True, exclude=[])
    if not len(picks):
        return None
    names = [raw.ch_names[i] for i in picks]
    g = gaze_table(raw.times.copy(), raw.get_data(picks=picks).T, names, screen, "eyetrack",
                   raw.info["sfreq"])
    if drop:
        raw.drop_channels(names)
    return g


def annotations_table(raw) -> pd.DataFrame:
    a = raw.annotations
    off = raw.first_time if a.orig_time is not None else 0.0
    return pd.DataFrame({"onset": a.onset - off, "duration": a.duration,
                         "description": a.description})


def read_file(path, screen=(1920, 1080)):
    """EEG (and any eye-tracking channels) of a file MNE reads, or an EyeLink .asc."""
    path = Path(path)
    if path.name.lower().endswith(".asc"):
        raw = mne.io.read_raw_eyelink(path)
        return None, annotations_table(raw), gaze_from_raw(raw, screen, drop=False), {}
    raw = mne.io.read_raw(path, preload=True)
    return raw, pd.DataFrame(columns=["onset", "duration", "description"]), \
        gaze_from_raw(raw, screen), {}


# ---- BIDS
BIDS_EEG_EXTS = (".vhdr", ".edf", ".bdf", ".set", ".fif", ".cnt", ".mff")
BIDS_ENTITIES = ("subject", "session", "task", "acquisition", "run")


def sidecar(path) -> dict:
    """The JSON sidecar of a BIDS file."""
    path = Path(path)
    for ext in (".tsv.gz", ".tsv", ".vhdr", ".edf", ".bdf", ".set", ".fif"):
        if path.name.endswith(ext):
            js = path.with_name(path.name[: -len(ext)] + ".json")
            return json.loads(js.read_text(encoding="utf-8")) if js.exists() else {}
    return {}


def bids_file(eeg, suffix) -> Path | None:
    """A sidecar of an EEG file (same entities, another suffix, inherited up)."""
    parts = Path(eeg).name.rsplit("_", 1)[0].split("_")
    for k in range(len(parts), 0, -1):
        c = Path(eeg).with_name("_".join(parts[:k]) + f"_{suffix}")
        if c.exists():
            return c
    return None


def electrodes_file(eeg) -> Path | None:
    exact = bids_file(eeg, "electrodes.tsv")
    if exact is not None:
        return exact
    from mne_bids import get_entities_from_fname
    ent = get_entities_from_fname(Path(eeg).name)
    prefix = f"sub-{ent['subject']}" + (f"_ses-{ent['session']}" if ent.get("session") else "")
    found = sorted(Path(eeg).parent.glob(f"{prefix}*_electrodes.tsv"))
    return found[0] if found else None


def channels_tsv(eeg) -> pd.DataFrame:
    """``channels.tsv``: name, type, status, status_description."""
    f = bids_file(eeg, "channels.tsv")
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False) if f else \
        pd.DataFrame(columns=["name"])
    for c in ("status", "status_description", "type"):
        df[c] = df[c] if c in df else ""
    return df


def bids_events(path) -> pd.DataFrame:
    """Every row of an ``events.tsv``, every column kept: ``trial_type`` (else ``value``)
    is the name, the other columns become ``NAME|column=value`` attributes."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "onset" not in df:
        return pd.DataFrame(columns=["onset", "duration", "description"])
    df = df[df["onset"].str.lower() != "n/a"]
    name_col = "trial_type" if "trial_type" in df else "value" if "value" in df else None
    attrs = [c for c in df.columns if c not in ("onset", "duration", "trial_type", "sample")
             and c != name_col]
    safe = lambda v: str(v).replace("|", "/").replace("=", ":").strip()  # noqa: E731
    desc = ["|".join([safe(r[name_col]) if name_col and r[name_col] not in ("", "n/a")
                      else "event"] + [f"{c}={safe(r[c])}" for c in attrs
                                       if r[c] not in ("", "n/a")])
            for _, r in df.iterrows()]
    out = pd.DataFrame({"onset": pd.to_numeric(df["onset"], errors="coerce").to_numpy(float),
                        "duration": pd.to_numeric(df.get("duration", "0"), errors="coerce")
                        .fillna(0.0).to_numpy(float) if "duration" in df else 0.0,
                        "description": desc})
    return out[np.isfinite(out["onset"])].reset_index(drop=True)


def read_bids_eyetrack(path, eeg_meta=None, screen=None):
    """Gaze and messages of a BIDS eye-tracking file (``*_physio.tsv.gz`` + JSON, on the
    EEG's time base via ``StartTime``)."""
    path = Path(path)
    meta = {**(eeg_meta or {}), **sidecar(path)}
    cols = meta.get("Columns")
    df = pd.read_csv(path, sep="\t", header=None if cols else "infer", names=cols,
                     na_values=["n/a", "."]).apply(pd.to_numeric, errors="coerce")
    sfreq = float(meta["SamplingFrequency"]) if meta.get("SamplingFrequency") not in (
        None, "n/a") else None
    start = float(meta.get("StartTime", 0.0) or 0.0)
    labels = [str(c) for c in df.columns]
    tcol = next((c for c in labels if c.lower() in ("timestamp", "time", "eye_timestamp")), None)
    ts = df[tcol].to_numpy(float) if tcol else None
    scale = 1.0
    if ts is not None:
        units = str((meta.get(tcol) or {}).get("Units", "") if isinstance(meta.get(tcol), dict)
                    else "").lower()
        known = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
        if units in known:
            scale = known[units]
        elif sfreq and len(ts) > 1 and np.median(np.diff(ts)) > 0:
            scale = 10.0 ** round(np.log10((1.0 / sfreq) / np.median(np.diff(ts))))
        times = (ts - ts[0]) * scale + start
    elif sfreq:
        times = start + np.arange(len(df)) / sfreq
    else:
        raise ValueError(f"{path.name}: no timestamp column and no SamplingFrequency")
    data_cols = [c for c in labels if c != tcol]
    size = tuple(int(v) for v in (screen or meta.get("ScreenResolution")
                                  or meta.get("ScreenResolutionPx") or (1920, 1080)))[:2]
    gaze = gaze_table(times, df[data_cols].to_numpy(float), data_cols, size,
                      path.name.split("_physio")[0], sfreq)
    events = pd.DataFrame(columns=["onset", "duration", "description"])
    ev_path = path.with_name(path.name.replace("_physio.tsv", "_physioevents.tsv"))
    if ev_path != path and ev_path.exists():
        ev_cols = sidecar(ev_path).get("Columns")
        ev = pd.read_csv(ev_path, sep="\t", header=None if ev_cols else "infer", names=ev_cols,
                         na_values=["n/a"])
        desc_col = next((c for c in ("message", "trial_type", "value") if c in ev), None)
        if "onset" in ev and desc_col:
            on = pd.to_numeric(ev["onset"], errors="coerce").to_numpy(float)
            on = (on - ts[0]) * scale + start if ts is not None else on + start
            dur = pd.to_numeric(ev["duration"], errors="coerce").to_numpy(float) \
                if "duration" in ev else np.zeros(len(ev))
            keep = ev[desc_col].notna().to_numpy()
            events = pd.DataFrame({"onset": on[keep], "description":
                                   ev[desc_col].astype(str)[keep].to_numpy(),
                                   "duration": np.nan_to_num(dur * (scale if ts is not None
                                                                    else 1.0))[keep]})
    return gaze, events


def read_bids_run(root, run, screen=None):
    """EEG (MNE-BIDS; channels.tsv bad channels; electrode positions or the 10-05
    template), every events.tsv column and the eye-tracking files of a BIDS run.
    Returns ``(raw, events, gaze, info)``; ``info["metadata"]`` has the dataset,
    participant and recording information."""
    from mne_bids import BIDSPath, read_raw_bids
    root = Path(root)
    raw = gaze = None
    events = pd.DataFrame(columns=["onset", "duration", "description"])
    meta, eeg_meta = {}, {}
    if run.eeg is not None:
        e = run.entities
        meta = bids_metadata(root, run)
        try:
            raw = read_raw_bids(BIDSPath(root=root, subject=e.get("subject"),
                                         session=e.get("session"), task=e.get("task"),
                                         acquisition=e.get("acquisition"), run=e.get("run"),
                                         datatype=run.eeg.parent.name, suffix="eeg",
                                         extension=run.eeg.suffix))
        except Exception:  # noqa: BLE001  broken sidecars: plain MNE + channels.tsv
            raw = mne.io.read_raw(run.eeg)
            ch = channels_tsv(run.eeg)
            to_mne = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "MISC": "misc",
                      "TRIG": "stim", "STIM": "stim", "RESP": "resp", "GSR": "gsr",
                      "TEMP": "temperature", "PPG": "bio", "HEOG": "eog", "VEOG": "eog"}
            types = {r.name: to_mne[r.type.upper()] for r in ch.itertuples()
                     if r.name in raw.ch_names and r.type.upper() in to_mne}
            if types:
                raw.set_channel_types(types, on_unit_change="ignore")
            raw.info["bads"] = [r.name for r in ch.itertuples()
                                if r.status.lower() == "bad" and r.name in raw.ch_names]
            pl = sidecar(run.eeg).get("PowerLineFrequency")
            if isinstance(pl, (int, float)):
                with raw.info._unlock():
                    raw.info["line_freq"] = float(pl)
        raw.load_data()
        eeg_meta = sidecar(run.eeg)
        ev_file = bids_file(run.eeg, "events.tsv")
        if ev_file is not None and len(tsv := bids_events(ev_file)):
            a = raw.annotations  # keep only BAD/EDGE spans: events.tsv has every column
            keep = np.flatnonzero(is_data_annotation(pd.Series(a.description)).to_numpy())
            raw.set_annotations(a[keep] if len(keep) else
                                mne.Annotations([], [], [], orig_time=a.orig_time))
            events = tsv
        if raw.get_montage() is not None and meta["electrodes_file"]:
            meta["montage_source"] = f"{meta['electrodes_file']} (from the dataset)"
        else:
            standard_montage(raw)
            n = sum(ch["kind"] == 2 and np.isfinite(ch["loc"][:3]).all()
                    and np.abs(ch["loc"][:3]).sum() > 0 for ch in raw.info["chs"])
            meta["montage_source"] = (f"standard 10-05 template by channel name ({n} EEG "
                                      "channels)" if n else "none (names do not match the "
                                      "10-05 template: use Montage)")
        gaze = gaze_from_raw(raw, screen or (1920, 1080))
    messages = []
    for gp in run.gaze:
        g, ev = read_bids_eyetrack(gp, {k: v for k, v in eeg_meta.items()
                                        if k.startswith("Screen")}, screen)
        messages.append(ev)
        if gaze is None or "eye1" in gp.name:
            gaze = g
    if messages:
        events = pd.concat([f for f in messages if len(f)] or [events], ignore_index=True)
    if raw is None and gaze is not None and gaze["t"].iloc[0] != 0:
        t0 = gaze["t"].iloc[0]
        gaze["t"] -= t0
        events = events.assign(onset=events["onset"] - t0)
    desc = f"BIDS {root.name}: " + " ".join(f"{k}-{v}" for k, v in run.entities.items() if v)
    if meta:
        desc += "\n" + describe_metadata(meta)
    return raw, events, gaze, {"metadata": meta, "description": desc}


EEG_SIDECAR_KEYS = (
    "TaskName", "TaskDescription", "PowerLineFrequency", "EEGReference", "EEGGround",
    "EEGPlacementScheme", "CapManufacturer", "CapManufacturersModelName", "Manufacturer",
    "ManufacturersModelName", "SoftwareVersions", "SoftwareFilters", "HardwareFilters",
    "RecordingDuration", "RecordingType", "InstitutionName",
    "InstitutionalDepartmentName", "RecordingTiming")


def bids_metadata(root, run) -> dict:
    """Dataset, participant, eeg.json and channels.tsv information of a run."""
    root = Path(root)
    desc_file = root / "dataset_description.json"
    desc = json.loads(desc_file.read_text(encoding="utf-8")) if desc_file.exists() else {}
    meta = {"dataset": {k: desc[k] for k in ("Name", "BIDSVersion", "Authors", "License",
                                             "DatasetDOI", "HowToAcknowledge")
                        if desc.get(k) not in (None, "", "n/a", ["n/a"], [])},
            "participant": {}, "eeg": {}, "bad_channels": {}, "montage_source": "",
            "electrodes_file": ""}
    sub, part = run.entities.get("subject"), root / "participants.tsv"
    if sub and part.exists():
        df = pd.read_csv(part, sep="\t", dtype=str, keep_default_na=False)
        row = df[df.get("participant_id", pd.Series(dtype=str)) == f"sub-{sub}"]
        if len(row):
            meta["participant"] = {k: v for k, v in row.iloc[0].items()
                                   if k != "participant_id" and v not in ("", "n/a")}
    if run.eeg is not None:
        side = sidecar(run.eeg)
        meta["eeg"] = {k: side[k] for k in EEG_SIDECAR_KEYS if k in side and side[k] != "n/a"}
        ch = channels_tsv(run.eeg)
        meta["bad_channels"] = {r.name: r.status_description if r.status_description not in
                                ("", "n/a") else "status bad in channels.tsv"
                                for r in ch[ch["status"].str.lower() == "bad"].itertuples()}
        el = electrodes_file(run.eeg)
        meta["electrodes_file"] = el.name if el is not None else ""
    return meta


def describe_metadata(meta: dict) -> str:
    """Readable summary of BIDS metadata for the info panel."""
    if not meta:
        return ""
    out, d, e = [], meta.get("dataset", {}), meta.get("eeg", {})
    if d:
        a = d.get("Authors")
        out.append("Dataset: " + " — ".join(x for x in (
            d.get("Name", ""), ", ".join(a) if isinstance(a, list) else a,
            f"licence {d['License']}" if d.get("License") else "", d.get("DatasetDOI", ""))
            if x))
    if meta.get("participant"):
        out.append("Participant: " + ", ".join(f"{k} {v}" for k, v in
                                               meta["participant"].items()))
    dev = " ".join(str(e[k]) for k in ("Manufacturer", "ManufacturersModelName") if k in e)
    cap = " ".join(str(e[k]) for k in ("CapManufacturer", "CapManufacturersModelName") if k in e)
    if dev or cap:
        out.append("Device: " + dev + (f"; cap: {cap}" if cap and cap != dev else ""))
    ref = [f"{label} {e[k]}" for k, label in (("EEGReference", "reference"),
                                              ("EEGGround", "ground"),
                                              ("EEGPlacementScheme", "placement")) if k in e]
    if "PowerLineFrequency" in e:
        ref.append(f"power line {e['PowerLineFrequency']} Hz")
    if ref:
        out.append("Recording: " + ", ".join(ref))
    sf = e.get("SoftwareFilters")
    if isinstance(sf, dict) and sf:
        out.append("Online filters already applied: " + "; ".join(
            f"{n} {f.get('CutoffFrequency', f.get('Frequencies', ''))} Hz"
            + (" (causal)" if f.get("Causal") else "") if isinstance(f, dict) else f"{n}: {f}"
            for n, f in sf.items()))
    rt = e.get("RecordingTiming")
    if isinstance(rt, dict) and "MeasuredSamplingFrequency" in rt:
        out.append(f"Timing: measured rate {rt['MeasuredSamplingFrequency']} Hz"
                   + (f", clock deviation {rt['ClockDeviationPPM']} ppm"
                      if "ClockDeviationPPM" in rt else ""))
    if meta.get("montage_source"):
        out.append("Electrode positions: " + meta["montage_source"])
    bad = meta.get("bad_channels", {})
    plain = [c for c, why in bad.items() if why.startswith("status bad")]
    told = [f"{c} ({why})" for c, why in bad.items() if not why.startswith("status bad")]
    out.append("Bad channels (channels.tsv): " + (", ".join(plain + told) if bad else "none"))
    return "\n".join(out)


def is_data_annotation(descriptions: pd.Series) -> pd.Series:
    """True for annotations about the data (BAD_… spans, EDGE/BAD boundaries)."""
    d = descriptions.astype(str).str.upper()
    return d.str.startswith("BAD") | d.str.startswith("EDGE")


# ---- several files as one
ENTITY_NAMES = {"sub": "subject", "ses": "session", "task": "bids_task", "acq": "acq",
                "run": "run"}
SEGMENT_COLUMNS = ("subject", "session", "bids_task", "acq", "run", "file")


def entities_from_path(path) -> dict:
    """BIDS entities in a file name and its folders (``sub-01_ses-2_run-1_eeg.xdf``)."""
    p, out = Path(path), {}
    pattern = re.compile(r"(?:^|[_/\\])(sub|ses|task|acq|run)-([A-Za-z0-9.+]+?)(?=_|\.|/|\\|$)")
    for part in list(p.parent.parts[-3:]) + [p.name]:
        for key, val in pattern.findall("_" + part):
            out[ENTITY_NAMES[key]] = val
    out["file"] = p.name
    return out


def join(parts: list[tuple]) -> tuple:
    """Join recordings ``(raw, events, gaze, entities)`` end to end.

    EEG: the channels all parts share, the first part's rate (rates < 0.1 % apart are
    taken as equal, others resampled), bad channels from every part, ``BAD boundary``
    annotations at the joins (MNE). Gaze and events are shifted onto the joined time
    base; each part becomes a segment with its entities. Returns
    ``(raw, events, gaze, segments, notes)``."""
    notes, starts, t = [], [], 0.0
    raws = [p[0] for p in parts]
    if any(r is not None for r in raws) and not all(r is not None for r in raws):
        raise ValueError("Cannot combine files with and without EEG")
    raw = None
    if raws[0] is not None:
        common = [c for c in raws[0].ch_names if all(c in r.ch_names for r in raws[1:])]
        if not common:
            raise ValueError("The files have no EEG channel names in common")
        dropped = sorted({c for r in raws for c in r.ch_names} - set(common))
        if dropped:
            notes.append(f"channels not in every file left out: {', '.join(dropped[:12])}"
                         f"{' …' if len(dropped) > 12 else ''}")
        sf, types = raws[0].info["sfreq"], dict(zip(raws[0].ch_names,
                                                    raws[0].get_channel_types()))
        bads = sorted({b for r in raws for b in r.info["bads"] if b in common}, key=common.index)
        if any(set(r.info["bads"]) & set(common) != set(bads) for r in raws):
            notes.append(f"bad channels combined from all files: {', '.join(bads)}")
        ready = []
        for r in raws:
            r = r.copy().load_data().pick(common)
            r.info["bads"] = list(bads)
            ann = annotations_table(r)
            r.set_annotations(None)
            r.set_meas_date(None)
            r.set_annotations(mne.Annotations(ann.onset, ann.duration, ann.description))
            d = abs(r.info["sfreq"] - sf)
            if d and d / sf < 1e-3:  # the same amplifier's effective rates differ by ppm
                with r.info._unlock():
                    r.info["sfreq"] = sf
                if not any(n.startswith("rates aligned") for n in notes):
                    notes.append(f"rates aligned to {sf:.4f} Hz (< 0.1 % apart)")
            elif d:
                notes.append(f"resampled {r.info['sfreq']:g} → {sf:g} Hz")
                r.resample(sf)
            r.set_channel_types({c: types[c] for c, k in zip(r.ch_names, r.get_channel_types())
                                 if types[c] != k})
            starts.append(t)
            t += r.n_times / sf
            ready.append(r)
        raw = mne.concatenate_raws(ready)
        if raws[0].get_montage() is not None:
            raw.set_montage(raws[0].get_montage(), on_missing="ignore")
    else:
        for p in parts:
            starts.append(t)
            t += part_duration(p) + 1.0
    gazes = [(g, s) for (_, _, g, _), s in zip(parts, starts) if g is not None and len(g)]
    gaze = None
    if gazes:
        cols = set.intersection(*[set(g.columns) for g, _ in gazes])
        gaze = pd.concat([g[[c for c in g.columns if c in cols]].assign(t=g["t"] + s)
                          for g, s in gazes], ignore_index=True)
        gaze.attrs.update(gazes[0][0].attrs)
        if len(gazes) != len(parts):
            notes.append(f"gaze in {len(gazes)} of {len(parts)} files")
    events = pd.concat([pd.DataFrame(columns=["onset", "duration", "description"])]
                       + [ev.assign(onset=ev["onset"] + s) for (_, ev, _, _), s
                          in zip(parts, starts) if len(ev)], ignore_index=True)
    segments = []
    for p, s in zip(parts, starts):
        for seg in p[3].get("segments") or [{"start": 0.0, "end": part_duration(p),
                                             **p[3].get("entities", {})}]:
            segments.append({**seg, "start": seg["start"] + s, "end": seg["end"] + s})
    return raw, events, gaze, segments, notes


def part_duration(part) -> float:
    raw, _, gaze, _ = part
    ends = ([raw.times[-1]] if raw is not None else []) + \
        ([float(gaze["t"].iloc[-1])] if gaze is not None and len(gaze) else [])
    return float(max(ends)) if ends else 0.0


def segment_columns(onsets, segments) -> dict:
    """subject/session/run/file of the segment each event falls in."""
    if not segments:
        return {}
    i = np.clip(np.searchsorted([s["start"] for s in segments], onsets, side="right") - 1,
                0, len(segments) - 1)
    out = {}
    for c in SEGMENT_COLUMNS:
        vals = [s.get(c) for s in segments]
        if any(v not in (None, "") for v in vals):
            out[c] = np.array(["" if vals[k] is None else vals[k] for k in i], dtype=object)
    return out


def segment_label(seg: dict, with_subject: bool = True) -> str:
    """``sub-01 · ses-1 · run-2`` (the entities a segment has)."""
    parts = [f"{pre}{seg[k]}" for k, pre in (("subject", "sub-"), ("session", "ses-"),
                                              ("run", "run-"))
             if seg.get(k) not in (None, "") and (with_subject or k != "subject")]
    return " · ".join(parts) or str(seg.get("file", ""))


def subject_spans(segments) -> list[dict]:
    """Consecutive segments of the same subject merged: subject, label, start, end."""
    out: list[dict] = []
    for s in segments or []:
        if s.get("subject") in (None, ""):
            continue
        if out and out[-1]["subject"] == s["subject"] and abs(out[-1]["end"] - s["start"]) < 1e-6:
            out[-1]["end"] = s["end"]
        else:
            out.append({"subject": s["subject"], "label": f"sub-{s['subject']}",
                        "start": float(s["start"]), "end": float(s["end"])})
    return out


class BidsRun:
    """One run of a BIDS dataset: its EEG file and/or eye-tracking files."""

    def __init__(self, entities: dict, eeg=None, gaze=None, datatype="eeg"):
        self.entities, self.eeg, self.gaze, self.datatype = entities, eeg, gaze or [], datatype

    @property
    def key(self) -> tuple:
        return tuple(self.entities.get(k) or "" for k in BIDS_ENTITIES)

    @property
    def label(self) -> str:
        e = self.entities
        parts = [f"{p}-{e[k]}" for k, p in (("task", "task"), ("acquisition", "acq"),
                                            ("run", "run")) if e.get(k)]
        mods = (["EEG"] if self.eeg else []) + (["gaze"] if self.gaze else [])
        return " ".join(parts) + f"  [{'+'.join(mods)}]"


def scan_bids(root) -> list[BidsRun]:
    """Every EEG / eye-tracking run of a BIDS dataset."""
    from mne_bids import get_entities_from_fname
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    def eyetrack(p: Path) -> bool:
        if "_eyetrack.tsv" in p.name:
            return True
        if not p.name.endswith(("_physio.tsv.gz", "_physio.tsv")):
            return False
        meta = sidecar(p)
        cols = [str(c).lower() for c in meta.get("Columns", [])]
        return meta.get("PhysioType") == "eyetrack" or "eye" in p.name.split(
            "_recording-")[-1][:8] or any("coordinate" in c or "gaze" in c for c in cols)
    runs: dict = {}
    for p in sorted(root.rglob("sub-*")):
        rel = p.relative_to(root).parts[:-1]
        if not p.is_file() or any(x in ("derivatives", "sourcedata", "code")
                                  or x.startswith(".") for x in rel):
            continue
        is_eeg = p.name.endswith("_eeg" + p.suffix) and p.suffix in BIDS_EEG_EXTS
        if not is_eeg and not eyetrack(p):
            continue
        ent = get_entities_from_fname(p.name, on_error="ignore")
        run = runs.setdefault(tuple(ent.get(k) or "" for k in BIDS_ENTITIES),
                              BidsRun({k: ent.get(k) for k in BIDS_ENTITIES},
                                      datatype=p.parent.name))
        if is_eeg:
            run.eeg = p
        else:
            run.gaze.append(p)
    return sorted(runs.values(), key=lambda r: r.key)


def dataset_description(root) -> dict:
    f = Path(root) / "dataset_description.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


# =========================================================================== gaze events
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


#: detection settings for common kinds of gaze data: (label, parameters, explanation)
PRESETS = {
    "standard": (
        "pymovements defaults (raw gaze)", DetectionParams(),
        "pymovements' own settings, with about 40 px for its 1° dispersion: for raw, "
        "unfiltered gaze (e.g. Gazepoint BPOG, EyeLink, Tobii)."),
    "gazepoint_fpog": (
        "Gazepoint, fixation-filtered gaze (FPOG)",
        DetectionParams(fix_dispersion=100.0, fix_min_dur=0.06, sac_threshold_factor=6.0,
                        sac_min_dur=0.006, blink_min_dur=0.05, blink_max_dur=0.5,
                        velocity_method="neighbors"),
        "For Gazepoint GP3 / GP3 HD data recorded with only the fixation point of gaze "
        "(FPOG), which wanders within a fixation: a 100 px dispersion and 60 ms minimum "
        "match the tracker's own fixations (FPOGID); saccades use central-difference "
        "velocity, as smoothing blurs them at 150 Hz. Saccade timing is usable, but "
        "amplitude and peak velocity are not (FPOG is a running average)."),
}


def suggest_preset(gaze) -> str:
    """Gazepoint gaze with FPOG but no BPOG channels is fixation-filtered; anything else
    is treated as raw gaze."""
    cols = {c.lower() for c in (gaze.columns if gaze is not None else [])}
    return "gazepoint_fpog" if "ch_fpogx" in cols and "ch_bpogx" not in cols else "standard"


def preset_name(p: DetectionParams) -> str | None:
    return next((k for k, (_, q, _) in PRESETS.items() if q == p), None)


@dataclass
class GazeEvents:
    """Fixations (onset, offset, duration, x, y), saccades (… amplitude, peak_velocity)
    and blinks (onset, offset, duration), in seconds."""
    fixations: pd.DataFrame
    saccades: pd.DataFrame
    blinks: pd.DataFrame

    @classmethod
    def empty(cls):
        return cls(pd.DataFrame(columns=["onset", "offset", "duration", "x", "y"]),
                   pd.DataFrame(columns=["onset", "offset", "duration", "amplitude",
                                         "peak_velocity"]),
                   pd.DataFrame(columns=["onset", "offset", "duration"]))


def gaze_rate(gaze) -> float:
    return float(gaze.attrs.get("sfreq") or 1.0 / float(np.median(np.diff(gaze["t"]))))


def detect_gaze_events(gaze, p: DetectionParams = DetectionParams()) -> GazeEvents:
    """Fixations (I-DT), saccades (Engbert & Kliegl) and blinks (from the pupil, or from
    tracking loss) with pymovements, on the screen-pixel gaze. pymovements gets the
    sample number as time axis (durations in samples), so any sampling rate works;
    onsets are mapped back to the sample times."""
    import warnings

    import polars as pl
    import pymovements as pm
    t, xy = gaze["t"].to_numpy(), gaze[["x", "y"]].to_numpy().T
    sf = gaze_rate(gaze)
    n = lambda s, least=1: max(int(round(s * sf)), least)  # noqa: E731  seconds → samples
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "no events were detected" and the like
        pg = pm.gaze.from_numpy(time=np.arange(len(t), dtype=np.int64), time_unit="ms",
                                pixel=xy, position=xy, experiment=pm.Experiment(sampling_rate=sf))
        pupil = gaze["pupil"].to_numpy(float) if "pupil" in gaze else \
            np.where(np.isfinite(xy).all(0), 1.0, np.nan)  # NaN = tracking lost
        pg.samples = pg.samples.with_columns(pl.Series("pupil", pupil))
        pg.pos2vel(method=p.velocity_method)
        pg.detect("idt", dispersion_threshold=p.fix_dispersion,
                  minimum_duration=n(p.fix_min_dur, 2))
        pg.detect("microsaccades", threshold_factor=p.sac_threshold_factor,
                  minimum_duration=n(p.sac_min_dur), name="saccade")
        pg.detect("blink", minimum_duration=n(p.blink_min_dur),
                  maximum_duration=n(p.blink_max_dur), minimum_gap=n(0.005))
        names = set(pg.events.frame["name"].to_list())
        if "fixation" in names:
            pg.compute_event_properties("location", name="fixation")
        if "saccade" in names:
            pg.compute_event_properties(["amplitude", "peak_velocity"], name="saccade")
    ev = pg.events.frame
    at = lambda i: t[np.clip(np.asarray(i, np.int64), 0, len(t) - 1)]  # noqa: E731

    def table(name, extra=()):
        e = ev.filter(pl.col("name") == name)
        on = at(e["onset"].to_numpy()) if len(e) else np.empty(0)
        off = at(e["offset"].to_numpy()) if len(e) else np.empty(0)
        df = pd.DataFrame({"onset": on, "offset": off, "duration": off - on})
        for c in extra:
            df[c] = e[c].to_numpy().astype(float) if len(e) and c in e.columns else np.nan
        return df
    fix = table("fixation")
    loc = ev.filter(pl.col("name") == "fixation")
    xy_f = np.array(loc["location"].to_list(), float).reshape(-1, 2) if len(loc) and \
        "location" in loc.columns else np.empty((0, 2))
    fix["x"], fix["y"] = xy_f[:, 0], xy_f[:, 1]
    return GazeEvents(fix, table("saccade", ("amplitude", "peak_velocity")), table("blink"))


def clean_pupil(gaze, blinks, pad: float = 0.05):
    """Pupil with blinks (± ``pad`` s) linearly interpolated."""
    if "pupil" not in gaze:
        return None
    t, pupil = gaze["t"].to_numpy(), gaze["pupil"].to_numpy(float).copy()
    bad = ~np.isfinite(pupil)
    for on, off in blinks[["onset", "offset"]].to_numpy():
        bad |= (t >= on - pad) & (t < off + pad)
    if (~bad).sum() >= 2:
        pupil[bad] = np.interp(t[bad], t[~bad], pupil[~bad])
    return pupil


def heatmap(fixations, screen_size, sigma: float = 40.0, downsample: int = 4) -> np.ndarray:
    """Duration-weighted fixation density map, (height, width) / ``downsample``."""
    from scipy.ndimage import gaussian_filter
    w, h = (int(np.ceil(s / downsample)) for s in screen_size)
    hm = np.zeros((h, w))
    if len(fixations):
        x, y = fixations["x"].to_numpy(), fixations["y"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        np.add.at(hm, (np.clip((y[ok] / downsample).astype(int), 0, h - 1),
                       np.clip((x[ok] / downsample).astype(int), 0, w - 1)),
                  fixations["duration"].to_numpy()[ok])
        hm = gaussian_filter(hm, sigma / downsample)
    return hm


def make_aois(specs: dict | None):
    """Rectangular areas of interest ``{name: (left, top, width, height)}`` (px) as a
    pymovements AOI table."""
    if not specs:
        return None
    import polars as pl
    import pymovements as pm
    names = list(specs)
    box = np.array([specs[n] for n in names], float).reshape(-1, 4)
    return pm.stimulus.TextStimulus(
        pl.DataFrame({"aoi": names, "x": box[:, 0], "y": box[:, 1], "width": box[:, 2],
                      "height": box[:, 3]}),
        aoi_column="aoi", start_x_column="x", start_y_column="y", width_column="width",
        height_column="height")


def fixation_aois(fixations, aois) -> np.ndarray:
    """The AOI each fixation falls in ("" for none), with pymovements' map_to_aois."""
    out = np.full(len(fixations), "", dtype=object)
    if aois is None or not len(fixations):
        return out
    import warnings

    import polars as pl
    import pymovements as pm
    ev = pm.Events(pl.DataFrame({"name": ["fixation"] * len(fixations),
                                 "onset": np.arange(len(fixations)),
                                 "offset": np.arange(len(fixations)) + 1,
                                 "location": fixations[["x", "y"]].to_numpy(float).tolist()}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ev.map_to_aois(aois, verbose=False)
    out[:] = ["" if h is None else str(h) for h in ev.frame["aoi"].to_list()]
    return out


def aoi_statistics(fixations, aois) -> pd.DataFrame:
    """Per AOI: fixations, dwell time, time to first fixation, mean fixation duration."""
    names = fixation_aois(fixations, aois)
    rows = []
    for name in (aois.aois["aoi"].to_list() if aois is not None else []):
        sel = fixations[names == name]
        rows.append({"aoi": name, "n_fixations": int(len(sel)),
                     "dwell_time": float(sel["duration"].sum()) if len(sel) else 0.0,
                     "first_fixation": float(sel["onset"].min()) if len(sel) else np.nan,
                     "mean_fix_dur": float(sel["duration"].mean()) if len(sel) else np.nan})
    return pd.DataFrame(rows)


def gaze_summary(ev: GazeEvents, duration: float) -> pd.DataFrame:
    f, s, b, dur = ev.fixations, ev.saccades, ev.blinks, max(duration, 1e-9)
    return pd.DataFrame([
        ("Fixations", len(f)), ("Fixation rate (/s)", len(f) / dur),
        ("Mean fixation duration (ms)", 1000 * f["duration"].mean() if len(f) else np.nan),
        ("Saccades", len(s)),
        ("Mean saccade amplitude (px)", s["amplitude"].mean() if len(s) else np.nan),
        ("Mean peak velocity (px/s)", s["peak_velocity"].mean() if len(s) else np.nan),
        ("Blinks", len(b)), ("Blink rate (/min)", 60 * len(b) / dur)],
        columns=["measure", "value"])


# =========================================================================== events
BASE_COLUMNS = ["onset", "duration", "name", "kind", "description"]
#: names of detected gaze events; markers with these names get a "_marker" suffix
GAZE_EVENT_NAMES = ("fixation", "saccade", "blink")
# "<PHASE>_START" … "<PHASE>_END"; a separator is required, so names that merely
# end in "on"/"off" (e.g. "fixation") are not phase markers
_START = re.compile(r"^(.+?)[_\- ](START|BEGIN|ON)$", re.I)
_END = re.compile(r"^(.+?)[_\- ](END|STOP|OFF)$", re.I)


def _safe_key(k: str) -> str:
    k = re.sub(r"\W+", "_", k.strip()).strip("_").lower() or "attr"
    if k[0].isdigit():
        k = "a_" + k
    if k in BASE_COLUMNS or k in ("phase", "aoi", "x", "y"):
        k = "attr_" + k
    return k


def parse_description(desc: str) -> tuple[str, dict]:
    """``"NAME|k=v|k2=v2"`` → ("NAME", {k: v, k2: v2}). Other strings → (desc, {})."""
    parts = str(desc).split("|")
    attrs = {}
    rest = []
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            attrs[_safe_key(k)] = v.strip()
        elif p.strip():
            rest.append(p.strip())
    if rest:
        attrs["extra"] = "|".join(rest)
    return parts[0].strip(), attrs


def _to_numeric_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            # convert only if every non-empty value is numeric
            nonempty = df[c].notna() & (df[c].astype(str) != "")
            if nonempty.any() and conv[nonempty].notna().all():
                df[c] = conv
    return df


def _phase_spans(markers: pd.DataFrame) -> list[tuple[str, float, float, dict]]:
    """(phase, start, end, attrs) spans from *_START / *_END marker pairs and from
    markers that have a duration (e.g. paired start/stop rows of a marker stream)."""
    spans, open_ = [], {}
    for on, dur, desc in markers[["onset", "duration", "description"]].itertuples(index=False):
        name, attrs = parse_description(desc)
        if dur and float(dur) > 0:
            spans.append((name, float(on), float(on) + float(dur), attrs))
        if m := _START.match(name):
            open_[m.group(1).upper()] = (on, attrs)
        elif (m := _END.match(name)) and m.group(1).upper() in open_:
            key = m.group(1).upper()
            start, a = open_.pop(key)
            spans.append((key, start, on, a))
    end = float(markers["onset"].max()) if len(markers) else np.inf
    for key, (start, a) in open_.items():  # never closed: extends to the end
        spans.append((key, start, max(end, start) + 1e9, a))
    return spans


def event_table(markers, gaze_events: GazeEvents | None = None, aois=None,
                segments=None) -> pd.DataFrame:
    """One row per marker and gaze event: onset, duration, name, kind, description, the
    marker's columns (``NAME|key=value``), the phase (innermost ``*_START``…``*_END``
    span or marker with a duration) and its attributes, the AOI of fixations and the
    subject/session/run/file of the part each event falls in."""
    markers = markers if markers is not None else pd.DataFrame(
        columns=["onset", "duration", "description"])
    rows = []
    for on, dur, desc in markers[["onset", "duration", "description"]].itertuples(index=False):
        name, attrs = parse_description(desc)
        if name.lower() in GAZE_EVENT_NAMES:  # e.g. a "fixation" cross marker
            name = f"{name}_marker"
        rows.append({"onset": float(on), "duration": float(dur or 0.0), "name": name,
                     "kind": "marker", "description": str(desc), **attrs})
    tables = [pd.DataFrame(rows)] if rows else []
    attr_cols = sorted({k for r in rows for k in r} - set(BASE_COLUMNS))

    if gaze_events is not None:
        f = gaze_events.fixations.copy()
        if len(f):
            f["name"] = "fixation"
            f["kind"] = "fixation"
            f["description"] = "fixation"
            if aois is not None:  # pymovements AOI table
                f["aoi"] = fixation_aois(f, aois)
            tables.append(f.drop(columns=["offset"]))
        s = gaze_events.saccades.copy()
        if len(s):
            s["name"] = s["kind"] = s["description"] = "saccade"
            tables.append(s.drop(columns=[c for c in ("offset", "x0", "y0", "x1", "y1")
                                          if c in s]))
        b = gaze_events.blinks.copy()
        if len(b):
            b["name"] = b["kind"] = b["description"] = "blink"
            tables.append(b.drop(columns=["offset"]))

    if not tables:
        return pd.DataFrame(columns=BASE_COLUMNS + ["phase"])
    df = pd.concat(tables, ignore_index=True).sort_values("onset", ignore_index=True)

    # ---- context from phases: the innermost (latest-starting) span decides the
    # phase and fills attributes first; outer spans fill what is still empty.
    # Inherited attributes never overwrite an event's own value.
    spans = sorted(_phase_spans(markers), key=lambda x: -x[1])
    df["phase"] = ""
    if spans:
        onset = df["onset"].to_numpy()
        phase = np.full(len(df), "", dtype=object)
        cols: dict[str, np.ndarray] = {}
        for key, start, end, attrs in spans:
            lo, hi = np.searchsorted(onset, start, "left"), np.searchsorted(onset, end, "right")
            if hi <= lo:
                continue
            seg = phase[lo:hi]
            seg[seg == ""] = key
            for k, v in attrs.items():
                if k not in cols:
                    base = df[k].to_numpy(dtype=object, copy=True) if k in df else \
                        np.full(len(df), None, dtype=object)
                    cols[k] = base
                    attr_cols.append(k)
                c = cols[k][lo:hi]
                empty = np.array([x is None or (isinstance(x, float) and np.isnan(x)) or x == ""
                                  for x in c], dtype=bool)
                c[empty] = v
        df["phase"] = phase
        for k, arr in cols.items():
            df[k] = arr
    # ---- which file / subject / session / run each event comes from
    seg_cols: list[str] = []
    if segments:
        
        for k, v in segment_columns(df["onset"].to_numpy(float), segments).items():
            if k in df:  # an event's own value wins
                own = df[k]
                empty = own.isna() | (own.astype(str) == "")
                df.loc[empty, k] = v[empty.to_numpy()]
            else:
                df[k] = v
            seg_cols.append(k)
    df = _to_numeric_columns(df, set(attr_cols) - set(seg_cols))  # entities stay text
    order = BASE_COLUMNS + ["phase"] + [c for c in df.columns
                                        if c not in BASE_COLUMNS and c != "phase"]
    return df[order]


#: operators of a filter condition (column, operator, value)
OPERATORS = ("=", "≠", "<", "≤", ">", "≥", "contains", "one of")


def _cond_mask(s: pd.Series, op: str, value: str, col: str) -> pd.Series:
    value = str(value).strip()
    if op == "contains":
        return s.astype(str).str.contains(value, case=False, regex=False, na=False)
    if op == "one of":
        wanted = [v.strip() for v in value.split(",") if v.strip()]
        if pd.api.types.is_numeric_dtype(s):
            try:
                return s.isin([float(v) for v in wanted])
            except ValueError:
                pass
        return s.astype(str).isin(wanted)
    num = pd.to_numeric(s, errors="coerce")
    try:
        v = float(value)
        numeric = num.notna().any()
    except ValueError:
        v, numeric = value, False
    if op in ("=", "≠"):
        if numeric and pd.api.types.is_numeric_dtype(s):
            m = num == v
        else:
            m = s.astype(str) == value
        return m if op == "=" else ~m
    if not numeric:
        raise ValueError(f"“{col} {op} {value}” compares numbers: use a number "
                         "(or =, ≠, contains, one of for text)")
    return {"<": num < v, "≤": num <= v, ">": num > v, "≥": num >= v}[op].fillna(False)


def apply_conditions(df: pd.DataFrame, conditions) -> pd.DataFrame:
    """Keep rows matching every (column, operator, value) condition (see OPERATORS)."""
    out = df
    for col, op, value in conditions or ():
        if not col or (value in (None, "") and op != "≠"):
            continue
        if col not in out:
            raise ValueError(f"No column “{col}” in the event table")
        if op not in OPERATORS:
            raise ValueError(f"Unknown operator {op!r}")
        out = out[_cond_mask(out[col], op, value, col).to_numpy()]
    return out


def describe_conditions(conditions) -> str:
    return " and ".join(f"{c} {op} {v}" for c, op, v in conditions or () if c and v != "")


def filter_events(df: pd.DataFrame, text: str = "", query: str = "",
                  names: set[str] | None = None, conditions=None) -> pd.DataFrame:
    """Apply column ``conditions`` [(column, operator, value), …], a regex on
    ``description``, a pandas ``query`` and a name whitelist.

    Raises ``ValueError`` with a readable message for invalid filters.
    """
    out = apply_conditions(df, conditions)
    if text.strip():
        try:
            out = out[out["description"].astype(str).str.contains(text, case=False, regex=True)]
        except re.error as e:
            raise ValueError(f"Invalid text pattern: {e}") from e
    if query.strip():
        try:
            out = out.query(query, engine="python")
        except Exception as e:  # pandas raises many types here
            raise ValueError(f"Invalid filter expression: {e}") from e
    if names is not None:
        out = out[out["name"].isin(names)]
    return out


def first_event_time(table: pd.DataFrame) -> float | None:
    """Onset of the first marker (not gaze events, file boundaries or BAD_ spans)."""
    if table is None or not len(table) or "kind" not in table:
        return None
    m = table[(table["kind"] == "marker")]
    name = m["name"].astype(str)
    m = m[~name.str.upper().str.startswith(("BAD", "EDGE"))]
    return float(m["onset"].min()) if len(m) else None


# =========================================================================== EEG basics
def has_positions(inst) -> bool:
    """True if at least 4 EEG channels of an MNE object have sensor locations."""
    locs = np.array([ch["loc"][:3] for ch in inst.info["chs"]
                     if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH])
    return len(locs) > 0 and int((np.isfinite(locs).all(1) &
                                  (np.abs(locs).sum(1) > 0)).sum()) >= 4


def eeg_picks(raw: mne.io.BaseRaw) -> np.ndarray:
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True,
                           misc=False, stim=False, exclude=[])
    if not len(picks):  # e.g. unknown channel types from XDF
        picks = mne.pick_types(raw.info, misc=True, exclude=[])
    return picks


def parse_reference(ref, raw: mne.io.BaseRaw):
    """Normalise a reference spec → None, "average" or a list of EEG channel names.

    Accepts None/"keep", "average"/"avg"/"car", a channel name, a comma- or
    space-separated string of names, or a list. Matching is case-insensitive.
    """
    if ref is None:
        return None
    if isinstance(ref, str):
        r = ref.strip()
        if not r or r.lower() in ("keep", "none", "as recorded"):
            return None
        if r.lower() in ("average", "avg", "car", "common average"):
            return "average"
        names = [n for n in r.replace(";", ",").replace(",", " ").split() if n]
    else:
        names = list(ref)
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    lower = {n.lower(): n for n in eeg}
    out, missing = [], []
    for n in names:
        (out if n.lower() in lower else missing).append(lower.get(n.lower(), n))
    if missing:
        raise ValueError(f"Reference channel(s) not found among EEG channels: "
                         f"{', '.join(missing)}")
    return out




# =========================================================================== preprocessing
@dataclass
class BadChannelParams:
    ptp_max_uv: float = 250.0      # a window above this peak-to-peak is "noisy"
    ptp_min_uv: float = 1.0        # a window below this is "flat"
    window: float = 1.0            # s
    max_bad_fraction: float = 0.2  # channel is bad above this fraction of bad windows
    z_thresh: float = 4.0          # robust z of log median ptp across channels
    l_freq: float | None = 0.1     # filtered like the preprocessing, so drifts and line
    h_freq: float | None = 30.0    # noise don't count as "noisy"
    notch: float | None = 50.0


def eeg_names(raw) -> list[str]:
    """The EEG channels (all data channels when nothing is typed EEG)."""
    return [raw.ch_names[i] for i in (mne.pick_types(raw.info, eeg=True, exclude=[])
                                      if "eeg" in raw else eeg_picks(raw))]


def start_time(raw, tmin) -> float:
    """``tmin`` (e.g. the first event) on the sample grid; 0 when it leaves < 1 s."""
    i = int(round(tmin * raw.info["sfreq"])) if tmin and tmin > 0 else 0
    return i / raw.info["sfreq"] if i < raw.n_times - int(raw.info["sfreq"]) else 0.0


def detect_bad_channels(raw, p: BadChannelParams | None = None, tmin=None) -> pd.DataFrame:
    """Peak-to-peak per EEG channel in ``p.window`` s windows of the filtered data (from
    ``tmin``): bad when more than ``p.max_bad_fraction`` of the windows are noisy or flat,
    or when its median peak-to-peak is a robust-z outlier among channels."""
    p = p or BadChannelParams()
    sf, rows = raw.info["sfreq"], []
    win, first = max(int(round(p.window * sf)), 2), int(round(start_time(raw, tmin) * sf))
    n_win = (raw.n_times - first) // win
    if n_win < 1:
        raise ValueError("Recording is shorter than one window")
    names = eeg_names(raw)
    for k in range(0, len(names), 8):  # 8 channels at a time bounds the memory
        x = raw.get_data(picks=names[k:k + 8], start=first) * 1e6
        lo = p.l_freq if p.l_freq and p.l_freq > 0 else None
        hi = p.h_freq if p.h_freq and p.h_freq < sf / 2 else None
        if lo and lo < IIR_BELOW_HZ:
            x, lo = mne.filter.filter_data(x, sf, lo, None, method="iir",
                                           iir_params=IIR_PARAMS), None
        if lo or hi:
            x = mne.filter.filter_data(x, sf, lo, hi)
        if p.notch and p.notch < sf / 2:
            x = mne.filter.notch_filter(x, sf, np.arange(p.notch, sf / 2, p.notch))
        ptp = np.ptp(x[:, :n_win * win].reshape(len(x), n_win, win), axis=2)
        rows += [{"channel": n, "median_ptp_uv": float(np.median(v)),
                  "noisy_pct": 100 * float((v > p.ptp_max_uv).mean()),
                  "flat_pct": 100 * float((v < p.ptp_min_uv).mean())}
                 for n, v in zip(names[k:k + 8], ptp)]
    df = pd.DataFrame(rows)
    lg = np.log(np.maximum(df["median_ptp_uv"].to_numpy(), 1e-6))
    df["z"] = (lg - np.median(lg)) / (np.median(np.abs(lg - np.median(lg))) * 1.4826 or 1.0)
    limit = 100 * p.max_bad_fraction
    df["reason"] = [", ".join(w for w, bad in (("flat", r.flat_pct > limit),
                                               (f"ptp > {p.ptp_max_uv:g} µV", r.noisy_pct > limit),
                                               ("outlier", abs(r.z) > p.z_thresh)) if bad)
                    for r in df.itertuples()]
    df["bad"] = df["reason"] != ""
    return df


#: below this high-pass cut-off an FIR filter gets very long: a zero-phase Butterworth
IIR_BELOW_HZ = 0.5
IIR_PARAMS = dict(order=4, ftype="butter", output="sos")


def filter_raw(raw, l_freq, h_freq, notch):
    """Band-pass and notch filter the EEG/EOG/ECG/EMG channels, in place. A high-pass
    below 0.5 Hz is a zero-phase 4th-order Butterworth, run on each stretch between
    boundaries (``Raw.filter``'s IIR path fails on a boundary at the first sample, which
    joined runs and cropped data have)."""
    picks = [t for t in ("eeg", "eog", "ecg", "emg") if t in raw]
    nyq = raw.info["sfreq"] / 2
    l_freq, h_freq = l_freq if l_freq and l_freq > 0 else None, \
        h_freq if h_freq and h_freq < nyq else None
    if not picks:
        return raw
    if l_freq and l_freq < IIR_BELOW_HZ:
        from mne.annotations import _annotations_starts_stops
        starts, stops = _annotations_starts_stops(raw, ("edge", "bad_acq_skip"), invert=True)
        idx = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True, exclude=[])
        for a, b in zip(*((starts, stops) if len(starts) else ([0], [raw.n_times]))):
            if b - a >= 2:
                raw._data[:, a:b] = mne.filter.filter_data(
                    raw._data[:, a:b], raw.info["sfreq"], l_freq, None, picks=idx,
                    method="iir", iir_params=IIR_PARAMS)
        l_freq = None
    if l_freq or h_freq:
        raw.filter(l_freq, h_freq, picks=picks)
    if notch and notch < nyq:
        raw.notch_filter(np.arange(notch, nyq, notch), picks=picks)
    return raw


def filtered_copy(raw, l_freq, h_freq, notch, picks=None, tmin=None):
    """A filtered copy of ``raw`` (``picks`` only, from ``tmin`` s)."""
    out = raw.copy().load_data()
    if picks is not None:
        out.pick(picks)
    if start_time(raw, tmin):
        out.crop(tmin=start_time(raw, tmin), include_tmax=True)
    return filter_raw(out, l_freq, h_freq, notch)


# =========================================================================== ICA
@dataclass
class ICAParams:
    """ICA as in the ERP guide (neuraldatascience.io): fitted on a 1–30 Hz copy cut into
    1 s segments, the bad segments found by AutoReject, enough components for 99 % of the
    variance, ``random_state=42``, every 3rd sample."""
    n_components: float = 0.99    # < 1: fraction of variance; ≥ 1: number of components
    method: str = "picard"        # picard (extended, as ICLabel expects) | fastica | infomax
    l_freq: float = 1.0           # ICA is fitted on ≥ 1 Hz high-passed data
    h_freq: float | None = 30.0   # … and ≤ 30 Hz low-passed
    clean: str = "autoreject"     # bad 1 s segments: "autoreject" | "threshold"
    reject_uv: float | None = 500.0  # threshold cleaning: skip segments above this ptp
    target_sfreq: float = 128.0   # resample the fitting data to about this rate
    decim: int = 3                # fit on every 3rd sample
    max_samples: int = 300_000    # … or fewer, so long recordings stay bounded
    autoreject_fraction: float = 1.0  # share of the segments AutoReject is fitted on
    max_iter: int = 500           # stop a fit that does not converge
    seed: int = 42
    iclabel: bool = True          # label the components with ICLabel (needs positions)


#: AutoReject's fit needs roughly this many times the memory of the epochs it is fitted
#: on (measured: 1.16 GB of epochs → +14.2 GB)
AUTOREJECT_MEMORY_FACTOR = 12.5


def fit_subset(n: int, fraction: float, seed: int = 42):
    """The epochs AutoReject is fitted on: all (``fraction`` 1) or a reproducible random
    share (quicker, less memory, but on noisy data the result depends on the draw)."""
    if fraction >= 1 or n < 2:
        return slice(None)
    return np.sort(np.random.default_rng(seed).choice(n, max(2, int(round(fraction * n))),
                                                      replace=False))


def autoreject(epochs, seed: int = 42, fraction: float = 1.0):
    """AutoReject (``n_interpolate=[1, 2, 4]``, as in the ERP guide), fitted on all or a
    share of the epochs."""
    import warnings

    from autoreject import AutoReject
    ar = AutoReject(n_interpolate=[1, 2, 4], random_state=seed, n_jobs=1, verbose=False,
                    picks=mne.pick_types(epochs.info, eeg=True, eog=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ar.fit(epochs[fit_subset(len(epochs), fraction, seed)])


def fit_times(fit_raw) -> np.ndarray:
    """Recording times of the samples ICA was fitted on (it may start at an event)."""
    return fit_raw.times + getattr(fit_raw, "recording_t0", 0.0)


def fit_ica(raw, bads=(), h_freq=None, notch=None, p: ICAParams | None = None, tmin=None):
    """ICA on the good, non-flat EEG channels of an average-referenced 1–30 Hz copy of
    ``raw`` from ``tmin`` (the first event), cut into 1 s segments; the bad segments
    (AutoReject, else a peak-to-peak threshold) are left out. Returns the ICA and the data
    it was fitted on (with ``fit_steps``: every decision, for exported scripts)."""
    import warnings
    p, notes = p or ICAParams(), []
    t0, h = start_time(raw, tmin), p.h_freq if h_freq is None else h_freq
    sf = raw.info["sfreq"]
    good_eeg = [c for c in eeg_names(raw) if c not in set(bads)]
    sd = raw.get_data(picks=good_eeg, start=int(round(t0 * sf)),
                      stop=min(raw.n_times, int(round(t0 * sf)) + int(60 * sf))).std(1)
    chans = [c for c, s in zip(good_eeg, sd) if s > 1e-12]  # a reference channel is flat
    if len(chans) < 3:
        raise ValueError("ICA needs at least 3 good EEG channels")
    ica_raw = filtered_copy(raw, max(p.l_freq, 0.1), h, notch, picks=chans, tmin=t0)
    ica_raw.info["bads"] = []  # the channels kept by the user, whatever the file says
    ica_raw.set_eeg_reference("average", projection=False)  # as ICLabel expects
    ica_raw.recording_t0 = t0
    decim = max(1, int(ica_raw.info["sfreq"] // p.target_sfreq))
    if decim > 1:
        ica_raw.resample(ica_raw.info["sfreq"] / decim)
    segments = mne.Epochs(ica_raw, mne.make_fixed_length_events(ica_raw, duration=1.0),
                          tmin=0.0, tmax=1.0, baseline=None, preload=True)
    how, good = "threshold", np.ones(len(segments), bool)
    if p.clean == "autoreject" and len(segments) >= 20 and has_positions(segments):
        try:
            ar = autoreject(segments, p.seed, p.autoreject_fraction)
            good, how = ~np.asarray(ar.get_reject_log(segments).bad_epochs, bool), "autoreject"
            notes.append(f"AutoReject left out {(~good).sum()} of {len(good)} 1 s segments")
        except Exception as e:  # noqa: BLE001
            notes.append(f"AutoReject failed ({type(e).__name__}), threshold used")
    elif p.clean == "autoreject":
        notes.append("AutoReject needs electrode positions and ≥ 20 segments: "
                     "peak-to-peak threshold used")
    if how == "threshold" and p.reject_uv:
        kept = segments.copy().drop_bad(reject=dict(eeg=p.reject_uv * 1e-6))
        good[:] = False
        good[np.searchsorted(segments.selection, kept.selection)] = True
    if good.sum() < max(10, len(good) // 10):
        notes.insert(0, f"only {good.sum()} of {len(good)} 1 s segments were clean (noisy "
                        "channels kept?), so ICA was fitted on all of them")
        good[:], how = True, "all"
    method = p.method
    if method == "picard":
        try:
            import picard  # noqa: F401
        except ImportError:
            method = "infomax"
    fit_params = (dict(ortho=False, extended=True) if method == "picard" else
                  dict(extended=True) if method == "infomax" else None)
    nc = p.n_components if p.n_components < 1 else int(min(p.n_components, len(chans) - 1))
    ica = mne.preprocessing.ICA(n_components=nc, method=method, fit_params=fit_params,
                                max_iter=p.max_iter, rng=p.seed)
    fit_on = segments[np.flatnonzero(good)]
    every = max(p.decim, int(np.ceil(len(fit_on) * len(fit_on.times) / p.max_samples)))
    if every > p.decim:
        notes.append(f"fitted on every {every}th sample")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(fit_on, decim=every)
    if getattr(ica, "n_iter_", 0) >= p.max_iter:
        notes.append(f"stopped after {p.max_iter} iterations without converging — fewer "
                     "components or more bad channels may help")
    ica_raw.ic_labels = ica_raw.ic_proba = None
    if p.iclabel:
        try:
            ica_raw.ic_labels, ica_raw.ic_proba = iclabel(raw, ica, chans, notch, t0)
        except Exception as e:  # noqa: BLE001
            notes.append(f"ICLabel not run: {e}")
    ica_raw.fit_note = "; ".join(notes)
    ica_raw.fit_steps = dict(
        chans=list(chans), l_freq=max(p.l_freq, 0.1), h_freq=h, notch=notch,
        tmin=t0 or None, resample=ica_raw.info["sfreq"] if decim > 1 else None,
        n_components=nc, method=method, fit_params=fit_params, max_iter=p.max_iter,
        seed=p.seed, segments=how, reject_uv=p.reject_uv,
        autoreject_fraction=p.autoreject_fraction, decim=every)
    return ica, ica_raw


def iclabel(raw, ica, chans, notch=None, tmin=None) -> tuple[list[str], np.ndarray]:
    """ICLabel class and probability of every component (MNE-ICALabel), on a 1–100 Hz
    average-referenced copy of the ICA channels (what ICLabel was trained on)."""
    import warnings

    from mne_icalabel import label_components
    if not has_positions(raw.copy().pick(chans)):
        raise ValueError("the EEG channels have no electrode positions (load a montage)")
    lab_raw = filtered_copy(raw, 1.0, min(100.0, raw.info["sfreq"] / 2 - 1), notch,
                            picks=chans, tmin=tmin)
    lab_raw.info["bads"] = []
    lab_raw.set_eeg_reference("average", projection=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = label_components(lab_raw, ica, method="iclabel")
    return list(out["labels"]), np.asarray(out["y_pred_proba"], float)


FRONTAL = ("Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4")
#: EOG stand-ins without EOG electrodes (ERP guide: Fp1, Fp2, F7, F8)
EOG_PROXIES = ("Fp1", "Fp2", "F7", "F8")


def score_blink_components(ica, fit_raw, blinks=None, eog_channels=None, z_thresh=3.0,
                           max_ic=2, iclabel_p=0.7) -> pd.DataFrame:
    """Blink candidates among the ICA components, suggested by any of:

    * ``eog_score``: |correlation| with the EOG channels (``ICA.find_bads_eog``; without
      EOG, Fp1/Fp2/F7/F8 stand in), the z threshold stepping down from 3.5 by 0.05 until
      ``max_ic`` components are found (ERP guide; not below 2.0);
    * ``blink_t``: max |mean| / SEM of the component around the eye tracker's blinks
      (−0.2…0.5 s), a robust-z outlier above ``z_thresh``;
    * ICLabel "eye blink" with probability ≥ ``iclabel_p``."""
    df = pd.DataFrame({"component": np.arange(ica.n_components_)})
    names = fit_raw.ch_names
    lower = {c.lower(): c for c in names}
    eog = [c for c in (eog_channels or []) if c in names] or \
        [c for c in names if mne.channel_type(fit_raw.info, names.index(c)) == "eog"]
    df.attrs["proxy"] = not eog
    if not eog:
        eog = [lower[c.lower()] for c in EOG_PROXIES if c.lower() in lower] or \
            [lower[c.lower()] for c in FRONTAL if c.lower() in lower][:2]
    flagged, df["eog_score"] = set(), np.nan
    if eog:
        z = 3.5
        while True:
            idx, scores = ica.find_bads_eog(fit_raw, ch_name=eog, threshold=z,
                                            measure="zscore")
            if len(idx) >= max_ic or z <= 2.0:
                break
            z = round(z - 0.05, 2)
        df.attrs["eog_z"] = z
        df["eog_score"] = np.abs(np.atleast_2d(np.asarray(scores))).max(0)
        flagged |= {int(i) for i in idx}
    df["blink_t"] = np.nan
    times, sf = fit_times(fit_raw), fit_raw.info["sfreq"]
    on = np.searchsorted(times, blinks["onset"].to_numpy(float)) if blinks is not None \
        else np.array([], int)
    i0, i1 = int(round(-0.2 * sf)), int(round(0.5 * sf))
    on = on[(on + i0 >= 0) & (on + i1 < len(times))][:3000]
    if len(on) >= 5 and (blinks is not None and len(blinks) >= 5):
        src = ica.get_sources(fit_raw).get_data()
        src = (src - src.mean(1, keepdims=True)) / (src.std(1, keepdims=True) + 1e-20)
        seg = src[:, on[:, None] + np.arange(i0, i1)[None, :]]       # comp × blink × time
        seg = seg - seg[:, :, :-i0].mean(2, keepdims=True)
        t = np.abs(seg.mean(1) / (seg.std(1) / np.sqrt(len(on)) + 1e-20)).max(1)
        df["blink_t"] = t
        mad = np.median(np.abs(t - np.median(t))) * 1.4826 or 1e-9
        flagged |= set(np.flatnonzero(((t - np.median(t)) / mad > z_thresh) & (t > 5)).tolist())
    labels, proba = getattr(fit_raw, "ic_labels", None), getattr(fit_raw, "ic_proba", None)
    ok = labels is not None and len(labels) == len(df)
    df["iclabel"] = labels if ok else ""
    df["iclabel_p"] = proba if ok else np.nan
    flagged |= set(np.flatnonzero((df["iclabel"] == "eye blink") &
                                  (df["iclabel_p"] >= iclabel_p)).tolist())
    df["suggested"] = df["component"].isin(flagged)
    df.attrs["eog_channels"] = eog
    return df


# =========================================================================== preprocessing
@dataclass
class PreprocParams:
    bads: list[str] = field(default_factory=list)
    l_freq: float | None = 0.1    # default band-pass 0.1–30 Hz with a 50 Hz notch
    h_freq: float | None = 30.0
    notch: float | None = 50.0
    reference: str | list[str] | None = None
    ica_exclude: list[int] = field(default_factory=list)
    interpolate_bads: bool = False

    def describe(self) -> str:
        band = f"{self.l_freq or 0:g}–{self.h_freq:g} Hz" if self.h_freq else \
            (f"high-pass {self.l_freq:g} Hz" if self.l_freq else "no band-pass")
        return " · ".join(
            [f"{len(self.bads)} bad channel(s)"
             + (" interpolated" if self.bads and self.interpolate_bads else ""), band]
            + ([f"notch {self.notch:g} Hz"] if self.notch else [])
            + ([f"ICA −{len(self.ica_exclude)} comp."] if self.ica_exclude else [])
            + [f"ref: {self.reference or 'as recorded'}"])


def run_preprocessing(raw, p: PreprocParams, ica=None):
    """Filter a copy of ``raw``, remove the ICA components, interpolate the bad channels
    and re-reference."""
    data = filtered_copy(raw, p.l_freq, p.h_freq, p.notch)
    data.info["bads"] = [b for b in p.bads if b in data.ch_names]
    if ica is not None and p.ica_exclude:
        missing = [c for c in ica.ch_names if c not in data.ch_names]
        if missing:
            raise ValueError(f"ICA was fitted on channels that no longer exist: {missing}")
        ica = ica.copy()
        ica.exclude = list(p.ica_exclude)
        ica.apply(data)
    if p.interpolate_bads and data.info["bads"]:
        data.interpolate_bads(reset_bads=True)
    ref = parse_reference(p.reference, data)
    if ref:
        data.set_eeg_reference(ref, projection=False)
    return data


def blink_locked_average(raw, blinks, channels, tmin=-0.3, tmax=0.6):
    """Average of ``channels`` around eye-tracker blink onsets (µV), to check ICA."""
    chans = [c for c in channels if c in raw.ch_names and c not in raw.info["bads"]]
    if blinks is None or not len(blinks) or not chans:
        return None
    sf = raw.info["sfreq"]
    i0, i1 = int(round(tmin * sf)), int(round(tmax * sf))
    on = np.round(blinks["onset"].to_numpy(float) * sf).astype(int)
    on = on[(on + i0 >= 0) & (on + i1 < raw.n_times)][:2000]
    if not len(on):
        return None
    seg = raw.get_data(picks=chans).mean(0)[on[:, None] + np.arange(i0, i1)] * 1e6
    seg -= seg[:, :-i0].mean(1, keepdims=True) if i0 < 0 else 0
    return np.arange(i0, i1) / sf, seg.mean(0)


def frontal_channels(raw, bads=()) -> list[str]:
    """Up to two good frontal channels (Fp1, Fp2, …), where blinks show most."""
    lower = {c.lower(): c for c in raw.ch_names}
    return [lower[c.lower()] for c in FRONTAL
            if c.lower() in lower and lower[c.lower()] not in bads][:2]


def blink_check(raw, clean, p: PreprocParams, blinks) -> tuple:
    """Blink-locked average of the frontal channels before (filtered only) and after
    preprocessing, to see what ICA removed."""
    if clean is None:
        return None, None
    fp = frontal_channels(raw, p.bads)
    before = filtered_copy(raw, p.l_freq, p.h_freq, p.notch, picks=fp) if fp else raw
    return blink_locked_average(before, blinks, fp), blink_locked_average(clean, blinks, fp)


def psd_check(raw, clean) -> tuple:
    """Power spectra of the good EEG channels before and after preprocessing."""
    eeg = [clean.ch_names[i] for i in mne.pick_types(clean.info, eeg=True)]
    good = [c for c in clean.ch_names if c not in clean.info["bads"] and c in eeg]
    return compute_psd(raw.copy().pick(good) if good else raw, 60.0), compute_psd(clean, 60.0)


# =========================================================================== epoching
@dataclass
class EpochSettings:
    """Defaults follow the ERP guide (neuraldatascience.io): −0.1 … 1.0 s, no fixed
    amplitude threshold, AutoReject interpolates bad channels per epoch and drops the
    epochs it cannot repair."""
    tmin: float = -0.1
    tmax: float = 1.0
    reject_uv: float | None = None    # peak-to-peak rejection (EEG)
    autoreject: bool = True           # AutoReject (local): interpolate, then drop
    autoreject_fraction: float = 1.0  # share of the epochs AutoReject is fitted on
    flat_uv: float | None = None      # reject epochs flatter than this
    drop_blinks: bool = False         # drop epochs overlapping an eye-tracker blink
    delay_ms: float = 0.0             # EEG recording delay: + moves events later

    def describe(self) -> str:
        f = self.autoreject_fraction
        return "".join([f"{self.tmin:g} … {self.tmax:g} s",
                        f", reject ptp > {self.reject_uv:g} µV" if self.reject_uv else "",
                        f", flat < {self.flat_uv:g} µV" if self.flat_uv else "",
                        ", AutoReject" + (f" (fitted on {f:.0%})" if f < 1 else "")
                        if self.autoreject else "",
                        ", no blinks" if self.drop_blinks else "",
                        f", EEG delay {self.delay_ms:+g} ms" if self.delay_ms else ""])


def epochs_memory(raw, groups: dict, settings) -> float:
    """Bytes building the epochs takes: events × EEG channels × samples × 8, plus about
    12× the epochs AutoReject is fitted on while it learns its thresholds."""
    n_ch = len(mne.pick_types(raw.info, eeg=True, exclude="bads")) or len(raw.ch_names)
    n_samp = int((settings.tmax - settings.tmin) * raw.info["sfreq"]) + 1
    data = float(sum(len(t) for t in groups.values())) * n_ch * n_samp * 8
    if getattr(settings, "autoreject", False):
        data *= 1 + AUTOREJECT_MEMORY_FACTOR * settings.autoreject_fraction
    return data


def overlaps(onsets, tmin, tmax, intervals) -> np.ndarray:
    """True for each onset whose window [onset+tmin, onset+tmax] overlaps an interval."""
    out = np.zeros(len(onsets), bool)
    if intervals is None or not len(intervals):
        return out
    order = np.argsort(intervals["onset"].to_numpy(float))
    s = intervals["onset"].to_numpy(float)[order]
    e = np.maximum.accumulate(s + intervals["duration"].to_numpy(float)[order])
    k = np.searchsorted(s, onsets + tmax, side="right") - 1  # last one starting before
    out[k >= 0] = e[k[k >= 0]] > (onsets + tmin)[k >= 0]
    return out


def epochs_from_table(raw, table, tmin=-0.2, tmax=0.8, baseline=(None, 0), reject_uv=None,
                      picks="eeg") -> mne.Epochs:
    """MNE Epochs locked to the rows of an event table, with the rows as metadata and
    their ``name`` as event id (``epochs["fixation"]``, ``epochs['cond == "AI"']``).
    Rows whose window leaves the recording, or on an already used sample, are dropped
    (counted in ``epochs.selection_counts``)."""
    rows, sf = table.reset_index(drop=True), raw.info["sfreq"]
    samples = np.round(rows["onset"].to_numpy(float) * sf).astype(int) + raw.first_samp
    inside = (samples >= raw.first_samp + int(np.ceil(-tmin * sf))) & \
        (samples <= raw.last_samp - int(np.ceil(tmax * sf)))
    rows, samples = rows[inside], samples[inside]
    first = np.sort(np.unique(samples, return_index=True)[1])  # MNE needs unique samples
    counts = {"outside recording": int((~inside).sum()), "same sample": len(rows) - len(first)}
    rows, samples = rows.iloc[first].reset_index(drop=True), samples[first]
    if not len(rows):
        err = ValueError("No selected events fall inside the EEG recording")
        err.counts = counts
        raise err
    event_id = {n: i + 1 for i, n in enumerate(sorted(rows["name"].astype(str).unique()))}
    events = np.c_[samples, np.zeros_like(samples), rows["name"].astype(str).map(event_id)]
    if picks == "eeg":
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        picks = picks if len(picks) else eeg_picks(raw)
    epochs = mne.Epochs(raw, events.astype(int), event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, picks=picks, preload=True,
                        reject=dict(eeg=reject_uv * 1e-6) if reject_uv else None,
                        metadata=rows.astype({c: object for c in rows.columns
                                              if isinstance(rows[c].dtype, pd.StringDtype)}))
    epochs.selection_counts = counts
    return epochs


def build_group_epochs(raw, groups: dict, settings: EpochSettings, blinks=None):
    """One ``mne.Epochs`` per group (metadata = the group's event rows), and a summary per
    group: selected; dropped for a blink, a window leaving the recording or another
    event on the same sample; rejected by amplitude (peak-to-peak / flat) or a BAD
    annotation; kept; and the channels behind the amplitude rejections."""
    out, rows = {}, []
    for name, tab in groups.items():
        tab = tab.assign(onset=tab["onset"] + settings.delay_ms / 1000.0) \
            if settings.delay_ms else tab.copy()  # the EEG lags the events
        row = {"group": name, "selected": len(tab), "blink": 0, "outside recording": 0,
               "same sample": 0, "rejected (amplitude)": 0, "rejected (BAD annotation)": 0,
               "kept": 0, "rejected by": ""}
        if settings.drop_blinks and blinks is not None and len(tab):
            hit = overlaps(tab["onset"].to_numpy(float), settings.tmin, settings.tmax, blinks)
            row["blink"], tab = int(hit.sum()), tab[~hit]
        if len(tab):
            try:
                ep = epochs_from_table(raw, tab.assign(group=name), settings.tmin,
                                       settings.tmax, baseline=None,
                                       reject_uv=settings.reject_uv)
            except ValueError as e:
                ep = None
                row.update(getattr(e, "counts", {"outside recording": len(tab)}))
            if ep is not None:
                if settings.flat_uv:
                    ep.drop_bad(flat=dict(eeg=settings.flat_uv * 1e-6))
                row.update(ep.selection_counts)
                dropped = [r for r in ep.drop_log if r and r != ("IGNORED",)]
                bad = [any(str(x).upper().startswith(("BAD", "EDGE")) for x in r)
                       for r in dropped]
                row["rejected (BAD annotation)"] = sum(bad)
                row["rejected (amplitude)"] = len(dropped) - sum(bad)
                row["kept"] = len(ep)
                causes = pd.Series([x for r in ep.drop_log for x in r if x in ep.ch_names],
                                   dtype=object).value_counts(sort=False)
                row["rejected by"] = ", ".join(f"{c} ({n})" for c, n in causes.sort_values(
                    ascending=False, kind="stable").head(6).items())
                if len(ep):
                    out[name] = ep
        rows.append(row)
    summary = pd.DataFrame(rows)
    if settings.autoreject and out:
        out, summary = apply_autoreject(out, summary, settings.autoreject_fraction)
    return out, summary


def apply_autoreject(epochs: dict, summary, fraction=1.0, seed=42):
    """AutoReject (ERP guide), fitted once on the epochs of every group (all, or a share),
    then each group cleaned: bad channels interpolated within an epoch, the epochs that
    cannot be repaired dropped. Needs electrode positions and ≥ 10 epochs."""
    import warnings
    summary = summary.assign(**{"rejected (AutoReject)": 0, "interpolated (AutoReject)": 0})
    every = list(epochs.values())
    n = sum(len(e) for e in every)
    why = ("the EEG channels have no positions" if not has_positions(every[0]) else
           "fewer than 10 epochs" if n < 10 else "")
    if why:
        summary.attrs["autoreject"] = f"not applied: {why}"
        return epochs, summary
    # fitted on one array of (a share of) every group's epochs: their event codes may
    # clash, which mne.concatenate_epochs refuses
    pick = fit_subset(n, fraction, seed)
    pick = np.arange(n) if isinstance(pick, slice) else pick
    starts = np.cumsum([0] + [len(e) for e in every])
    data = np.concatenate([e.get_data(item=pick[(pick >= a) & (pick < b)] - a)
                           for e, a, b in zip(every, starts, starts[1:])])
    ar = autoreject(mne.EpochsArray(data, every[0].info, tmin=every[0].tmin, baseline=None),
                    seed)
    out = {}
    for g, ep in epochs.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ep = ep.copy().load_data()
            log, clean = ar.get_reject_log(ep), ar.transform(ep)
        i = summary.index[summary["group"] == g][0]
        summary.loc[i, "rejected (AutoReject)"] = int(np.sum(log.bad_epochs))
        summary.loc[i, "interpolated (AutoReject)"] = int((log.labels[~log.bad_epochs] == 2).sum())
        summary.loc[i, "kept"] = len(clean)
        if len(clean):
            out[g] = clean
    summary.attrs["autoreject"] = "applied"
    return out, summary


# =========================================================================== EEG: montages, triggers, ERP
def compute_psd(raw, fmax: float = 60.0, tmin=None, tmax=None):
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    return raw.compute_psd(fmax=min(fmax, raw.info["sfreq"] / 2), tmin=tmin, tmax=tmax,
                           picks=picks if len(picks) else eeg_picks(raw))


def add_gaze_annotations(raw, ev, kinds=("blinks", "saccades")) -> None:
    """Annotate blinks/saccades/fixations on ``raw`` (e.g. to reject segments)."""
    label = {"blinks": "BAD_blink", "saccades": "saccade", "fixations": "fixation"}
    df = pd.concat([getattr(ev, k).assign(description=label[k]) for k in kinds])
    if len(df):
        orig = raw.annotations.orig_time
        raw.set_annotations(raw.annotations + mne.Annotations(
            df["onset"].to_numpy() + (raw.first_time if orig is not None else 0.0),
            df["duration"].to_numpy(), df["description"].to_numpy(), orig_time=orig))


#: Folders with channel-layout montages: text files listing electrode names in amplifier
#: input order (``<number> <name>`` or just ``<name>`` per line, ``#`` comments). Positions
#: come from an MNE template, set with a ``# positions: <template>`` line (default
#: standard_1005). Files in the user folder show up as built-ins too.
MONTAGE_DIRS = (Path(__file__).parent / "montages", Path.home() / ".brainstudio" / "montages")


def layout_montages() -> dict[str, Path]:
    """Channel-layout montages shipped with brAInStudio or in ~/.brainstudio/montages."""
    return {f.stem: f for d in MONTAGE_DIRS if d.is_dir() for f in sorted(d.glob("*.txt"))}


def builtin_montages() -> list[str]:
    """Layout montages first, then MNE's built-in montages."""
    extra = layout_montages()
    return list(extra) + [m for m in mne.channels.get_builtin_montages() if m not in extra]


def read_channel_layout(path) -> tuple[list[str], str]:
    """Electrode names in input order and the MNE template that gives their positions."""
    names, template = [], "standard_1005"
    for line in map(str.strip, Path(path).read_text(encoding="utf-8").splitlines()):
        key, _, val = line.lstrip("# ").partition(":")
        if line.startswith("#") and key.strip().lower() == "positions" and val.strip():
            template = val.strip()
        elif line and not line.startswith("#"):
            names.append(line.split()[-1])
    return names, template


def montage_from_layout(names: list[str], template: str = "standard_1005"):
    """DigMontage whose channel order is ``names``, positioned from ``template``."""
    if template == "standard_1005" and "colin27_1005" in mne.channels.get_builtin_montages():
        template = "colin27_1005"  # MNE's new name for the 10-05 template
    pos = mne.channels.make_standard_montage(template).get_positions()
    lookup = {k.upper(): v for k, v in pos["ch_pos"].items()}
    missing = [n for n in names if n.upper() not in lookup]
    if missing:
        raise ValueError(f"No position in {template} for: {', '.join(missing)}")
    return mne.channels.make_dig_montage(
        ch_pos={n: lookup[n.upper()] for n in names}, nasion=pos["nasion"],
        lpa=pos["lpa"], rpa=pos["rpa"], coord_frame=pos["coord_frame"])


def montage_from_source(source: str):
    """Montage name (layout montage or ``mne.channels.get_builtin_montages``) or file."""
    layouts, p = layout_montages(), Path(source)
    if source in layouts:
        return montage_from_layout(*read_channel_layout(layouts[source]))
    if not p.exists():  # a montage name (incl. deprecated aliases MNE still accepts)
        try:
            return mne.channels.make_standard_montage(source)
        except ValueError as e:
            raise ValueError(f"'{source}' is neither a file nor a built-in montage") from e
    if p.suffix.lower() == ".fif":
        return mne.channels.read_dig_fif(p)
    if p.name.endswith("_electrodes.tsv"):  # BIDS electrodes file
        df = pd.read_csv(p, sep="\t", na_values=["n/a"]).dropna(subset=["x", "y", "z"])
        pos = df[["x", "y", "z"]].to_numpy(float)
        if np.nanmax(np.abs(pos)) > 1.0:  # mm / cm → m
            pos = pos / (1000.0 if np.nanmax(np.abs(pos)) > 30 else 100.0)
        return mne.channels.make_dig_montage(dict(zip(df["name"].astype(str), pos)),
                                             coord_frame="head")
    return mne.channels.read_custom_montage(p)


def apply_montage(raw, montage, new_names: list[str] | None = None, set_types=True) -> dict:
    """Rename the channels (in recording order), set their types from their names and
    apply ``montage``, in place. Returns n_renamed, n_positioned and missing (EEG
    channels without a position)."""
    mapping = {old: new.strip() for old, new in zip(raw.ch_names, new_names or [])
               if new.strip() and new.strip() != old}
    if len(set(mapping.values())) != len(mapping) or \
            set(mapping.values()) & (set(raw.ch_names) - set(mapping)):
        raise ValueError("Channel names must be unique")
    raw.rename_channels(mapping)
    if set_types:
        changed = {c: guess_channel_type(c) for c, t in zip(raw.ch_names,
                                                           raw.get_channel_types())
                   if t == "eeg" and guess_channel_type(c) != "eeg"}
        if changed:
            raw.set_channel_types(changed)
    if montage is not None:
        raw.set_montage(montage, match_case=False, on_missing="ignore")
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    loc = np.array([raw.info["chs"][i]["loc"][:3] for i in eeg]).reshape(-1, 3)
    placed = np.isfinite(loc).all(1) & (np.abs(loc).sum(1) > 0)
    return {"n_renamed": len(mapping), "n_positioned": int(placed.sum()),
            "missing": [raw.ch_names[i] for i in eeg[~placed]]}


TRIGGER_WORDS = ("trig", "status", "marker", "event", "stim", "sti ")
_UNIT_BACK = {"uV": 1e6, "µV": 1e6, "μV": 1e6, "mV": 1e3, "nV": 1e9, "V": 1.0}


def guess_trigger_channel(raw) -> str | None:
    """A stim channel, else one named like a trigger (TRIG, Status, …), else None."""
    stims = [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True)]
    return stims[0] if stims else next(
        (c for c in raw.ch_names if any(w in c.lower() + " " for w in TRIGGER_WORDS)), None)


def trigger_codes(raw, channel: str, mask: int | None = None, mode="codes") -> pd.DataFrame:
    """Events on a trigger channel: onset (s) and code, with ``mne.find_events``.

    ``codes``: the channel holds integer codes (EDF/BDF trigger or Status channels): the
    reader's unit scaling is undone, values rounded, and every change to a non-zero code
    is an event (``mask`` keeps only some bits). ``threshold``: an analog pulse channel
    (e.g. photodiode) is cut half-way between its resting level (median) and its pulse
    extreme; each pulse onset is code 1."""
    if channel not in raw.ch_names:
        raise ValueError(f"No channel named {channel!r}")
    x = raw.get_data(picks=[channel])[0]
    if mode == "threshold":
        base = float(np.median(x))  # pulses are rare: the median is the resting level
        up, down = float(x.max()) - base, base - float(x.min())
        if max(up, down) <= 0:
            raise ValueError(f"{channel} is flat")
        data = (x > base + up / 2 if up >= down else x < base - down / 2).astype(float)
    else:  # from the file's unit, whatever the channel's type is now
        data = np.round(x * _UNIT_BACK.get(getattr(raw, "_orig_units", {}).get(channel, ""), 1.0))
    trig = mne.io.RawArray(data[None], mne.create_info(["TRIG"], raw.info["sfreq"], "stim"),
                           first_samp=raw.first_samp)
    ev = mne.find_events(trig, stim_channel="TRIG", shortest_event=1, consecutive=True,
                         initial_event=True, mask=mask or None, mask_type="and")
    return pd.DataFrame({"onset": (ev[:, 0] - raw.first_samp) / raw.info["sfreq"],
                         "code": ev[:, 2].astype(int)})


@dataclass
class Component:
    """An ERP component: time window, channel cluster and polarity."""
    name: str
    tmin: float
    tmax: float
    channels: tuple[str, ...]
    polarity: int  # +1 positive-going (peak = maximum), -1 negative-going


#: Conventional windows and sites (starting points: they depend on paradigm and
#: population). The same set as the lab's Rhino ERP.
COMPONENT_PRESETS = {
    "P1": Component("P1", 0.08, 0.13, ("O1", "O2", "Oz", "PO7", "PO8"), +1),
    "N1": Component("N1", 0.08, 0.15, ("Fz", "FCz", "Cz"), -1),
    "N170": Component("N170", 0.13, 0.20, ("PO7", "PO8", "P7", "P8"), -1),
    "P2": Component("P2", 0.15, 0.25, ("Fz", "FCz", "Cz"), +1),
    "N2": Component("N2", 0.20, 0.35, ("Fz", "FCz", "Cz"), -1),
    "MMN": Component("MMN", 0.10, 0.25, ("Fz", "FCz", "Cz"), -1),
    "P300": Component("P300", 0.25, 0.50, ("Pz", "CPz", "Cz"), +1),
    "N400": Component("N400", 0.30, 0.50, ("Cz", "CPz", "Pz"), -1),
    "LPP": Component("LPP", 0.40, 0.80, ("Pz", "CPz", "Cz"), +1),
}


def component_window(groups: dict, comp: Component) -> tuple[list[str], np.ndarray, str]:
    """The component's channels among the epochs', its window as a mask on the epoch
    times, and its label."""
    first = next(iter(groups.values()))
    chans = [c for c in comp.channels if c in first.ch_names]
    if comp.tmin >= comp.tmax:
        raise ValueError("The component window must start before it ends")
    if not chans:
        raise ValueError(f"None of {', '.join(comp.channels)} are among the epoch channels "
                         "(renamed, or marked bad?)")
    m = (first.times >= comp.tmin) & (first.times <= comp.tmax)
    if not m.any():
        raise ValueError("The component window lies outside the epochs")
    return chans, m, (f"{comp.name} {comp.tmin * 1000:.0f}–{comp.tmax * 1000:.0f} ms "
                      f"{'+' if comp.polarity > 0 else '−'} @ {'+'.join(chans)}")


def peak(ev, chans, comp) -> tuple[float, float]:
    """Peak amplitude (µV) and latency (ms) of the cluster mean, in the component's
    polarity (``Evoked.get_peak``)."""
    roi = mne.channels.combine_channels(ev, {"roi": [ev.ch_names.index(c) for c in chans]},
                                        method="mean")
    _, lat, amp = roi.get_peak(tmin=comp.tmin, tmax=comp.tmax, return_amplitude=True,
                               mode="neg" if comp.polarity < 0 else "pos", strict=False)
    return float(amp) * 1e6, float(lat) * 1000


def component_analysis(groups: dict, comp: Component) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``comp`` in every group, epochs as observations (baseline-corrected epochs): per
    epoch the window mean of the cluster mean (µV); per group n, mean ± SEM and the peak
    of the average. Two groups: Welch t, Mann-Whitney U, Hedges' g and the peak
    differences; more: one-way ANOVA. Returns (summary, per-epoch table)."""
    chans, _, label = component_window(groups, comp)
    rows, per, vals = [], [], {}
    for g, ep in groups.items():
        w = (ep.times >= comp.tmin) & (ep.times <= comp.tmax)
        if not w.any():
            raise ValueError("The component window lies outside the epochs")
        vals[g] = amps = (ep.get_data(picks=chans).mean(1) * 1e6)[:, w].mean(1)
        pa, pl = peak(ep.average(), chans, comp)
        rows.append({"group": g, "n": len(amps), "window mean µV": float(amps.mean()),
                     "sem": float(amps.std(ddof=1) / np.sqrt(len(amps))) if len(amps) > 1
                     else np.nan, "peak µV": pa, "peak latency ms": pl})
        md = ep.metadata
        per.append(pd.DataFrame({"group": g, "onset": md["onset"].to_numpy(float)
                                 if md is not None and "onset" in md else np.nan,
                                 "window mean µV": amps}))
    summary = pd.DataFrame(rows)
    if len(groups) == 2 and min(map(len, vals.values())) >= 2:
        (ga, a), (gb, b) = vals.items()
        w = scipy_stats.ttest_ind(a, b, equal_var=False)
        (pa, pb), (la, lb) = summary["peak µV"], summary["peak latency ms"]
        summary = pd.concat([summary, pd.DataFrame([{
            "group": f"{ga} − {gb}", "n": np.nan, "window mean µV": float(a.mean() - b.mean()),
            "sem": np.nan, "peak µV": pa - pb, "peak latency ms": la - lb,
            "Welch t": float(w.statistic), "df": float(w.df), "p": float(w.pvalue),
            "p (Mann-Whitney)": float(scipy_stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
            "hedges g": hedges_g(a, b)}])], ignore_index=True)
    elif len(groups) > 2:
        f = scipy_stats.f_oneway(*[v for v in vals.values() if len(v) >= 2])
        summary = pd.concat([summary, pd.DataFrame([{
            "group": "all groups", "ANOVA F": float(f.statistic), "p": float(f.pvalue)}])],
            ignore_index=True)
    summary.insert(0, "component", label)
    return summary, pd.concat(per, ignore_index=True)


def epoch_subjects(ep) -> np.ndarray | None:
    """Subject label of every epoch ("" when unknown), or None without a subject column."""
    if ep.metadata is None or "subject" not in ep.metadata:
        return None
    return np.array(["" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
                     for v in ep.metadata["subject"]], dtype=object)


def subjects_in(groups: dict) -> list[str]:
    """Subjects with at least one epoch in any group (sorted)."""
    keys = [epoch_subjects(ep) for ep in groups.values()]
    return sorted({s for k in keys if k is not None for s in k} - {""})


def grand_average(evokeds: list) -> mne.Evoked:
    """Average of averages: every subject weighs the same, whatever its epoch count."""
    return evokeds[0].copy() if len(evokeds) == 1 else \
        mne.grand_average(list(evokeds), interpolate_bads=False, drop_bads=False)


def group_component_analysis(groups: dict, comp: Component) -> dict:
    """ERP component across subjects (ERP guide): one Evoked per subject and group (the
    subject's epochs averaged), then their grand average with equal weights; subjects
    are the observations.

    * ``summary``: per group the subjects' window means (mean ± SEM, 95 % CI) and the
      grand average's peak, then the test. Two groups: paired t-test on the subjects
      that have both (= their difference waves against 0), Wilcoxon and Cohen's d_z;
      without shared subjects Welch t on the subject means. More: repeated-measures
      ANOVA and Friedman test on the subjects that have every group.
    * ``subjects``: one row per subject and group; ``subject_tests``: within each subject,
      the epoch-level test between the groups.
    * ``evokeds`` / ``grand`` / ``diffs`` / ``grand_diff``."""
    chans, m, label = component_window(groups, comp)
    keys = {g: epoch_subjects(ep) for g, ep in groups.items()}
    evs, rows = {}, []
    for g, ep in groups.items():
        evs[g] = {}
        for s in (sorted(set(keys[g]) - {""}) if keys[g] is not None else []):
            ev = evs[g][s] = ep[np.flatnonzero(keys[g] == s)].average()
            ev.comment = f"{g} · sub-{s}"
            pa, pl = peak(ev, chans, comp)
            rows.append({"subject": s, "group": g, "epochs": ev.nave,
                         "window mean µV": float((ev.get_data(picks=chans).mean(0) * 1e6)[m].mean()),
                         "peak µV": pa, "peak latency ms": pl})
    subjects = pd.DataFrame(rows)
    vals = {g: {s: w for s, w in zip(subjects.loc[subjects.group == g, "subject"],
                                     subjects.loc[subjects.group == g, "window mean µV"])}
            for g in evs} if len(subjects) else {g: {} for g in evs}
    grand = {g: grand_average(list(by_s.values())) for g, by_s in evs.items() if by_s}
    for g, ev in grand.items():
        ev.comment = f"{g} (grand average)"
    summary = []
    for g, v in vals.items():
        x, n = np.array(list(v.values())), len(v)
        sem = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        pa, pl = peak(grand[g], chans, comp) if g in grand else (np.nan, np.nan)
        summary.append({"group": g, "subjects": n, "epochs": int(sum(e.nave for e in evs[g].values())),
                        "window mean µV": float(x.mean()) if n else np.nan, "sem": sem,
                        "95% CI ±": float(scipy_stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan,
                        "grand-average peak µV": pa, "grand-average peak latency ms": pl})
    labels, diffs, grand_diff = list(vals), {}, None
    if len(labels) == 2:
        ga, gb = labels
        both = sorted(set(vals[ga]) & set(vals[gb]))
        row = {"group": f"{ga} − {gb}"}
        if len(both) >= 2:
            a, b = np.array([vals[ga][s] for s in both]), np.array([vals[gb][s] for s in both])
            d, t = a - b, scipy_stats.ttest_rel(a, b)
            sd = d.std(ddof=1)
            try:
                wp = float(scipy_stats.wilcoxon(a, b).pvalue)
            except ValueError:
                wp = np.nan
            row.update({"subjects": len(both), "window mean µV": float(d.mean()),
                        "sem": float(sd / np.sqrt(len(d))),
                        "95% CI ±": float(scipy_stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))),
                        "test": "paired t (subjects)", "t": float(t.statistic),
                        "df": float(len(d) - 1), "p": float(t.pvalue), "p (Wilcoxon)": wp,
                        "cohen dz": float(d.mean() / sd) if sd > 0 else np.nan})
        else:
            a, b = np.array(list(vals[ga].values())), np.array(list(vals[gb].values()))
            if min(len(a), len(b)) >= 2:
                t = scipy_stats.ttest_ind(a, b, equal_var=False)
                row.update({"subjects": len(a) + len(b), "window mean µV": float(a.mean() - b.mean()),
                            "test": "Welch t (subject means, independent groups)",
                            "t": float(t.statistic), "df": float(t.df), "p": float(t.pvalue),
                            "p (Mann-Whitney)": float(scipy_stats.mannwhitneyu(a, b).pvalue),
                            "hedges g": hedges_g(a, b)})
        summary.append(row)
        for s in both:
            diffs[s] = mne.combine_evoked([evs[ga][s], evs[gb][s]], weights=[1, -1])
            diffs[s].comment = f"{ga} − {gb} · sub-{s}"
        if diffs:
            grand_diff = grand_average(list(diffs.values()))
            grand_diff.comment = f"{ga} − {gb}"
    elif len(labels) > 2:
        common = sorted(set.intersection(*[set(v) for v in vals.values()]))
        if len(common) >= 2:
            from mne.stats import f_mway_rm
            y = np.array([[vals[g][s] for g in labels] for s in common])
            fv, pv = f_mway_rm(y[:, :, None], [len(labels)], effects="A", correction=False)
            summary.append({"group": "all groups", "subjects": len(common),
                            "test": "repeated-measures ANOVA", "F": float(np.ravel(fv)[0]),
                            "df": float(len(labels) - 1),
                            "df2": float((len(labels) - 1) * (len(common) - 1)), "p": float(np.ravel(pv)[0]),
                            "p (Friedman)": float(scipy_stats.friedmanchisquare(*y.T).pvalue)})
    tests = []
    for s in subjects_in(groups):
        sub = {g: ep[np.flatnonzero(keys[g] == s)] for g, ep in groups.items()
               if keys[g] is not None and (keys[g] == s).sum() >= 2}
        if len(sub) >= 2:
            r = compare_groups({g: ep.get_data(picks=chans).mean(1)[:, m].mean(1) * 1e6
                                for g, ep in sub.items()})
            tests.append({"subject": s, **{f"n {g}": len(ep) for g, ep in sub.items()},
                          "test (epochs)": r["test"], "statistic": r["stat"], "p": r["p"],
                          "effect (hedges g)": r["effect"], "p (non-parametric)": r["p_np"]})
    subject_tests = pd.DataFrame(tests)
    if len(subject_tests):
        subject_tests["q (FDR)"] = fdr(subject_tests["p"].to_numpy())
    summary = pd.DataFrame(summary)
    summary.insert(0, "component", label)
    return dict(summary=summary, subjects=subjects, subject_tests=subject_tests,
                evokeds=evs, grand=grand, diffs=diffs, grand_diff=grand_diff, channels=chans)


def group_cluster_test(diffs: list, tfce=False, n_permutations=5000, seed=42, alpha=0.05,
                       tmin=None, tmax=None) -> dict:
    """Across-subject test of the difference waves against 0 (ERP guide):
    ``spatio_temporal_cluster_1samp_test`` with the channels' adjacency, clusters formed
    from the t distribution or with TFCE (``threshold=dict(start=0, step=0.2)``).
    Returns t values, the significance mask (times × channels), one row per significant
    cluster (time span, channels, p) and the settings."""
    from mne.stats import spatio_temporal_cluster_1samp_test
    if len(diffs) < 3:
        raise ValueError("The cluster test needs at least 3 subjects with both groups")
    ev0 = diffs[0].copy().pick("eeg")
    if tmin is not None or tmax is not None:
        ev0.crop(tmin, tmax)
    chans, times = ev0.ch_names, ev0.times
    x = np.array([d.copy().pick(chans).crop(times[0], times[-1]).data.T for d in diffs])
    t_obs, clusters, pv, _ = spatio_temporal_cluster_1samp_test(
        x, adjacency=mne.channels.find_ch_adjacency(ev0.info, ch_type="eeg")[0],
        n_permutations=int(n_permutations), out_type="mask", seed=seed, n_jobs=1,
        **(dict(threshold=dict(start=0, step=0.2)) if tfce else {}))

    def row(name, m, p, **extra):
        ti, ci = np.nonzero(m)
        return {"cluster": name, "from ms": times[ti.min()] * 1000, "to ms": times[ti.max()] * 1000,
                **extra, "channels": ", ".join(sorted({chans[c] for c in ci})), "p": p}
    if tfce:
        mask = pv.reshape(t_obs.shape) < alpha
        rows = [row("TFCE", mask, float(pv.min()))] if mask.any() else []
    else:
        sig = np.flatnonzero(pv < alpha)
        mask = np.any([clusters[k] for k in sig], axis=0) if len(sig) else np.zeros(t_obs.shape, bool)
        rows = [row(int(k), clusters[k], float(pv[k]),
                    sign="+" if t_obs[clusters[k]].mean() > 0 else "−") for k in sig]
    return dict(t_obs=t_obs, mask=mask, table=pd.DataFrame(rows), times=times, channels=chans,
                n_subjects=len(diffs), tfce=tfce, n_permutations=int(n_permutations))


#: Reference chosen with each ERP component: the average reference for posterior
#: visual P1 and the N170, averaged mastoids for the others (the ERP CORE convention).
COMPONENT_REFERENCE = {
    "P1": "average", "N1": "mastoids", "N170": "average", "P2": "mastoids",
    "N2": "mastoids", "MMN": "mastoids", "P300": "mastoids", "N400": "mastoids",
    "LPP": "mastoids",
}
#: Mastoid-like channel pairs, in order of preference (10-05 names)
MASTOID_PAIRS = (("M1", "M2"), ("TP9", "TP10"), ("A1", "A2"), ("P9", "P10"))


def resolve_erp_reference(spec: str, ch_names: list[str]):
    """A reference choice ("recorded", "average", "mastoids" or channel names "Cz" /
    "M1, M2") as what ``set_eeg_reference`` needs, plus a label. The channels are the
    epochs' good ones; without a mastoid pair the data stay as recorded."""
    low, lower = (spec or "").strip().lower(), {c.lower(): c for c in ch_names}
    if low in ("recorded", "keep", "as recorded", ""):
        return None, "as recorded"
    if low in ("average", "avg", "car"):
        return "average", "average reference"
    if low == "mastoids":
        pair = next(([lower[a.lower()], lower[b.lower()]] for a, b in MASTOID_PAIRS
                     if a.lower() in lower and b.lower() in lower), None)
        return (pair, f"mean of {'+'.join(pair)}") if pair else (
            None, "as recorded — no good mastoid pair (M1/M2, TP9/TP10, A1/A2, P9/P10) "
                  "among the channels")
    names = [n for n in spec.replace(";", ",").replace(",", " ").split() if n]
    missing = [n for n in names if n.lower() not in lower]
    if missing:
        raise ValueError(f"Reference channel(s) not among the epochs' channels (bad, or "
                         f"another name?): {', '.join(missing)}")
    chans = [lower[n.lower()] for n in names]
    return chans, f"mean of {'+'.join(chans)}" if len(chans) > 1 else chans[0]


def reference_epochs(epochs, ref):
    """A re-referenced copy (``ref`` from :func:`resolve_erp_reference`)."""
    return epochs if ref is None else epochs.copy().set_eeg_reference(ref, projection=False)


# =========================================================================== statistics
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / sp * j)


def describe_groups(values: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for g, v in values.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        n = len(v)
        out[g] = {"n": n, "mean": float(v.mean()) if n else np.nan,
                  "sd": float(v.std(ddof=1)) if n > 1 else np.nan,
                  "sem": float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan}
    return out


def compare_groups(values: dict[str, np.ndarray]) -> dict:
    """Two groups: Welch t-test, Mann-Whitney U and Hedges' g (first − second).
    More groups: one-way ANOVA and Kruskal-Wallis."""
    vals = {g: np.asarray(v, float)[np.isfinite(np.asarray(v, float))]
            for g, v in values.items()}
    vals = {g: v for g, v in vals.items() if len(v) >= 2}
    if len(vals) < 2:
        return {"test": "", "stat": np.nan, "p": np.nan, "effect": np.nan, "p_np": np.nan}
    if len(vals) == 2:
        a, b = vals.values()
        t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"test": "Welch t", "stat": float(t), "p": float(p),
                "effect": hedges_g(a, b), "p_np": float(u.pvalue)}
    f, p = scipy_stats.f_oneway(*vals.values())
    h = scipy_stats.kruskal(*vals.values())
    return {"test": "ANOVA F", "stat": float(f), "p": float(p), "effect": np.nan,
            "p_np": float(h.pvalue)}


def fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (NaNs kept)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from mne.stats import fdr_correction
        q[ok] = fdr_correction(p[ok])[1]
    return q


def comparison_table(per_epoch: pd.DataFrame, measures: list[str], groups: list[str],
                     group_col: str = "group") -> pd.DataFrame:
    """One row per measure: mean ± SEM per group, test, p, FDR q, effect size."""
    rows = []
    for m in measures:
        vals = {g: per_epoch.loc[per_epoch[group_col] == g, m].to_numpy(float)
                for g in groups}
        d = describe_groups(vals)
        c = compare_groups(vals)
        row = {"measure": m}
        for g in groups:
            row[f"{g} mean"] = d[g]["mean"]
            row[f"{g} sem"] = d[g]["sem"]
            row[f"{g} n"] = d[g]["n"]
        row.update({"test": c["test"], "stat": c["stat"], "p": c["p"],
                    "p (non-param.)": c["p_np"], "hedges g": c["effect"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(df.columns.get_loc("p") + 1, "q (FDR)", fdr(df["p"].to_numpy()))
    return df


# =========================================================================== band power
DEFAULT_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                 "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


DEFAULT_ROIS = {
    "frontal midline": ["Fz", "FCz", "F1", "F2", "AFz"],
    "parietal": ["Pz", "P3", "P4", "POz", "CPz"],
    "engagement": ["Cz", "Pz", "P3", "P4"],
    "left frontal": ["F3"],
    "right frontal": ["F4"],
}


NORMALISATIONS = ("absolute", "dB", "% change", "ratio", "z")


@dataclass
class BandSettings:
    bands: dict = field(default_factory=lambda: dict(DEFAULT_BANDS))
    tmin: float | None = 0.0       # analysis window within the epoch
    tmax: float | None = None
    method: str = "welch"          # welch | multitaper


@dataclass
class BandBaseline:
    """Band-power baseline: ``t0``…``t1`` s after each baseline event onset.

    ``subjects`` (one per onset) makes the normalisation per subject: an epoch is
    only ever compared with baseline segments of its own subject."""
    onsets: np.ndarray             # baseline event onsets (s, recording time)
    t0: float = 0.0
    t1: float = 10.0
    mode: str = "mean"             # mean | nearest (most recent preceding event)
    label: str = ""
    subjects: np.ndarray | None = None


def _integrate(psd: np.ndarray, freqs: np.ndarray, bands: dict) -> np.ndarray:
    """(…, n_freqs) V²/Hz → (…, n_bands) µV²."""
    df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
    out = []
    for lo, hi in bands.values():
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            m = np.abs(freqs - (lo + hi) / 2) == np.abs(freqs - (lo + hi) / 2).min()
        out.append(psd[..., m].sum(-1) * df)
    return np.stack(out, -1) * 1e12


def band_power(epochs: mne.BaseEpochs, settings: BandSettings) -> np.ndarray:
    """Absolute band power per epoch, channel and band: (n_epochs, n_ch, n_bands) µV²."""
    bands = settings.bands
    fmin = min(lo for lo, _ in bands.values())
    fmax = min(max(hi for _, hi in bands.values()), epochs.info["sfreq"] / 2)
    tmin = settings.tmin if settings.tmin is not None else epochs.tmin
    tmax = settings.tmax if settings.tmax is not None else epochs.tmax
    tmin, tmax = max(tmin, epochs.tmin), min(tmax, epochs.tmax)
    if tmax - tmin <= 0:
        raise ValueError("The band-power window lies outside the epochs")
    n = int(round((tmax - tmin) * epochs.info["sfreq"])) + 1
    kw = {}
    if settings.method == "welch":
        seg = min(n, int(2 * epochs.info["sfreq"]))  # ≤ 2 s segments, 50 % overlap
        kw = dict(n_fft=seg, n_per_seg=seg, n_overlap=seg // 2)
    psd = epochs.compute_psd(method=settings.method, fmin=fmin, fmax=fmax, tmin=tmin,
                             tmax=tmax, picks="eeg", exclude=(), **kw)
    data, freqs = psd.get_data(return_freqs=True)
    bp = _integrate(data, freqs, bands)
    bp[bp <= 0] = np.nan  # flat (reference) channels
    return bp


def baseline_epochs(raw: mne.io.BaseRaw, base: BandBaseline, seg_len: float,
                    ch_names: list[str]) -> tuple[mne.Epochs, np.ndarray]:
    """Cut ``t0``…``t1`` after each baseline onset into ``seg_len`` s segments.

    Returns the segments (as Epochs, tmin=0) and, per segment, the index of the
    baseline event it came from.
    """
    seg_len = max(seg_len, 0.25)
    sf = raw.info["sfreq"]
    starts, owner = [], []
    for k, on in enumerate(np.asarray(base.onsets, float)):
        t = on + base.t0
        while t + seg_len <= on + base.t1 + 1e-9:
            starts.append(t)
            owner.append(k)
            t += seg_len
    if not starts:
        raise ValueError("The baseline window is shorter than one analysis window "
                         f"({seg_len:.2f} s)")
    samples = np.round(np.asarray(starts) * sf).astype(int) + raw.first_samp
    ok = (samples >= raw.first_samp) & (samples + int(seg_len * sf) <= raw.last_samp)
    if not ok.any():
        raise ValueError("The baseline window falls outside the recording")
    samples, owner = samples[ok], np.asarray(owner)[ok]
    events = np.column_stack([samples, np.zeros_like(samples), np.ones_like(samples)])
    picks = [c for c in ch_names if c in raw.ch_names]
    ep = mne.Epochs(raw, events, {"baseline": 1}, tmin=0.0, tmax=seg_len - 1 / sf,
                    baseline=None, picks=picks, preload=True, reject_by_annotation=True)
    owner = owner[[i for i, d in enumerate(ep.drop_log) if not d]]
    if not len(ep):
        raise ValueError("All baseline segments overlap BAD annotations")
    return ep, owner


def baseline_band_power(raw: mne.io.BaseRaw, base: BandBaseline, settings: BandSettings,
                        ch_names: list[str], seg_len: float, ref=None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Band power of baseline segments: (n_segments, n_ch, n_bands) and segment owners.

    ``ref`` (see :func:`resolve_erp_reference`) re-references the
    segments the same way as the epochs they are compared with."""
    ep, owner = baseline_epochs(raw, base, seg_len, ch_names)
    ep = reference_epochs(ep, ref)
    s = BandSettings(settings.bands, None, None, settings.method)
    return band_power(ep, s), owner


def _subject_keys(values, n: int) -> np.ndarray:
    """Subject labels as text ("" = unknown); one subject when there are none."""
    if values is None:
        return np.full(n, "", dtype=object)
    v = pd.Series(values).fillna("").astype(str).to_numpy(dtype=object)
    return v if len(v) == n else np.full(n, "", dtype=object)


def reference_for(onsets: np.ndarray, base: BandBaseline, base_bp: np.ndarray,
                  owner: np.ndarray, subjects=None) -> tuple[np.ndarray, np.ndarray]:
    """Baseline mean and SD per epoch: (n_epochs, n_ch, n_bands) each.

    Normalisation is **per subject**: an epoch of subject *s* only uses baseline
    segments of baseline events of *s* (``base.subjects`` and the epochs'
    ``subjects``); without subject labels everything is one subject.
    ``mode="mean"`` uses all of the subject's baseline segments; ``"nearest"`` the
    segments of the subject's most recent baseline event before the epoch (or its
    first one). Epochs whose subject has no baseline get NaN.
    """
    onsets = np.asarray(onsets, float)
    n = len(onsets)
    b_on = np.asarray(base.onsets, float)
    b_sub = _subject_keys(base.subjects, len(b_on))
    e_sub = _subject_keys(subjects, n)
    if not (b_sub != "").any() or not (e_sub != "").any():  # no subject information
        b_sub, e_sub = np.full(len(b_on), "", object), np.full(n, "", object)
    seg_sub = b_sub[owner]
    shape = base_bp.shape[1:]
    mean = np.full((n, *shape), np.nan)
    sd = np.full((n, *shape), np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for subj in np.unique(e_sub):
                ei = np.flatnonzero(e_sub == subj)
                si = seg_sub == subj
                if not si.any():
                    continue  # no baseline for this subject: leave NaN
                if base.mode == "nearest":
                    ks = np.flatnonzero(b_sub == subj)          # this subject's events
                    pos = np.searchsorted(b_on[ks], onsets[ei], side="right") - 1
                    pick = ks[np.clip(pos, 0, len(ks) - 1)]
                    for e, k in zip(ei, pick):
                        own = owner == k
                        if not own.any():
                            own = si
                        mean[e] = np.nanmean(base_bp[own], 0)
                        sd[e] = np.nanstd(base_bp[own if own.sum() > 1 else si], 0)
                else:
                    mean[ei] = np.nanmean(base_bp[si], 0)
                    sd[ei] = np.nanstd(base_bp[si], 0)
    return mean, sd


def subjects_without_baseline(base: BandBaseline, subjects) -> list[str]:
    """Subjects of the epochs that have no baseline event."""
    b = set(_subject_keys(base.subjects, len(base.onsets)))
    e = set(_subject_keys(subjects, len(subjects) if subjects is not None else 0))
    if b == {""} or e <= {""}:
        return []
    return sorted(e - b)


def normalise(bp: np.ndarray, ref: np.ndarray | None, sd: np.ndarray | None,
              mode: str) -> np.ndarray:
    if mode == "absolute" or ref is None:
        return bp
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "dB":
            return 10 * np.log10(bp / ref)
        if mode == "% change":
            return 100 * (bp - ref) / ref
        if mode == "ratio":
            return bp / ref
        if mode == "z":
            return (bp - ref) / sd
    raise ValueError(f"Unknown normalisation {mode!r}")


def unit(mode: str) -> str:
    return {"absolute": "µV²", "dB": "dB", "% change": "%", "ratio": "×", "z": "z"}[mode]


def resolve_roi(names, ch_names: list[str]) -> list[int]:
    lower = {c.lower(): i for i, c in enumerate(ch_names)}
    return [lower[n.lower()] for n in names if n.lower() in lower]


def roi_mean(bp: np.ndarray, idx: list[int]) -> np.ndarray:
    """Mean over ROI channels: (n_epochs, n_ch, n_bands) → (n_epochs, n_bands)."""
    if not idx:
        return np.full((bp.shape[0], bp.shape[2]), np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(bp[:, idx, :], axis=1)


INDEX_INFO = {
    "engagement β/(α+θ)": "Pope et al. 1995 — higher = more engaged (Cz, Pz, P3, P4)",
    "workload FMθ/Pα": "Gevins & Smith 2003 — frontal-midline theta / parietal alpha; "
                       "rises with working-memory load",
    "θ/α": "theta / alpha over all channels; rises with mental workload",
    "θ/β frontal": "frontal theta / beta; higher = lower attentional control",
    "FAA ln α(F4)−ln α(F3)": "frontal alpha asymmetry; positive = relatively more left "
                             "frontal activity (approach motivation)",
    "α ERD % (parietal)": "parietal alpha change vs baseline; negative = alpha "
                          "suppression (attention / load). Needs a baseline",
    "FMθ ERS % (frontal)": "frontal-midline theta change vs baseline; positive = theta increase "
                  "(working memory load). Needs a baseline",
}


def eeg_indices(bp: np.ndarray, ch_names: list[str], bands: dict,
                rois: dict | None = None, ref: np.ndarray | None = None
                ) -> tuple[pd.DataFrame, list[str]]:
    """Per-epoch cognitive-state indices from absolute band power.

    ``ref`` (same shape as ``bp``) is the baseline band power used for ERD/ERS.
    Returns the table and notes about missing bands/channels.
    """
    rois = {**DEFAULT_ROIS, **(rois or {})}
    names = [b.lower() for b in bands]
    notes = []

    def band(b):
        return names.index(b) if b in names else None

    th, al, be = band("theta"), band("alpha"), band("beta")
    roi_idx = {k: resolve_roi(v, ch_names) for k, v in rois.items()}
    for k, v in roi_idx.items():
        if not v:
            notes.append(f"no channel of ROI “{k}” ({', '.join(rois[k])})")
    allidx = list(range(len(ch_names)))
    r = {k: roi_mean(bp, v) for k, v in roi_idx.items()}
    r["all"] = roi_mean(bp, allidx)
    n = bp.shape[0]
    nan = np.full(n, np.nan)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if None in (th, al, be):
            notes.append("indices need bands named theta, alpha and beta")
        else:
            e = r["engagement"]
            out["engagement β/(α+θ)"] = e[:, be] / (e[:, al] + e[:, th])
            out["workload FMθ/Pα"] = r["frontal midline"][:, th] / r["parietal"][:, al]
            out["θ/α"] = r["all"][:, th] / r["all"][:, al]
            fr = roi_mean(bp, roi_idx["frontal midline"] + roi_idx["left frontal"] +
                          roi_idx["right frontal"])
            out["θ/β frontal"] = fr[:, th] / fr[:, be]
            out["FAA ln α(F4)−ln α(F3)"] = (np.log(r["right frontal"][:, al])
                                             - np.log(r["left frontal"][:, al]))
            if ref is not None:
                rp = roi_mean(ref, roi_idx["parietal"])
                rf = roi_mean(ref, roi_idx["frontal midline"])
                out["α ERD % (parietal)"] = 100 * (r["parietal"][:, al] - rp[:, al]) / rp[:, al]
                out["FMθ ERS % (frontal)"] = 100 * (r["frontal midline"][:, th] - rf[:, th]) / rf[:, th]
            else:
                notes.append("ERD/ERS indices need a band-power baseline")
    df = pd.DataFrame({k: (v if v is not None else nan) for k, v in out.items()})
    return df, notes


def morlet_band(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """(n, n_ch, n_times) V → Morlet power averaged over the band's frequencies and the
    channels: (n, n_times) µV²."""
    lo, hi = band
    freqs = np.arange(lo, hi + 1e-9, max((hi - lo) / 6, 0.5))
    if len(freqs) < 2:
        freqs = np.array([lo, hi])
    n_cycles = np.maximum(freqs / 2.0, 3.0)
    p = mne.time_frequency.tfr_array_morlet(data, sfreq, freqs, n_cycles=n_cycles,
                                            output="power")
    with np.errstate(invalid="ignore"):
        return np.nanmean(p, axis=(1, 2)) * 1e12


def band_timecourse(epochs: mne.BaseEpochs, band: tuple[float, float], picks: list[str]
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Band power over epoch time on a channel set: times, (n_epochs, n_times) µV²."""
    picks = [c for c in picks if c in epochs.ch_names]
    if not picks:
        raise ValueError("None of the time-course channels are in the epochs")
    x = epochs.get_data(picks=picks)
    return epochs.times, morlet_band(x, epochs.info["sfreq"], band)


def baseline_timecourse_levels(raw: mne.io.BaseRaw, base: BandBaseline,
                               band: tuple[float, float], picks: list[str],
                               seg_len: float = 4.0, ref=None,
                               ch_names: list[str] | None = None
                               ) -> dict[str, tuple[float, float]]:
    """Mean and SD of the same Morlet band power in the baseline window (edges trimmed),
    **per subject** (key "" when there are no subject labels). With ``ref``, segments
    over ``ch_names`` are re-referenced like the epochs first."""
    picks = [c for c in picks if c in raw.ch_names and c not in raw.info["bads"]]
    chans = [c for c in (ch_names or picks) if c in raw.ch_names]
    ep, owner = baseline_epochs(raw, base, max(seg_len, 6.0 / band[0]), chans)
    ep = reference_epochs(ep, ref)
    p = morlet_band(ep.get_data(picks=picks), ep.info["sfreq"], band)
    trim = int(np.ceil(1.5 / band[0] * ep.info["sfreq"]))
    core = p[:, trim:-trim] if p.shape[1] > 2 * trim + 2 else p
    seg_sub = _subject_keys(base.subjects, len(base.onsets))[owner]
    return {str(s): (float(np.nanmean(core[seg_sub == s])),
                     float(np.nanstd(core[seg_sub == s].mean(1))))
            for s in np.unique(seg_sub)}


def gaze_metrics(onsets: np.ndarray, tmin: float, tmax: float, gaze=None, gaze_events=None,
                 pupil=None, pupil_baseline=(-0.2, 0.0)) -> pd.DataFrame:
    """Per-epoch eye metrics in [onset+tmin, onset+tmax]:

    * pupil dilation (task-evoked, vs ``pupil_baseline`` before the event) — load
    * blink rate (/min) and mean fixation duration (ms) inside the window
    """
    n = len(onsets)
    out = {"pupil dilation": np.full(n, np.nan), "blink rate /min": np.full(n, np.nan),
           "mean fixation ms": np.full(n, np.nan)}
    if gaze is not None:
        p = pupil if pupil is not None else gaze.pupil
        if p is not None and np.isfinite(p).any():
            t = gaze.times
            for i, on in enumerate(onsets):
                b = (t >= on + pupil_baseline[0]) & (t < on + pupil_baseline[1])
                w = (t >= on + max(tmin, 0)) & (t <= on + tmax)
                if b.any() and w.any():
                    with np.errstate(invalid="ignore"):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            out["pupil dilation"][i] = np.nanmean(p[w]) - np.nanmean(p[b])
    if gaze_events is not None:
        dur_min = (tmax - tmin) / 60.0
        bl = gaze_events.blinks["onset"].to_numpy(float) if len(gaze_events.blinks) else \
            np.array([])
        fx = gaze_events.fixations
        fo = fx["onset"].to_numpy(float) if len(fx) else np.array([])
        fd = fx["duration"].to_numpy(float) if len(fx) else np.array([])
        for i, on in enumerate(onsets):
            a, b = on + tmin, on + tmax
            out["blink rate /min"][i] = ((bl >= a) & (bl < b)).sum() / dur_min
            m = (fo >= a) & (fo < b)
            if m.any():
                out["mean fixation ms"][i] = fd[m].mean() * 1000
    return pd.DataFrame(out)


# =========================================================================== analyses
@dataclass
class GroupSpec:
    """An event group: the event names ticked, the filters and the recordings kept
    (``recordings``: column → values kept, e.g. ``{"subject": ["01", "02"]}``)."""
    label: str
    names: list[str]
    text: str = ""
    query: str = ""
    conditions: list = field(default_factory=list)
    recordings: dict = field(default_factory=dict)


def select_events(table: pd.DataFrame, spec: GroupSpec) -> pd.DataFrame:
    """Rows of the event table in a group (raises ValueError for invalid filters)."""
    df = filter_events(table, spec.text, spec.query, set(spec.names),
                          [tuple(c) for c in spec.conditions])
    for col, keep in spec.recordings.items():
        if col in df:
            df = df[df[col].fillna("").astype(str).isin(keep)]
    return df


@dataclass
class ErpSpec:
    groups: list[str]
    component: Component
    baseline: tuple[float, float] | None = (-0.1, 0.0)
    reference: str = "recorded"   # average | mastoids | recorded | channel names


def run_erp(epochs: dict[str, mne.BaseEpochs], spec: ErpSpec) -> dict:
    """ERP component analysis. With several subjects: one Evoked per subject and group,
    grand average of those, subjects as observations (:func:`eeg.group_component_analysis`);
    otherwise trials are the observations (:func:`eeg.component_analysis`)."""
    groups = {g: epochs[g] for g in spec.groups}
    first = next(iter(groups.values()))
    ref, ref_label = resolve_erp_reference(spec.reference, list(first.ch_names))
    comp, bl = spec.component, spec.baseline
    notes = []
    eps = {g: reference_epochs(ep, ref) for g, ep in groups.items()}
    eps = {g: (ep.copy().apply_baseline(bl) if bl else ep)
           for g, ep in eps.items()}
    evokeds = {}
    for g, ep in eps.items():
        ev = ep.average()
        ev.comment = g
        evokeds[g] = ev
    summary, per = component_analysis(eps, comp)
    subj = {g: epoch_subjects(ep) for g, ep in eps.items()}
    if any(k is not None for k in subj.values()):
        per.insert(1, "subject", np.concatenate(
            [k if k is not None else np.full(len(ep), "", object)
             for k, ep in zip(subj.values(), eps.values())]))
    group = None
    if len(subjects_in(eps)) >= 2:
        group = group_component_analysis(eps, comp)
        unknown = sum(int((k == "").sum()) for k in subj.values() if k is not None)
        if unknown:
            notes.append(f"⚠ {unknown} epochs without a subject are left out of the "
                         "subject averages")
        return dict(eps=eps, evokeds=group["grand"], summary=group["summary"],
                    per_epoch=per, diff=group["grand_diff"], roi=list(comp.channels),
                    comp=comp, baseline=bl, notes=notes, reference=ref_label, group=group,
                    spec=spec)
    diff = None
    if len(eps) == 2:
        (la, a), (lb, b) = eps.items()
        diff = mne.combine_evoked([evokeds[la], evokeds[lb]], weights=[1, -1])
        diff.comment = f"{la} − {lb}"
        ov = len(set(a.events[:, 0]) & set(b.events[:, 0]))
        if ov:
            notes.append(f"⚠ {ov} events are in both groups (tests assume independence)")
    return dict(eps=eps, evokeds=evokeds, summary=summary, per_epoch=per, diff=diff,
                roi=list(comp.channels), comp=comp, baseline=bl, notes=notes,
                reference=ref_label, group=None, spec=spec)


@dataclass
class BaselineSpec:
    """Baseline after a baseline event: ``t0``…``t1`` s after each ``event`` matching
    ``query``; ``mode`` "all" or "nearest" (see :class:`bandpower.BandBaseline`)."""
    event: str
    t0: float
    t1: float
    mode: str = "all"
    query: str = ""


def band_baseline(table: pd.DataFrame, spec: BaselineSpec | None) -> BandBaseline | None:
    if spec is None:
        return None
    sel = table[table["name"].astype(str) == spec.event]
    if spec.query.strip():
        sel = filter_events(sel, query=spec.query)
    if not len(sel):
        raise ValueError(f"No “{spec.event}” events match the baseline filter")
    if spec.t1 <= spec.t0:
        raise ValueError("The baseline end must be after its start")
    sel = sel.sort_values("onset")
    subjects = sel["subject"].to_numpy() if "subject" in sel else None
    return BandBaseline(sel["onset"].to_numpy(float), spec.t0, spec.t1, spec.mode,
                           spec.event, subjects)


@dataclass
class BandPowerSpec:
    groups: list[str]
    bands: dict
    tmin: float
    tmax: float
    method: str = "welch"
    roi: list[str] = field(default_factory=list)   # empty: every channel
    reference: str = "average"
    baseline: BaselineSpec | None = None
    norm: str = "dB"                                # used only with a baseline


def run_bandpower(epochs: dict[str, mne.BaseEpochs], raw: mne.io.BaseRaw,
                  table: pd.DataFrame, spec: BandPowerSpec) -> dict:
    """Band power per epoch (``Epochs.compute_psd``), normalised per subject against the
    baseline after a baseline event, on the ROI mean; group comparisons per band."""
    groups = {g: epochs[g] for g in spec.groups}
    first = next(iter(groups.values()))
    bands = dict(spec.bands)
    base = band_baseline(table, spec.baseline)
    roi = list(spec.roi) or list(first.ch_names)
    s = BandSettings(bands, spec.tmin, spec.tmax, spec.method)
    eeg_ref, ref_label = resolve_erp_reference(spec.reference, list(first.ch_names))
    mode = spec.norm if base is not None else "absolute"
    names = list(bands)
    notes = []
    out = {"bp": {}, "ref": {}, "sd": {}, "norm": {}, "per": [], "bands": bands,
           "mode": mode, "roi": roi, "base": base, "settings": s,
           "reference": eeg_ref, "reference_label": ref_label, "spec": spec}
    eps = {g: reference_epochs(ep, eeg_ref) for g, ep in groups.items()}
    ch = list(first.ch_names)
    seg = (min(s.tmax, first.tmax) - max(s.tmin, first.tmin))
    base_bp = owner = None
    if base is not None:
        base_bp, owner = baseline_band_power(raw, base, s, ch, seg, ref=eeg_ref)
        notes.append(f"baseline: {len(base_bp)} segments of {seg:.2f} s from "
                     f"{len(np.unique(owner))} “{base.label}” event(s), "
                     f"{base.t0:g}…{base.t1:g} s after")
        subj_all = sorted({str(x) for ep in eps.values() if ep.metadata is not None
                           and "subject" in ep.metadata
                           for x in ep.metadata["subject"].fillna("")} - {""})
        if subj_all:
            notes.append(f"normalised per subject ({len(subj_all)}: "
                         f"{', '.join(subj_all[:8])}{' …' if len(subj_all) > 8 else ''})")
            missing = subjects_without_baseline(base, subj_all)
            if missing:
                notes.append("⚠ no baseline event for subject(s) "
                             f"{', '.join(missing)}: their epochs are left out "
                             "of the normalised values")
    ridx = resolve_roi(roi, ch)
    for g, ep in eps.items():
        bp = band_power(ep, s)
        out["bp"][g] = bp
        if base_bp is not None:
            subj = (ep.metadata["subject"].to_numpy()
                    if "subject" in ep.metadata else None)
            ref, sd = reference_for(ep.metadata["onset"].to_numpy(float), base,
                                       base_bp, owner, subj)
        else:
            ref = sd = None
        out["ref"][g], out["sd"][g] = ref, sd
        if mode == "dB" and ref is not None:
            # dB of ROI-mean power (averaging dB across channels is biased)
            val = 10 * np.log10(roi_mean(bp, ridx) / roi_mean(ref, ridx))
        else:
            val = roi_mean(normalise(bp, ref, sd, mode), ridx)
        out["norm"][g] = normalise(bp, ref, sd, mode)
        df = pd.DataFrame(val, columns=names)
        df.insert(0, "group", g)
        df.insert(1, "onset", ep.metadata["onset"].to_numpy(float))
        out["per"].append(df)
    per = pd.concat(out["per"], ignore_index=True)
    out["per_epoch"] = per
    out["table"] = comparison_table(per, names, list(eps))
    out["notes"] = notes
    return out


def sem(x: np.ndarray, axis=0) -> np.ndarray:
    """``scipy.stats.sem`` ignoring NaNs; NaN where fewer than two values (no warnings)."""
    import warnings

    from scipy import stats
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return np.asarray(stats.sem(np.asarray(x, float), axis=axis, nan_policy="omit"),
                          float)


def run_timecourse(bp: dict, epochs: dict[str, mne.BaseEpochs], raw: mne.io.BaseRaw,
                   band: str) -> dict:
    """Morlet power time course of one band (ERD/ERS) on the band-power result's ROI,
    reference and baseline (per subject)."""
    rng = bp["bands"][band]
    groups = {g: epochs[g] for g in bp["bp"] if g in epochs}
    base, mode, roi = bp["base"], bp["mode"], bp["roi"]
    out = {"band": band, "mode": mode, "curves": {}}
    levels = None
    if base is not None and mode != "absolute":
        levels = baseline_timecourse_levels(   # per subject
            raw, base, rng, roi, ref=bp["reference"],
            ch_names=list(next(iter(groups.values())).ch_names))
    for g, ep in groups.items():
        t, pw = band_timecourse(reference_epochs(ep, bp["reference"]), rng, roi)
        if levels is not None:
            subj = _subject_keys(ep.metadata["subject"].to_numpy()
                                    if ep.metadata is not None
                                    and "subject" in ep.metadata else None, len(pw))
            if set(levels) == {""}:
                subj[:] = ""
            lvl = np.array([levels.get(x, (np.nan, np.nan))[0] for x in subj])[:, None]
            sd = np.array([levels.get(x, (np.nan, np.nan))[1] for x in subj])[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                pw = {"dB": 10 * np.log10(pw / lvl), "% change": 100 * (pw - lvl) / lvl,
                      "ratio": pw / lvl, "z": (pw - lvl) / sd}[mode]
        out["curves"][g] = (t, np.nanmean(pw, 0), sem(pw))
    return out


@dataclass
class IndicesSpec:
    rois: dict
    measures: list[str]


def run_indices(bp: dict, epochs: dict[str, mne.BaseEpochs], gaze, gaze_events,
                spec: IndicesSpec) -> dict:
    """Cognitive indices per epoch from the band-power result, plus eye metrics."""
    s = bp["settings"]
    per, notes = [], []
    for g, b in bp["bp"].items():
        ep = epochs[g]
        df, n = eeg_indices(b, list(ep.ch_names), bp["bands"], spec.rois, bp["ref"][g])
        notes += [x for x in n if x not in notes]
        on = ep.metadata["onset"].to_numpy(float)
        gm = gaze_metrics(on, max(s.tmin, ep.tmin), min(s.tmax, ep.tmax), gaze,
                             gaze_events)
        df = pd.concat([df, gm], axis=1)
        df.insert(0, "group", g)
        df.insert(1, "onset", on)
        per.append(df)
    per = pd.concat(per, ignore_index=True)
    cols = [c for c in spec.measures if c in per.columns and per[c].notna().any()]
    missing = [c for c in spec.measures if c not in cols]
    if missing:
        notes.append("not available: " + ", ".join(missing))
    table = comparison_table(per, cols, list(bp["bp"]))
    return dict(per_epoch=per, table=table, measures=cols, notes=notes, spec=spec)
