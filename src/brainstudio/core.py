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

# =========================================================================== loading: XDF
GAZE_TYPE = re.compile(r"gaze|eye|^et$|pupil|tobii|eyelink", re.I)
EEG_TYPE = re.compile(r"eeg|exg", re.I)
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
                          mne.create_info(labels, srate, [types[k] for k in sel]),
                          verbose="error")
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
                          np.asarray(gz["time_series"], float)[idx], labels, screen,
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
        raw.set_montage(name, on_missing="ignore", match_case=False, verbose="error")
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
        raw.set_channel_types(changed, verbose="error")
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
        raw = mne.io.read_raw_eyelink(path, verbose="error")
        return None, annotations_table(raw), gaze_from_raw(raw, screen, drop=False), {}
    raw = mne.io.read_raw(path, preload=True, verbose="error")
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
                                         extension=run.eeg.suffix), verbose="error")
        except Exception:  # noqa: BLE001  broken sidecars: plain MNE + channels.tsv
            raw = mne.io.read_raw(run.eeg, verbose="error")
            ch = channels_tsv(run.eeg)
            to_mne = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "MISC": "misc",
                      "TRIG": "stim", "STIM": "stim", "RESP": "resp", "GSR": "gsr",
                      "TEMP": "temperature", "PPG": "bio", "HEOG": "eog", "VEOG": "eog"}
            types = {r.name: to_mne[r.type.upper()] for r in ch.itertuples()
                     if r.name in raw.ch_names and r.type.upper() in to_mne}
            if types:
                raw.set_channel_types(types, on_unit_change="ignore", verbose="error")
            raw.info["bads"] = [r.name for r in ch.itertuples()
                                if r.status.lower() == "bad" and r.name in raw.ch_names]
            pl = sidecar(run.eeg).get("PowerLineFrequency")
            if isinstance(pl, (int, float)):
                with raw.info._unlock():
                    raw.info["line_freq"] = float(pl)
        raw.load_data(verbose="error")
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
                r.resample(sf, verbose="error")
            r.set_channel_types({c: types[c] for c, k in zip(r.ch_names, r.get_channel_types())
                                 if types[c] != k}, verbose="error")
            starts.append(t)
            t += r.n_times / sf
            ready.append(r)
        raw = mne.concatenate_raws(ready, verbose="error")
        if raws[0].get_montage() is not None:
            raw.set_montage(raws[0].get_montage(), on_missing="ignore", verbose="error")
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


def epochs_from_table(raw: mne.io.BaseRaw, table, tmin: float = -0.2, tmax: float = 0.8,
                      baseline=(None, 0), reject_uv: float | None = None,
                      picks="eeg") -> mne.Epochs:
    """MNE Epochs locked to the rows of an event table, with the table as metadata.

    Event IDs are the table's ``name`` column, so standard MNE selection works:
    ``epochs["fixation"]``, ``epochs['cond == "AI"']`` (metadata query), etc.
    Rows outside the recording, or falling on an already used sample, are
    dropped.
    """
    tab = table.reset_index(drop=True).copy()
    sf = raw.info["sfreq"]
    samples = np.round(tab["onset"].to_numpy(float) * sf).astype(int) + raw.first_samp
    lo = raw.first_samp + int(np.ceil(-tmin * sf))
    hi = raw.last_samp - int(np.ceil(tmax * sf))
    ok = (samples >= lo) & (samples <= hi)
    counts = {"outside recording": int((~ok).sum())}
    tab, samples = tab[ok], samples[ok]
    _, first = np.unique(samples, return_index=True)  # MNE needs unique samples
    keep = np.sort(first)
    counts["same sample"] = len(tab) - len(keep)
    tab, samples = tab.iloc[keep], samples[keep]
    if not len(tab):
        err = ValueError("No selected events fall inside the EEG recording")
        err.counts = counts
        raise err
    names = sorted(tab["name"].astype(str).unique())
    event_id = {n: i + 1 for i, n in enumerate(names)}
    codes = tab["name"].astype(str).map(event_id).to_numpy()
    events = np.column_stack([samples, np.zeros_like(samples), codes]).astype(int)
    picks_idx = mne.pick_types(raw.info, eeg=True, exclude="bads") if picks == "eeg" else picks
    if picks == "eeg" and not len(picks_idx):
        picks_idx = eeg_picks(raw)
    reject = dict(eeg=reject_uv * 1e-6) if reject_uv else None
    meta = tab.reset_index(drop=True)
    meta = meta.astype({c: object for c in meta.columns
                        if isinstance(meta[c].dtype, pd.StringDtype)})
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, picks=picks_idx, reject=reject, preload=True,
                        metadata=meta, verbose="error")
    epochs.selection_counts = counts  # events never epoched, by reason
    return epochs


def drop_reasons(epochs: mne.BaseEpochs) -> dict[str, int]:
    """Dropped epochs by kind: amplitude (peak-to-peak or flat, named by channels) and
    BAD annotation (overlaps a BAD_… span or a file boundary); anything else is other."""
    out = {"amplitude": 0, "BAD annotation": 0, "other": 0}
    chans = set(epochs.ch_names)
    for reasons in epochs.drop_log:
        if not reasons:
            continue
        if any(str(r).upper().startswith(("BAD", "EDGE")) for r in reasons):
            out["BAD annotation"] += 1
        elif any(r in chans for r in reasons):
            out["amplitude"] += 1
        elif reasons != ("IGNORED",):
            out["other"] += 1
    return out


def rejection_causes(epochs: mne.BaseEpochs) -> dict[str, int]:
    """How many epochs each channel caused to be rejected, worst first."""
    counts: dict[str, int] = {}
    for reasons in epochs.drop_log:
        for r in reasons:
            if r in epochs.ch_names:
                counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# =========================================================================== preprocessing
# =========================================================================== bad channels
@dataclass
class BadChannelParams:
    ptp_max_uv: float = 250.0     # a window above this peak-to-peak is "noisy"
    ptp_min_uv: float = 1.0       # a window below this is "flat"
    window: float = 1.0           # s
    max_bad_fraction: float = 0.2  # channel is bad above this fraction of bad windows
    z_thresh: float = 4.0         # robust z of log median ptp across channels
    l_freq: float | None = 0.1    # band-pass before measuring, so drifts and line
    h_freq: float | None = 30.0   # noise don't count as "noisy"
    notch: float | None = 50.0


def _eeg_idx(raw: mne.io.BaseRaw) -> np.ndarray:
    idx = mne.pick_types(raw.info, eeg=True, exclude=[])
    return idx if len(idx) else eeg_picks(raw)


def detect_bad_channels(raw: mne.io.BaseRaw, p: BadChannelParams | None = None,
                        tmin: float | None = None) -> pd.DataFrame:
    """Peak-to-peak statistics per EEG channel and a suggested ``bad`` flag.

    The data are cut into ``p.window`` s windows (after a ``p.l_freq`` high-pass).
    The data are band-passed (and notched) like the preprocessing will do.
    A channel is flagged when more than ``p.max_bad_fraction`` of its windows
    exceed ``p.ptp_max_uv`` or fall below ``p.ptp_min_uv``, or when its median
    peak-to-peak amplitude is a robust-z outlier among channels. Data before
    ``tmin`` (s, e.g. the first event) are ignored.
    """
    p = p or BadChannelParams()
    picks = _eeg_idx(raw)
    sf = raw.info["sfreq"]
    win = max(int(round(p.window * sf)), 2)
    first = _start_sample(raw, tmin)
    n_win = (raw.n_times - first) // win
    if n_win < 1:
        raise ValueError("Recording is shorter than one window")
    rows = []
    for start in range(0, len(picks), 8):  # blocks of channels bound memory use
        block = picks[start:start + 8]
        x = raw.get_data(picks=block, start=first) * 1e6
        h = p.h_freq if p.h_freq and p.h_freq < sf / 2 else None
        lf = p.l_freq if p.l_freq and p.l_freq > 0 else None
        if lf and highpass_method(lf):
            x = mne.filter.filter_data(x, sf, lf, None, verbose="error", **highpass_method(lf))
            lf = None
        if lf or h:
            x = mne.filter.filter_data(x, sf, lf, h, verbose="error")
        if p.notch and p.notch < sf / 2:
            x = mne.filter.notch_filter(x, sf, np.arange(p.notch, sf / 2, p.notch),
                                        verbose="error")
        x = x[:, : n_win * win].reshape(len(block), n_win, win)
        ptp = np.ptp(x, axis=2)
        for k, ch in enumerate(block):
            rows.append({"channel": raw.ch_names[ch],
                         "median_ptp_uv": float(np.median(ptp[k])),
                         "noisy_pct": 100 * float((ptp[k] > p.ptp_max_uv).mean()),
                         "flat_pct": 100 * float((ptp[k] < p.ptp_min_uv).mean())})
    df = pd.DataFrame(rows)
    lg = np.log(np.maximum(df["median_ptp_uv"].to_numpy(), 1e-6))
    mad = np.median(np.abs(lg - np.median(lg))) * 1.4826 or 1.0
    df["z"] = (lg - np.median(lg)) / mad
    reasons = []
    for r in df.itertuples():
        why = []
        if r.flat_pct > 100 * p.max_bad_fraction:
            why.append("flat")
        if r.noisy_pct > 100 * p.max_bad_fraction:
            why.append(f"ptp > {p.ptp_max_uv:g} µV")
        if abs(r.z) > p.z_thresh:
            why.append("outlier")
        reasons.append(", ".join(why))
    df["reason"] = reasons
    df["bad"] = df["reason"] != ""
    return df


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


#: below this high-pass cut-off an FIR filter gets very long: use a zero-phase IIR
IIR_BELOW_HZ = 0.5
IIR_PARAMS = dict(order=4, ftype="butter", output="sos")


def highpass_method(l_freq) -> dict:
    """Filter arguments for a high-pass: FIR, or zero-phase Butterworth below 0.5 Hz."""
    if l_freq and l_freq < IIR_BELOW_HZ:
        return dict(method="iir", iir_params=IIR_PARAMS)
    return {}


def _iir_highpass(raw: mne.io.BaseRaw, l_freq: float, picks) -> None:
    """Zero-phase Butterworth high-pass, in place, on each stretch between
    ``edge``/``bad_acq_skip`` annotations (as ``Raw.filter`` does). Empty stretches are
    skipped: MNE's IIR path divides by their length (a boundary annotation at the very
    start or end of the data would crash it)."""
    from mne.annotations import _annotations_starts_stops
    starts, stops = _annotations_starts_stops(raw, ("edge", "bad_acq_skip"), invert=True)
    if not len(starts):
        starts, stops = np.array([0]), np.array([raw.n_times])
    data = raw._data
    for a, b in zip(starts, stops):
        if b - a < 2:
            continue
        data[:, a:b] = mne.filter.filter_data(
            data[:, a:b], raw.info["sfreq"], l_freq, None, picks=picks, method="iir",
            iir_params=IIR_PARAMS, verbose="error")


def filtered_copy(raw: mne.io.BaseRaw, l_freq, h_freq, notch, picks=None,
                  tmin: float | None = None) -> mne.io.BaseRaw:
    """A filtered copy of ``raw`` (``picks`` only, from ``tmin`` s): MNE's ``Raw.filter``
    and ``Raw.notch_filter`` on the EEG/EOG/ECG/EMG channels. A high-pass below 0.5 Hz
    is a zero-phase 4th-order Butterworth applied with MNE's ``filter_data`` to each
    stretch between boundaries (``Raw.filter``'s IIR path fails on a boundary at the
    first sample, which joined BIDS runs and cropped data have)."""
    out = raw.copy().load_data()
    if picks is not None:
        out.pick(picks)
    a = _start_sample(raw, tmin)
    if a:
        out.crop(tmin=a / raw.info["sfreq"], include_tmax=True)
    fp = mne.pick_types(out.info, eeg=True, eog=True, ecg=True, emg=True, exclude=[])
    nyq = out.info["sfreq"] / 2
    h = h_freq if h_freq and h_freq < nyq else None
    l_freq = l_freq if l_freq and l_freq > 0 else None  # 0 Hz = no high-pass
    if len(fp) and l_freq and highpass_method(l_freq):
        _iir_highpass(out, l_freq, fp)
        if h:
            out.filter(None, h, picks=fp, verbose="error")
    elif len(fp) and (l_freq or h):
        out.filter(l_freq, h, picks=fp, verbose="error")
    if len(fp) and notch and notch < nyq:
        out.notch_filter(np.arange(notch, nyq, notch), picks=fp, verbose="error")
    return out


def _start_sample(raw: mne.io.BaseRaw, tmin: float | None) -> int:
    """Sample index of recording time ``tmin`` (0 when None or outside the data)."""
    if not tmin or tmin <= 0:
        return 0
    i = int(round(tmin * raw.info["sfreq"]))
    return i if i < raw.n_times - int(raw.info["sfreq"]) else 0  # keep ≥ 1 s of data


def ica_channels(raw: mne.io.BaseRaw, bads=(), tmin: float | None = None) -> list[str]:
    """Good, non-flat EEG channels (a reference channel is flat and would break ICA)."""
    names = [raw.ch_names[i] for i in _eeg_idx(raw) if raw.ch_names[i] not in set(bads)]
    a = _start_sample(raw, tmin)
    sd = raw.get_data(picks=names, start=a, stop=min(raw.n_times, a + int(60 * raw.info["sfreq"])))
    return [n for n, s in zip(names, sd.std(1)) if s > 1e-12]


def fit_times(fit_raw: mne.io.BaseRaw) -> np.ndarray:
    """Recording times of the samples ICA was fitted on (it may start at an event)."""
    return fit_raw.times + getattr(fit_raw, "recording_t0", 0.0)


def fit_ica(raw: mne.io.BaseRaw, bads=(), h_freq=None, notch=None,
            p: ICAParams | None = None, tmin: float | None = None
            ) -> tuple[mne.preprocessing.ICA, mne.io.BaseRaw]:
    """Fit ICA on good EEG channels of a ≥ 1 Hz high-passed copy of ``raw``.

    Data before ``tmin`` (s, e.g. the first event) are left out of the fit; the
    ICA is still applied to the whole recording. Returns the ICA and the
    (decimated) data it was fitted on, which :func:`score_blink_components` and
    MNE's ICA plots need (:func:`fit_times` gives its recording times).
    """
    p = p or ICAParams()
    chans = ica_channels(raw, bads, tmin)
    if len(chans) < 3:
        raise ValueError("ICA needs at least 3 good EEG channels")
    h = p.h_freq if h_freq is None else h_freq
    start = _start_sample(raw, tmin) / raw.info["sfreq"]
    fit_raw = filtered_copy(raw, max(p.l_freq, 0.1), h, notch, picks=chans, tmin=tmin)
    # ``chans`` are the channels the user kept: the file's own bad list (e.g. BIDS
    # channels.tsv) must not drop any of them again inside ICA.fit
    fit_raw.info["bads"] = []
    # average reference, as ICLabel expects (it does not change what ICA removes: the
    # common mode is not a component of average-referenced data)
    fit_raw.set_eeg_reference("average", projection=False, verbose="error")
    fit_raw.recording_t0 = start
    decim = max(1, int(fit_raw.info["sfreq"] // p.target_sfreq))
    if decim > 1:
        fit_raw.resample(fit_raw.info["sfreq"] / decim, verbose="error")
    rank = len(chans) - 1  # average reference removes one dimension
    nc = p.n_components if p.n_components < 1 else int(min(p.n_components, rank))
    method = p.method
    if method == "picard":
        try:
            import picard  # noqa: F401
        except ImportError:
            method = "infomax"
    fit_params = (dict(ortho=False, extended=True) if method == "picard" else
                  dict(extended=True) if method == "infomax" else None)
    ica = mne.preprocessing.ICA(n_components=nc, method=method, rng=p.seed,
                                max_iter=p.max_iter, fit_params=fit_params)
    notes = []
    # 1 s segments; bad ones (AutoReject, or a peak-to-peak threshold) are left out
    seg = ica_segments(fit_raw)
    how = "all"
    if p.clean == "autoreject" and len(seg) >= 20 and has_positions(seg):
        try:
            good = ~autoreject_bad_segments(seg, p.seed, p.autoreject_fraction)
            how = "autoreject"
            notes.append(f"AutoReject left out {int((~good).sum())} of {len(seg)} 1 s "
                         "segments")
        except Exception as e:  # noqa: BLE001
            notes.append(f"AutoReject failed ({type(e).__name__}), threshold used")
            good, how = ptp_good_segments(seg, p.reject_uv), "threshold"
    else:
        if p.clean == "autoreject":
            notes.append("AutoReject needs electrode positions and ≥ 20 segments: "
                         "peak-to-peak threshold used")
        good, how = ptp_good_segments(seg, p.reject_uv), "threshold"
    if good.sum() < max(10, len(good) // 10):
        notes.insert(0, f"only {int(good.sum())} of {len(good)} 1 s segments were clean "
                        "(noisy channels kept?), so ICA was fitted on all of them")
        good[:] = True
        how = "all"
    fit_on = seg[np.flatnonzero(good)]
    every = max(p.decim, int(np.ceil(len(fit_on) * len(fit_on.times) / p.max_samples)))
    if every > p.decim:
        notes.append(f"fitted on every {every}th sample")
    ica.fit(fit_on, decim=every, verbose="error")
    if getattr(ica, "n_iter_", 0) >= p.max_iter:
        notes.append(f"stopped after {p.max_iter} iterations without converging — fewer "
                     "components or more bad channels may help")
    fit_raw.ic_labels = fit_raw.ic_proba = None
    if p.iclabel:
        try:
            fit_raw.ic_labels, fit_raw.ic_proba = iclabel(raw, ica, chans, notch, tmin)
        except Exception as e:  # noqa: BLE001
            notes.append(f"ICLabel not run: {e}")
    fit_raw.fit_note = "; ".join(notes)
    # every decision taken, so an exported script can repeat exactly this fit
    fit_raw.fit_steps = dict(
        chans=list(chans), l_freq=max(p.l_freq, 0.1), h_freq=h, notch=notch,
        tmin=start or None, resample=(fit_raw.info["sfreq"] if decim > 1 else None),
        n_components=nc, method=method, fit_params=fit_params, max_iter=p.max_iter,
        seed=p.seed, segments=how, reject_uv=p.reject_uv,
        autoreject_fraction=p.autoreject_fraction, decim=every)
    return ica, fit_raw


def ica_segments(fit_raw: mne.io.BaseRaw) -> mne.Epochs:
    """The ICA data cut into 1 s segments (ERP guide)."""
    events = mne.make_fixed_length_events(fit_raw, duration=1.0)
    return mne.Epochs(fit_raw, events, tmin=0.0, tmax=1.0, baseline=None, preload=True,
                      reject_by_annotation=True, verbose="error")


def ptp_good_segments(seg: mne.BaseEpochs, reject_uv) -> np.ndarray:
    """Segments within a peak-to-peak threshold (MNE's ``drop_bad``)."""
    good = np.ones(len(seg), bool)
    if reject_uv:
        kept = seg.copy().drop_bad(reject=dict(eeg=reject_uv * 1e-6), verbose="error")
        good[:] = False
        good[np.searchsorted(seg.selection, kept.selection)] = True
    return good


def fit_subset(n: int, fraction: float, seed: int = 42):
    """Indices of the epochs AutoReject is fitted on: all of them (``fraction`` 1), or a
    random, reproducible share. Fitting on a share is quicker and needs less memory, but
    on noisy data the result can depend strongly on which epochs are drawn."""
    if fraction >= 1 or n < 2:
        return slice(None)
    k = max(2, int(round(fraction * n)))
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


#: AutoReject's fit needs roughly this many times the memory of the epochs it is fitted
#: on (measured: 1.16 GB of epochs → +14.2 GB)
AUTOREJECT_MEMORY_FACTOR = 12.5


def autoreject_bad_segments(epochs: mne.BaseEpochs, seed: int = 42,
                            fraction: float = 1.0) -> np.ndarray:
    """AutoReject (``n_interpolate=[1, 2, 4]``, ``random_state=42``) marks bad segments,
    fitted on all segments or on a random share (:func:`fit_subset`)."""
    import warnings

    from autoreject import AutoReject
    ar = AutoReject(n_interpolate=[1, 2, 4], random_state=seed, n_jobs=1, verbose=False,
                    picks=mne.pick_types(epochs.info, eeg=True, eog=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar.fit(epochs[fit_subset(len(epochs), fraction, seed)])
        return np.asarray(ar.get_reject_log(epochs).bad_epochs, bool)


#: ICLabel classes (MNE-ICALabel)
IC_CLASSES = ("brain", "muscle artifact", "eye blink", "heart beat", "line noise",
              "channel noise", "other")


def iclabel(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, chans: list[str], notch=None,
            tmin: float | None = None) -> tuple[list[str], np.ndarray]:
    """ICLabel class and probability of every component (MNE-ICALabel). ICLabel was
    trained on 1–100 Hz, average-referenced data, so it gets such a copy of the channels
    the ICA was fitted on."""
    from mne_icalabel import label_components
    if not has_positions(raw.copy().pick(chans)):
        raise ValueError("the EEG channels have no electrode positions (load a montage)")
    nyq = raw.info["sfreq"] / 2
    lab_raw = filtered_copy(raw, 1.0, min(100.0, nyq - 1), notch, picks=chans, tmin=tmin)
    lab_raw.info["bads"] = []
    lab_raw.set_eeg_reference("average", projection=False, verbose="error")
    with mne.utils.use_log_level("error"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = label_components(lab_raw, ica, method="iclabel")
    return list(out["labels"]), np.asarray(out["y_pred_proba"], float)


FRONTAL = ("Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4")
#: EOG stand-ins without EOG electrodes (ERP guide: Fp1, Fp2, F7, F8)
EOG_PROXIES = ("Fp1", "Fp2", "F7", "F8")



def blink_locked_score(src: np.ndarray, sfreq: float, times: np.ndarray,
                       blinks: pd.DataFrame | None, tmin=-0.2, tmax=0.5) -> np.ndarray | None:
    """Per component: max over time of |mean| / SEM of blink-locked segments."""
    if blinks is None or len(blinks) < 5:
        return None
    on = np.searchsorted(times, blinks["onset"].to_numpy(float))
    i0, i1 = int(round(tmin * sfreq)), int(round(tmax * sfreq))
    on = on[(on + i0 >= 0) & (on + i1 < src.shape[1])][:3000]
    if len(on) < 5:
        return None
    z = (src - src.mean(1, keepdims=True)) / (src.std(1, keepdims=True) + 1e-20)
    idx = on[:, None] + np.arange(i0, i1)[None, :]
    seg = z[:, idx]                                  # (comp, blinks, time)
    seg = seg - seg[:, :, : -i0].mean(2, keepdims=True)
    t = seg.mean(1) / (seg.std(1) / np.sqrt(len(on)) + 1e-20)
    return np.abs(t).max(1)


def score_blink_components(ica: mne.preprocessing.ICA, fit_raw: mne.io.BaseRaw,
                           blinks: pd.DataFrame | None = None, eog_channels=None,
                           z_thresh: float = 3.0, max_ic: int = 2,
                           iclabel_p: float = 0.7) -> pd.DataFrame:
    """Rank ICA components as blink candidates.

    * ``eog_score``: |correlation| with EOG channels via ``ICA.find_bads_eog``.
      Without EOG channels, the frontal electrodes Fp1, Fp2, F7 and F8 act as proxies.
      As in the ERP guide, the z threshold starts at 3.5 and steps down by 0.05 until
      ``max_ic`` components are found (not below 2.0).
    * ``blink_t``: how strongly the component's blink-locked average (eye-tracker
      blink onsets, −0.2…0.5 s) stands out: max |mean| / SEM, robust to jitter
      (z-outliers above ``z_thresh`` are suggested).
    * ``iclabel`` / ``iclabel_p``: the ICLabel class and its probability (from
      :func:`fit_ica`); "eye blink" with p ≥ ``iclabel_p`` is suggested.
    """
    n = ica.n_components_
    df = pd.DataFrame({"component": np.arange(n)})
    eog = [c for c in (eog_channels or []) if c in fit_raw.ch_names]
    if not eog:
        eog = [c for c in fit_raw.ch_names
               if mne.channel_type(fit_raw.info, fit_raw.ch_names.index(c)) == "eog"]
    proxy = False
    if not eog:
        lower = {c.lower(): c for c in fit_raw.ch_names}
        eog = [lower[c.lower()] for c in EOG_PROXIES if c.lower() in lower]
        if not eog:
            eog = [lower[c.lower()] for c in FRONTAL if c.lower() in lower][:2]
        proxy = True
    flagged = set()
    if eog:
        z = 3.5
        while True:
            idx, scores = ica.find_bads_eog(fit_raw, ch_name=eog, threshold=z,
                                            measure="zscore", verbose="error")
            if len(idx) >= max_ic or z <= 2.0:
                break
            z = round(z - 0.05, 2)
        df.attrs["eog_z"] = z
        scores = np.atleast_2d(np.asarray(scores))
        df["eog_score"] = np.abs(scores).max(0)
        flagged |= set(int(i) for i in idx)
    else:
        df["eog_score"] = np.nan
    src = ica.get_sources(fit_raw).get_data()
    t_score = blink_locked_score(src, fit_raw.info["sfreq"], fit_times(fit_raw), blinks)
    if t_score is not None:
        df["blink_t"] = t_score
        mad = np.median(np.abs(t_score - np.median(t_score))) * 1.4826 or 1e-9
        z = (t_score - np.median(t_score)) / mad
        flagged |= set(np.flatnonzero((z > z_thresh) & (t_score > 5)).tolist())
    else:
        df["blink_t"] = np.nan
    labels, proba = getattr(fit_raw, "ic_labels", None), getattr(fit_raw, "ic_proba", None)
    if labels is not None and len(labels) == n:
        df["iclabel"] = labels
        df["iclabel_p"] = proba
        flagged |= set(np.flatnonzero((df["iclabel"] == "eye blink") &
                                      (df["iclabel_p"] >= iclabel_p)).tolist())
    else:
        df["iclabel"] = ""
        df["iclabel_p"] = np.nan
    df["suggested"] = df["component"].isin(flagged)
    df.attrs["eog_channels"] = eog
    df.attrs["proxy"] = proxy
    return df


# =========================================================================== pipeline
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
        parts = [f"{len(self.bads)} bad channel(s)" + (" interpolated" if self.bads and
                                                        self.interpolate_bads else "")]
        band = f"{self.l_freq or 0:g}–{self.h_freq:g} Hz" if self.h_freq else \
            (f"high-pass {self.l_freq:g} Hz" if self.l_freq else "no band-pass")
        parts.append(band)
        if self.notch:
            parts.append(f"notch {self.notch:g} Hz")
        if self.ica_exclude:
            parts.append(f"ICA −{len(self.ica_exclude)} comp.")
        parts.append(f"ref: {self.reference or 'as recorded'}")
        return " · ".join(parts)


def run_preprocessing(raw: mne.io.BaseRaw, p: PreprocParams,
                      ica: mne.preprocessing.ICA | None = None) -> mne.io.BaseRaw:
    """Apply the preprocessing stage to a copy of ``raw``."""
    out = filtered_copy(raw, p.l_freq, p.h_freq, p.notch)
    out.info["bads"] = [b for b in p.bads if b in out.ch_names]
    if ica is not None and p.ica_exclude:
        missing = [c for c in ica.ch_names if c not in out.ch_names]
        if missing:
            raise ValueError(f"ICA was fitted on channels that no longer exist: {missing}")
        ica = ica.copy()
        ica.exclude = list(p.ica_exclude)
        ica.apply(out, verbose="error")
    if p.interpolate_bads and out.info["bads"]:
        out.interpolate_bads(reset_bads=True, verbose="error")
    ref = parse_reference(p.reference, out)
    if ref == "average":
        out.set_eeg_reference("average", projection=False, verbose="error")
    elif ref:
        out.set_eeg_reference(ref, projection=False, verbose="error")
    return out


def blink_locked_average(raw: mne.io.BaseRaw, blinks: pd.DataFrame, channels,
                         tmin=-0.3, tmax=0.6) -> tuple[np.ndarray, np.ndarray] | None:
    """Average of ``channels`` around eye-tracker blink onsets (µV), to check ICA."""
    chans = [c for c in channels if c in raw.ch_names and c not in raw.info["bads"]]
    if blinks is None or not len(blinks) or not chans:
        return None
    sf = raw.info["sfreq"]
    on = np.round(blinks["onset"].to_numpy(float) * sf).astype(int)
    i0, i1 = int(round(tmin * sf)), int(round(tmax * sf))
    on = on[(on + i0 >= 0) & (on + i1 < raw.n_times)]
    if not len(on):
        return None
    x = raw.get_data(picks=chans).mean(0) * 1e6
    seg = np.stack([x[o + i0:o + i1] for o in on[:2000]])
    seg -= seg[:, : -i0].mean(1, keepdims=True) if i0 < 0 else 0
    return np.arange(i0, i1) / sf, seg.mean(0)





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
        s = f"{self.tmin:g} … {self.tmax:g} s"
        if self.reject_uv:
            s += f", reject ptp > {self.reject_uv:g} µV"
        if self.flat_uv:
            s += f", flat < {self.flat_uv:g} µV"
        if self.autoreject:
            s += ", AutoReject" + (f" (fitted on {self.autoreject_fraction:.0%})"
                                   if self.autoreject_fraction < 1 else "")
        if self.drop_blinks:
            s += ", no blinks"
        if self.delay_ms:
            s += f", EEG delay {self.delay_ms:+g} ms"
        return s


def overlaps(onsets: np.ndarray, tmin: float, tmax: float,
             intervals: pd.DataFrame | None) -> np.ndarray:
    """True for each onset whose window [onset+tmin, onset+tmax] overlaps an interval."""
    out = np.zeros(len(onsets), bool)
    if intervals is None or not len(intervals):
        return out
    s = intervals["onset"].to_numpy(float)
    e = s + intervals["duration"].to_numpy(float)
    order = np.argsort(s)
    s, e = s[order], e[order]
    emax = np.maximum.accumulate(e)
    a, b = onsets + tmin, onsets + tmax
    k = np.searchsorted(s, b, side="right") - 1  # last interval starting before b
    valid = k >= 0
    out[valid] = emax[k[valid]] > a[valid]
    return out


def build_group_epochs(raw: mne.io.BaseRaw, groups: dict[str, pd.DataFrame],
                       settings: EpochSettings, blinks: pd.DataFrame | None = None
                       ) -> tuple[dict[str, mne.Epochs], pd.DataFrame]:
    """One ``mne.Epochs`` per group (metadata = event rows + ``group`` column).

    Returns the epochs and a summary table per group: selected; dropped before
    epoching because of a blink, the window falling outside the recording or another
    event on the same sample; rejected by amplitude
    (peak-to-peak/flat) or by a BAD annotation; kept; and the channels behind
    amplitude rejections.
    """
    out, rows = {}, []
    for name, table in groups.items():
        tab = table.copy()
        if settings.delay_ms:  # the EEG lags the events: cut epochs later
            tab["onset"] = tab["onset"] + settings.delay_ms / 1000.0
        n_sel = len(tab)
        n_blink = 0
        if settings.drop_blinks and blinks is not None and len(tab):
            hit = overlaps(tab["onset"].to_numpy(float), settings.tmin, settings.tmax, blinks)
            n_blink = int(hit.sum())
            tab = tab[~hit]
        row = {"group": name, "selected": n_sel, "blink": n_blink,
               "outside recording": 0, "same sample": 0,
               "rejected (amplitude)": 0, "rejected (BAD annotation)": 0, "kept": 0,
               "rejected by": ""}
        if len(tab):
            tab = tab.assign(group=name)
            try:
                ep = epochs_from_table(raw, tab, settings.tmin, settings.tmax, baseline=None,
                                         reject_uv=settings.reject_uv)
            except ValueError as e:
                ep = None
                row.update(getattr(e, "counts", {"outside recording": len(tab)}))
            if ep is not None and settings.flat_uv:
                ep.drop_bad(flat=dict(eeg=settings.flat_uv * 1e-6), verbose="error")
            if ep is not None:
                row.update(ep.selection_counts)
                why = drop_reasons(ep)
                row["rejected (amplitude)"] = why["amplitude"] + why["other"]
                row["rejected (BAD annotation)"] = why["BAD annotation"]
                row["kept"] = len(ep)
                causes = rejection_causes(ep)
                row["rejected by"] = ", ".join(f"{c} ({n})" for c, n in list(causes.items())[:6])
                if len(ep):
                    out[name] = ep
        rows.append(row)
    summary = pd.DataFrame(rows)
    if settings.autoreject and out:
        out, summary = apply_autoreject(out, summary, settings.autoreject_fraction)
    return out, summary


def apply_autoreject(epochs: dict[str, mne.Epochs], summary: pd.DataFrame,
                     fraction: float = 1.0, seed: int = 42
                     ) -> tuple[dict[str, mne.Epochs], pd.DataFrame]:
    """AutoReject (``n_interpolate=[1, 2, 4]``, ``random_state=42``), as in the ERP guide:
    fitted once on the epochs of every group (all, or a random share), then each
    group is cleaned — bad channels are interpolated within an epoch and the epochs that
    cannot be repaired are dropped. Needs electrode positions and ≥ 10 epochs."""
    import warnings

    summary = summary.copy()
    summary["rejected (AutoReject)"] = 0
    summary["interpolated (AutoReject)"] = 0
    every = list(epochs.values())
    n = sum(len(e) for e in every)
    if not has_positions(every[0]):
        summary.attrs["autoreject"] = "not applied: the EEG channels have no positions"
        return epochs, summary
    if n < 10:
        summary.attrs["autoreject"] = "not applied: fewer than 10 epochs"
        return epochs, summary
    from autoreject import AutoReject
    with warnings.catch_warnings(), mne.utils.use_log_level("error"):
        warnings.simplefilter("ignore")
        # fitting needs only the data: one array of every group's epochs (their event
        # codes may clash, which mne.concatenate_epochs refuses)
        sizes = [len(e) for e in every]
        pick = fit_subset(n, fraction, seed)
        pick = np.arange(n) if isinstance(pick, slice) else pick
        starts = np.cumsum([0] + sizes)
        data = np.concatenate([every[k].get_data(item=pick[(pick >= starts[k]) &
                                                         (pick < starts[k + 1])] - starts[k])
                               for k in range(len(every))])
        allep = mne.EpochsArray(data, every[0].info, tmin=every[0].tmin, baseline=None,
                                verbose="error")
        ar = AutoReject(n_interpolate=[1, 2, 4], random_state=seed, n_jobs=1,
                        verbose=False, picks=mne.pick_types(allep.info, eeg=True, eog=False))
        ar.fit(allep)
        out = {}
        for g, ep in epochs.items():
            ep = ep.copy().load_data()
            log = ar.get_reject_log(ep)
            clean = ar.transform(ep)
            i = summary.index[summary["group"] == g][0]
            summary.loc[i, "rejected (AutoReject)"] = int(np.sum(log.bad_epochs))
            summary.loc[i, "interpolated (AutoReject)"] = int(
                (log.labels[~log.bad_epochs] == 2).sum())
            summary.loc[i, "kept"] = len(clean)
            if len(clean):
                out[g] = clean
    summary.attrs["autoreject"] = "applied"
    return out, summary
