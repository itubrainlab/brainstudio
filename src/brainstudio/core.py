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
