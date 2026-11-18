"""Several recordings as one: file entities (subject/session/run…) and concatenation.

Loading several files (XDF, EDF/BDF/…, or BIDS runs) concatenates them into one
:class:`Recording`, so preprocessing, epoching and analysis run unchanged:

* EEG is joined with :func:`mne.concatenate_raws` on the channels all files share
  (resampled to the first file's rate if needed); MNE marks the joins with
  ``BAD boundary`` annotations, so no epoch spans two files.
* gaze samples and markers are shifted onto the joined time base.
* each file becomes a *segment* (``Recording.segments``) with its BIDS entities —
  from the BIDS dataset or parsed from a BIDS-style file name
  (``sub-01_ses-02_task-x_run-003_eeg.xdf``). Every event inherits them as
  ``subject``, ``session``, ``bids_task``, ``acq``, ``run`` and ``file`` columns
  (text, exactly as in the file name: ``run == "003"``).
"""

from __future__ import annotations

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from ..model import GazeData, Recording

# "task" is called bids_task in the event table: markers often carry a task of their own
ENTITY_NAMES = {"sub": "subject", "ses": "session", "task": "bids_task", "acq": "acq",
                "run": "run"}
SEGMENT_COLUMNS = ("subject", "session", "bids_task", "acq", "run", "file")
_ENTITY = re.compile(r"(?:^|[_/\\])(sub|ses|task|acq|run)-([A-Za-z0-9.+]+?)(?=_|\.|/|\\|$)")


def entities_from_path(path: str | Path) -> dict:
    """BIDS entities from a file name and its folders (later parts win)."""
    p = Path(path)
    out = {}
    for part in list(p.parent.parts[-3:]) + [p.name]:
        for key, val in _ENTITY.findall("_" + part):
            out[ENTITY_NAMES[key]] = val
    out["file"] = p.name
    return out


def set_segments(rec: Recording, entities: dict) -> Recording:
    """One segment covering the whole recording."""
    ent = {k: v for k, v in entities.items() if v not in (None, "")}
    rec.segments = [{"start": 0.0, "end": float(rec.duration), **ent}]
    return rec


def _relative_annotations(raw: mne.io.BaseRaw) -> mne.Annotations:
    a = raw.annotations
    off = raw.first_time if a.orig_time is not None else 0.0
    return mne.Annotations(a.onset - off, a.duration, a.description)


def concatenate(recs: list[Recording], gap: float = 1.0) -> Recording:
    """Join recordings end to end (see module docstring)."""
    if len(recs) == 1:
        return recs[0]
    has_eeg = [r.raw is not None for r in recs]
    if any(has_eeg) and not all(has_eeg):
        raise ValueError("Cannot combine files with and without EEG")
    notes = []
    offsets, starts = [], 0.0
    if all(has_eeg):
        raws = [r.raw for r in recs]
        common = [c for c in raws[0].ch_names if all(c in r.ch_names for r in raws[1:])]
        if not common:
            raise ValueError("The files have no EEG channel names in common")
        dropped = sorted({c for r in raws for c in r.ch_names} - set(common))
        if dropped:
            notes.append(f"channels not in every file left out: {', '.join(dropped[:12])}"
                         f"{' …' if len(dropped) > 12 else ''}")
        sf = raws[0].info["sfreq"]
        types = dict(zip(raws[0].ch_names, raws[0].get_channel_types()))
        # a channel bad in any file is bad in the joined recording (MNE needs one list)
        bads = sorted({b for r in raws for b in r.info["bads"] if b in common},
                      key=common.index)
        if any(set(r.info["bads"]) & set(common) != set(bads) for r in raws):
            notes.append(f"bad channels combined from all files: {', '.join(bads)}")
        prepared = []
        for r in raws:
            r = r.copy().load_data().pick(common)
            r.info["bads"] = list(bads)
            ann = _relative_annotations(r)
            r.set_annotations(None)
            r.set_meas_date(None)
            r.set_annotations(ann)
            d = abs(r.info["sfreq"] - sf)
            if d and d / sf < 1e-3:  # effective rates of the same amplifier differ by ppm
                with r.info._unlock():
                    r.info["sfreq"] = sf
                if not any(n.startswith("rates aligned") for n in notes):
                    notes.append(f"rates aligned to {sf:.4f} Hz (< 0.1 % apart)")
            elif d:
                notes.append(f"resampled {r.info['sfreq']:g} → {sf:g} Hz")
                r.resample(sf, verbose="error")
            r.set_channel_types({c: types[c] for c in common
                                 if types[c] != dict(zip(r.ch_names,
                                                         r.get_channel_types()))[c]},
                                verbose="error")
            prepared.append(r)
        for r in prepared:
            offsets.append(starts)
            starts += r.n_times / sf
        raw = mne.concatenate_raws(prepared, verbose="error")
        montage = recs[0].raw.get_montage()
        if montage is not None:
            raw.set_montage(montage, on_missing="ignore", verbose="error")
    else:
        raw = None
        for r in recs:
            offsets.append(starts)
            starts += r.duration + gap

    gaze = None
    gz = [(r.gaze, off) for r, off in zip(recs, offsets) if r.has_gaze]
    if gz:
        first = gz[0][0]
        keys = set(first.extra)
        for g, _ in gz[1:]:
            keys &= set(g.extra)
        pupil = None
        if all(g.pupil is not None for g, _ in gz):
            pupil = np.concatenate([g.pupil for g, _ in gz])
        gaze = GazeData(np.concatenate([g.times + off for g, off in gz]),
                        np.concatenate([g.x for g, _ in gz]),
                        np.concatenate([g.y for g, _ in gz]), pupil, first.sfreq,
                        first.screen_size, first.label,
                        {k: np.concatenate([g.extra[k] for g, _ in gz]) for k in keys})
        if len(gz) != len(recs):
            notes.append(f"gaze in {len(gz)} of {len(recs)} files")

    events = [r.events.assign(onset=r.events["onset"] + off)
              for r, off in zip(recs, offsets) if len(r.events)]
    segments = []
    for r, off in zip(recs, offsets):
        for s in (r.segments or [{"start": 0.0, "end": r.duration}]):
            segments.append({**s, "start": s["start"] + off, "end": s["end"] + off})
    out = Recording(raw=raw, gaze=gaze, source=recs[0].source)
    out.events = (pd.concat(events, ignore_index=True) if events else recs[0].events.iloc[0:0])
    out.segments = segments
    out.streams = recs[0].streams
    out.sources = [r.source for r in recs]
    if any(r.metadata for r in recs):  # first file's info; bad channels from all files
        out.metadata = dict(recs[0].metadata)
        out.metadata["bad_channels"] = {c: why for r in recs
                                        for c, why in r.metadata.get("bad_channels", {}).items()}
    desc = [f"{len(recs)} files combined" + (": " + "; ".join(notes) if notes else "")]
    for s, r in zip(segments, recs):
        desc.append(f"[{s.get('file', r.source)} @ {s['start']:.1f} s] {r.description}")
    out.description = "\n".join(desc)
    return out


def segment_columns(onsets: np.ndarray, segments: list[dict]) -> dict[str, np.ndarray]:
    """Entity columns for events at ``onsets`` from the segment they fall in."""
    if not segments:
        return {}
    starts = np.array([s["start"] for s in segments])
    idx = np.clip(np.searchsorted(starts, onsets, side="right") - 1, 0, len(segments) - 1)
    out = {}
    for col in SEGMENT_COLUMNS:
        vals = [s.get(col) for s in segments]
        if any(v not in (None, "") for v in vals):
            out[col] = np.array([vals[i] if vals[i] is not None else "" for i in idx],
                                dtype=object)
    return out
