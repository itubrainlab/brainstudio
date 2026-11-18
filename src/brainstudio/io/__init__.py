"""Data loaders: BIDS datasets, XDF files, and standalone eye-tracking files."""

from __future__ import annotations

from pathlib import Path

import mne

from ..model import GazeData, Recording
from .bids import BidsRun, gaze_from_raw, load_bids_run, read_bids_eyetrack, scan_bids
from .multi import concatenate, entities_from_path, set_segments
from .xdf import describe_streams, load_xdf

__all__ = [
    "BidsRun", "scan_bids", "load_bids_run", "read_bids_eyetrack", "gaze_from_raw",
    "load_xdf", "describe_streams", "load", "load_many", "load_bids_runs",
    "load_eyetracking_file", "concatenate", "entities_from_path",
]


def load_eyetracking_file(path: str | Path, screen_size=(1920, 1080)) -> GazeData:
    """Load a standalone gaze file: EyeLink .asc, BIDS physio TSV, or XDF."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".asc"):
        raw = mne.io.read_raw_eyelink(path, verbose="error")
        g = gaze_from_raw(raw, screen_size, drop=False)
        if g is None:
            raise ValueError("No gaze channels in EyeLink file")
        return g
    if ".tsv" in name:
        return read_bids_eyetrack(path, screen_size=screen_size)[0]
    if name.endswith((".xdf", ".xdfz")):
        rec = load_xdf(path, screen_size)
        if rec.gaze is None:
            raise ValueError("No gaze stream in XDF file")
        return rec.gaze
    raise ValueError(f"Unsupported eye-tracking file: {path.name}")


def load(path: str | Path, screen_size=(1920, 1080), roles=None, options=None) -> Recording:
    """Load a single file (XDF, any MNE-readable EEG file, EyeLink .asc).

    ``roles`` (XDF only): stream id → EEG / gaze / markers / ignore overrides.
    The recording gets one segment with the BIDS entities of the file name."""
    rec = _load_one(path, screen_size, roles, options)
    rec.sources = [Path(path)]
    return set_segments(rec, entities_from_path(path))


def load_many(paths, screen_size=(1920, 1080), roles_by_name=None,
              options_by_name=None) -> Recording:
    """Load several files and join them (:func:`brainstudio.io.multi.concatenate`).

    For XDF, roles and options are given per stream *name*, so they apply to
    the same stream in every file."""
    paths = [Path(p) for p in paths]
    recs = []
    for p in paths:
        roles = options = None
        if p.name.lower().endswith((".xdf", ".xdfz")) and (roles_by_name or options_by_name):
            keys = {s["name"]: s["key"] for s in describe_streams(p).rename(
                columns={"stream_id": "key"}).astype({"key": str}).to_dict("records")}
            roles = {keys[n]: r for n, r in (roles_by_name or {}).items() if n in keys}
            options = {keys[n]: o for n, o in (options_by_name or {}).items() if n in keys}
        recs.append(load(p, screen_size, roles or None, options or None))
    return concatenate(recs)


def load_bids_runs(root, runs, screen_size=None) -> Recording:
    """Load and join several BIDS runs, each segment carrying its entities."""
    recs = []
    for run in runs:
        r = load_bids_run(root, run, screen_size)
        ent = {k: run.entities.get(e) for k, e in (("subject", "subject"), ("session", "session"),
                                                   ("bids_task", "task"), ("acq", "acquisition"),
                                                   ("run", "run"))}
        ent["file"] = Path(run.eeg or (run.gaze[0] if run.gaze else root)).name
        recs.append(set_segments(r, ent))
    return concatenate(recs)


def _load_one(path, screen_size, roles, options) -> Recording:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".xdf", ".xdfz")):
        return load_xdf(path, screen_size, roles=roles, options=options)
    if name.endswith(".asc"):
        raw = mne.io.read_raw_eyelink(path, verbose="error")
        rec = Recording(gaze=gaze_from_raw(raw, screen_size, drop=False), source=path)
        a = raw.annotations
        import pandas as pd
        rec.events = pd.DataFrame({"onset": a.onset - (raw.first_time if a.orig_time else 0),
                                   "duration": a.duration, "description": a.description})
        return rec
    raw = mne.io.read_raw(path, preload=True, verbose="error")
    rec = Recording(raw=raw, source=path)
    rec.gaze = gaze_from_raw(raw, screen_size)
    return rec
