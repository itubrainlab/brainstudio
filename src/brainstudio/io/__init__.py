"""Data loaders: BIDS datasets, XDF files, and standalone eye-tracking files."""

from __future__ import annotations

from pathlib import Path

import mne

from ..model import GazeData, Recording
from .bids import BidsRun, gaze_from_raw, load_bids_run, read_bids_eyetrack, scan_bids
from .xdf import describe_streams, load_xdf

__all__ = [
    "BidsRun", "scan_bids", "load_bids_run", "read_bids_eyetrack", "gaze_from_raw",
    "load_xdf", "describe_streams", "load", "load_eyetracking_file",
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


def load(path: str | Path, screen_size=(1920, 1080)) -> Recording:
    """Load a single file (XDF, any MNE-readable EEG file, EyeLink .asc)."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".xdf", ".xdfz")):
        return load_xdf(path, screen_size)
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
