"""BIDS dataset browsing and loading (EEG via MNE-BIDS, eye tracking via BIDS physio).

Eye-tracking data is read from the BIDS eye-tracking format (BIDS >= 1.10,
formerly BEP020): ``*_recording-<label>_physio.tsv.gz`` + ``.json`` sidecar
with ``"PhysioType": "eyetrack"`` and columns such as ``timestamp``,
``x_coordinate``, ``y_coordinate``, ``pupil_size``. Messages from
``*_physioevents.tsv.gz`` are added as events. Legacy ``*_eyetrack.tsv[.gz]``
files are also accepted. Gaze is aligned to the EEG using the sidecar's
``StartTime`` (seconds relative to the first EEG sample).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, get_entities_from_fname, read_raw_bids

from ..eeg import set_standard_montage
from ..model import GazeData, Recording
from .xdf import gaze_from_matrix

EEG_EXTS = (".vhdr", ".edf", ".bdf", ".set", ".fif", ".cnt", ".mff")
ENTITY_KEYS = ("subject", "session", "task", "acquisition", "run")


@dataclass
class BidsRun:
    """One acquisition run: an EEG file and/or eye-tracking files."""

    entities: dict
    eeg: Path | None = None
    gaze: list[Path] = field(default_factory=list)
    datatype: str = "eeg"

    @property
    def key(self) -> tuple:
        return tuple(self.entities.get(k) or "" for k in ENTITY_KEYS)

    @property
    def label(self) -> str:
        e = self.entities
        parts = [f"task-{e['task']}" if e.get("task") else None,
                 f"acq-{e['acquisition']}" if e.get("acquisition") else None,
                 f"run-{e['run']}" if e.get("run") else None]
        mods = (["EEG"] if self.eeg else []) + (["gaze"] if self.gaze else [])
        return " ".join(p for p in parts if p) + f"  [{'+'.join(mods)}]"


def _skip(p: Path, root: Path) -> bool:
    rel = p.relative_to(root).parts
    return any(part in ("derivatives", "sourcedata", "code") or part.startswith(".")
               for part in rel[:-1])


def _sidecar(path: Path) -> dict:
    name = path.name
    for ext in (".tsv.gz", ".tsv", ".vhdr", ".edf", ".bdf", ".set", ".fif"):
        if name.endswith(ext):
            js = path.with_name(name[: -len(ext)] + ".json")
            if js.exists():
                return json.loads(js.read_text())
    return {}


def _is_eyetrack_file(p: Path) -> bool:
    n = p.name
    if "_eyetrack.tsv" in n:
        return True
    if n.endswith("_physio.tsv.gz") or n.endswith("_physio.tsv"):
        meta = _sidecar(p)
        if meta.get("PhysioType") == "eyetrack" or "eye" in n.split("_recording-")[-1][:8]:
            return True
        cols = [str(c).lower() for c in meta.get("Columns", [])]
        return any("coordinate" in c or "gaze" in c for c in cols)
    return False


def scan_bids(root: str | Path) -> list[BidsRun]:
    """Find all EEG / eye-tracking runs in a BIDS dataset."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    runs: dict[tuple, BidsRun] = {}

    def get(path: Path) -> BidsRun:
        ent = get_entities_from_fname(path.name, on_error="ignore")
        ent = {k: ent.get(k) for k in ENTITY_KEYS}
        r = BidsRun(ent, datatype=path.parent.name)
        return runs.setdefault(r.key, r)

    for p in sorted(root.rglob("sub-*")):
        if not p.is_file() or _skip(p, root):
            continue
        if p.name.endswith("_eeg" + p.suffix) and p.suffix in EEG_EXTS:
            get(p).eeg = p
        elif _is_eyetrack_file(p):
            get(p).gaze.append(p)
    return sorted(runs.values(), key=lambda r: r.key)


def dataset_description(root: str | Path) -> dict:
    f = Path(root) / "dataset_description.json"
    return json.loads(f.read_text()) if f.exists() else {}


# --------------------------------------------------------------------------- gaze
def _infer_time_scale(ts: np.ndarray, sfreq: float | None, units: str | None) -> float:
    """Factor converting the timestamp column to seconds."""
    units = (units or "").lower()
    known = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
    if units in known:
        return known[units]
    if sfreq and len(ts) > 1:
        step = float(np.median(np.diff(ts)))
        if step > 0:
            ratio = (1.0 / sfreq) / step
            return 10.0 ** round(np.log10(ratio))
    return 1.0


def read_bids_eyetrack(path: str | Path, eeg_meta: dict | None = None,
                       screen_size=None) -> tuple[GazeData, pd.DataFrame]:
    """Read one BIDS eye-tracking physio file. Returns gaze and message events."""
    path = Path(path)
    meta = {**(eeg_meta or {}), **_sidecar(path)}
    cols = meta.get("Columns")
    compression = "gzip" if path.name.endswith(".gz") else None
    if cols:
        df = pd.read_csv(path, sep="\t", header=None, names=cols, na_values=["n/a", "."],
                         compression=compression)
    else:  # legacy files with header
        df = pd.read_csv(path, sep="\t", na_values=["n/a", "."], compression=compression)
    df = df.apply(pd.to_numeric, errors="coerce")
    sfreq = meta.get("SamplingFrequency")
    sfreq = float(sfreq) if sfreq not in (None, "n/a") else None
    start = float(meta.get("StartTime", 0.0) or 0.0)
    labels = [str(c) for c in df.columns]
    tcol = next((c for c in labels if c.lower() in ("timestamp", "time", "eye_timestamp")), None)
    if tcol is not None:
        ts = df[tcol].to_numpy(float)
        units = (meta.get(tcol) or {}).get("Units") if isinstance(meta.get(tcol), dict) else None
        scale = _infer_time_scale(ts, sfreq, units)
        times = (ts - ts[0]) * scale + start
    else:
        if not sfreq:
            raise ValueError(f"{path.name}: no timestamp column and no SamplingFrequency")
        ts, scale = None, 1.0
        times = start + np.arange(len(df)) / sfreq
    data_cols = [c for c in labels if c != tcol]
    res = meta.get("ScreenResolution") or meta.get("ScreenResolutionPx")
    size = tuple(int(v) for v in (screen_size or res or (1920, 1080)))[:2]
    gaze = gaze_from_matrix(times, df[data_cols].to_numpy(float), data_cols, size,
                            label=path.name.split("_physio")[0], sfreq=sfreq)

    # physioevents (messages)
    events = pd.DataFrame(columns=["onset", "duration", "description"])
    ev_path = path.with_name(path.name.replace("_physio.tsv", "_physioevents.tsv"))
    if ev_path != path and ev_path.exists():
        ev_meta = _sidecar(ev_path)
        ev_cols = ev_meta.get("Columns")
        ev = pd.read_csv(ev_path, sep="\t", header=None if ev_cols else "infer",
                         names=ev_cols, na_values=["n/a"],
                         compression="gzip" if ev_path.suffix == ".gz" else None)
        if "onset" in ev:
            on = pd.to_numeric(ev["onset"], errors="coerce").to_numpy(float)
            # onsets share the timestamp clock when a timestamp column exists
            on = (on - ts[0]) * scale + start if ts is not None else on + start
            dur = (pd.to_numeric(ev["duration"], errors="coerce").to_numpy(float)
                   if "duration" in ev else np.zeros(len(ev)))
            dur = np.nan_to_num(dur * (scale if ts is not None else 1.0))
            desc_col = next((c for c in ("message", "trial_type", "value") if c in ev), None)
            if desc_col:
                desc = ev[desc_col].astype(str)
                keep = ev[desc_col].notna().to_numpy()
                events = pd.DataFrame({"onset": on[keep], "duration": dur[keep],
                                       "description": desc[keep].to_numpy()})
    return gaze, events


def gaze_from_raw(raw: mne.io.BaseRaw, screen_size=(1920, 1080),
                  drop: bool = True) -> GazeData | None:
    """Extract MNE ``eyegaze``/``pupil`` channels (e.g. EyeLink, BIDS) as GazeData."""
    picks = mne.pick_types(raw.info, eyetrack=True, exclude=[])
    if not len(picks):
        return None
    names = [raw.ch_names[i] for i in picks]
    data = raw.get_data(picks=picks).T
    g = gaze_from_matrix(raw.times.copy(), data, names, screen_size, "eyetrack",
                         raw.info["sfreq"])
    if drop:
        raw.drop_channels(names)
    return g


def load_bids_run(root: str | Path, run: BidsRun, screen_size=None) -> Recording:
    root = Path(root)
    rec = Recording(source=run.eeg or (run.gaze[0] if run.gaze else root))
    eeg_meta = {}
    if run.eeg is not None:
        ent = run.entities
        bp = BIDSPath(root=root, subject=ent.get("subject"), session=ent.get("session"),
                      task=ent.get("task"), acquisition=ent.get("acquisition"),
                      run=ent.get("run"), datatype=run.eeg.parent.name, suffix="eeg",
                      extension=run.eeg.suffix)
        try:
            raw = read_raw_bids(bp, verbose="error")
        except Exception:  # fall back to plain MNE reader if sidecars are broken
            raw = mne.io.read_raw(run.eeg, verbose="error")
        raw.load_data(verbose="error")
        eeg_meta = _sidecar(run.eeg)
        rec.raw = raw
        if raw.get_montage() is None:
            set_standard_montage(raw)
        g = gaze_from_raw(raw, screen_size or (1920, 1080))
        if g is not None:
            rec.gaze = g
    ev_frames = []
    for gp in run.gaze:
        g, ev = read_bids_eyetrack(gp, {k: v for k, v in eeg_meta.items()
                                        if k.startswith("Screen")}, screen_size)
        ev_frames.append(ev)
        if rec.gaze is None or "eye1" in gp.name:
            rec.gaze = g
    if ev_frames:
        rec.events = pd.concat([f for f in ev_frames if len(f)] or [rec.events],
                               ignore_index=True)
    if rec.raw is None and rec.gaze is not None:
        t0 = rec.gaze.times[0]
        if t0 != 0:
            rec.gaze = rec.gaze.shifted(-t0)
            rec.events = rec.events.assign(onset=rec.events.onset - t0)
    rec.description = f"BIDS {root.name}: " + " ".join(
        f"{k}-{v}" for k, v in run.entities.items() if v)
    return rec
