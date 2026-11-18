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


# --------------------------------------------------------------------------- metadata
#: eeg.json fields shown in the info panel (and kept in Recording.metadata)
EEG_SIDECAR_KEYS = (
    "TaskName", "TaskDescription", "PowerLineFrequency", "EEGReference", "EEGGround",
    "EEGPlacementScheme", "CapManufacturer", "CapManufacturersModelName", "Manufacturer",
    "ManufacturersModelName", "SoftwareVersions", "SoftwareFilters", "HardwareFilters",
    "RecordingDuration", "RecordingType", "InstitutionName",
    "InstitutionalDepartmentName", "RecordingTiming")


def _bids_file(eeg: Path, suffix: str) -> Path | None:
    """Sidecar of an EEG file: same entities, another suffix (inherits up the tree)."""
    stem = eeg.name.rsplit("_", 1)[0]
    cand = eeg.with_name(f"{stem}_{suffix}")
    if cand.exists():
        return cand
    # BIDS inheritance: drop entities from the right (run, acq, task…)
    parts = stem.split("_")
    for k in range(len(parts) - 1, 0, -1):
        c = eeg.with_name("_".join(parts[:k]) + f"_{suffix}")
        if c.exists():
            return c
    return None


def _electrodes_file(eeg: Path) -> Path | None:
    """The subject's (session's) ``*_electrodes.tsv``: usually per subject/session with a
    ``space-`` entity (e.g. ``sub-01_space-CapTrak_electrodes.tsv``), not per run."""
    exact = _bids_file(eeg, "electrodes.tsv")
    if exact is not None:
        return exact
    ent = get_entities_from_fname(eeg.name)
    prefix = f"sub-{ent['subject']}" + (f"_ses-{ent['session']}" if ent.get("session") else "")
    found = sorted(eeg.parent.glob(f"{prefix}*_electrodes.tsv"))
    return found[0] if found else None


def channel_status(eeg: Path) -> pd.DataFrame:
    """``channels.tsv``: name, type, units, status, status_description (as text)."""
    f = _bids_file(eeg, "channels.tsv")
    if f is None:
        return pd.DataFrame(columns=["name", "type", "status", "status_description"])
    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
    for c in ("status", "status_description", "type"):
        if c not in df:
            df[c] = ""
    return df


def bids_metadata(root: str | Path, run: BidsRun) -> dict:
    """Dataset, participant, recording sidecar and channel information of a run."""
    root = Path(root)
    meta: dict = {"dataset": {}, "participant": {}, "eeg": {}, "bad_channels": {},
                  "montage_source": "", "electrodes_file": ""}
    desc = dataset_description(root)
    meta["dataset"] = {k: desc[k] for k in ("Name", "BIDSVersion", "Authors", "License",
                                            "DatasetDOI", "HowToAcknowledge")
                       if desc.get(k) not in (None, "", "n/a", ["n/a"], [])}
    sub = run.entities.get("subject")
    part = root / "participants.tsv"
    if sub and part.exists():
        df = pd.read_csv(part, sep="\t", dtype=str, keep_default_na=False)
        row = df[df.get("participant_id", pd.Series(dtype=str)) == f"sub-{sub}"]
        if len(row):
            meta["participant"] = {k: v for k, v in row.iloc[0].items()
                                   if k != "participant_id" and v not in ("", "n/a")}
    if run.eeg is not None:
        side = _sidecar(run.eeg)
        meta["eeg"] = {k: side[k] for k in EEG_SIDECAR_KEYS if k in side and side[k] != "n/a"}
        ch = channel_status(run.eeg)
        bad = ch[ch["status"].str.lower() == "bad"]
        meta["bad_channels"] = {r.name: (r.status_description if r.status_description
                                         not in ("", "n/a") else "status bad in channels.tsv")
                                for r in bad.itertuples()}
        el = _electrodes_file(run.eeg)
        meta["electrodes_file"] = el.name if el is not None else ""
    return meta


def describe_metadata(meta: dict) -> str:
    """Readable summary for the info panel."""
    if not meta:
        return ""
    out = []
    d = meta.get("dataset", {})
    if d:
        line = d.get("Name", "")
        if d.get("Authors"):
            a = d["Authors"]
            line += f" — {', '.join(a) if isinstance(a, list) else a}"
        if d.get("License"):
            line += f" — licence {d['License']}"
        if d.get("DatasetDOI"):
            line += f" — {d['DatasetDOI']}"
        out.append("Dataset: " + line.strip(" —"))
    if meta.get("participant"):
        out.append("Participant: " + ", ".join(f"{k} {v}" for k, v in meta["participant"].items()))
    e = meta.get("eeg", {})
    dev = " ".join(str(e[k]) for k in ("Manufacturer", "ManufacturersModelName") if k in e)
    cap = " ".join(str(e[k]) for k in ("CapManufacturer", "CapManufacturersModelName") if k in e)
    if dev or cap:
        out.append("Device: " + dev + (f"; cap: {cap}" if cap and cap != dev else ""))
    ref = [f"reference {e['EEGReference']}" if "EEGReference" in e else "",
           f"ground {e['EEGGround']}" if "EEGGround" in e else "",
           f"placement {e['EEGPlacementScheme']}" if "EEGPlacementScheme" in e else "",
           f"power line {e['PowerLineFrequency']} Hz" if "PowerLineFrequency" in e else ""]
    if any(ref):
        out.append("Recording: " + ", ".join(r for r in ref if r))
    sf = e.get("SoftwareFilters")
    if isinstance(sf, dict) and sf:
        parts = []
        for name, f in sf.items():
            if isinstance(f, dict):
                cut = f.get("CutoffFrequency", f.get("Frequencies", ""))
                parts.append(f"{name} {cut} Hz" + (" (causal)" if f.get("Causal") else ""))
            else:
                parts.append(f"{name}: {f}")
        out.append("Online filters already applied: " + "; ".join(parts))
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
        meta = bids_metadata(root, run)
        try:
            raw = read_raw_bids(bp, verbose="error")
            fallback = False
        except Exception:  # fall back to plain MNE reader if sidecars are broken
            raw = mne.io.read_raw(run.eeg, verbose="error")
            fallback = True
        raw.load_data(verbose="error")
        if fallback:  # apply channels.tsv ourselves: types and bad channels
            _apply_channels_tsv(raw, run.eeg)
            pl = _sidecar(run.eeg).get("PowerLineFrequency")
            if isinstance(pl, (int, float)):
                with raw.info._unlock():
                    raw.info["line_freq"] = float(pl)
        eeg_meta = _sidecar(run.eeg)
        rec.raw = raw
        ev_file = _bids_file(run.eeg, "events.tsv")
        if ev_file is not None:  # every column of events.tsv, not just trial_type
            tsv = events_from_tsv(ev_file)
            if len(tsv):
                from ..model import is_data_annotation
                a = raw.annotations
                keep = is_data_annotation(pd.Series(a.description)).to_numpy()
                raw.set_annotations(a[np.flatnonzero(keep)] if keep.any() else
                                    mne.Annotations([], [], [], orig_time=a.orig_time))
                rec.events = pd.concat([rec.events, tsv], ignore_index=True)
        if raw.get_montage() is not None and meta["electrodes_file"]:
            meta["montage_source"] = f"{meta['electrodes_file']} (from the dataset)"
        elif raw.get_montage() is None or not meta["electrodes_file"]:
            set_standard_montage(raw)
            n = sum(1 for ch in raw.info["chs"] if ch["kind"] == 2 and
                    np.isfinite(ch["loc"][:3]).all() and np.abs(ch["loc"][:3]).sum() > 0)
            meta["montage_source"] = (f"standard 10-05 template by channel name ({n} EEG "
                                      "channels)" if n else "none (names do not match the "
                                      "10-05 template: use Montage)")
        rec.metadata = meta
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
    if rec.metadata:
        rec.description += "\n" + describe_metadata(rec.metadata)
    return rec


#: events.tsv columns that are not attributes of an event
_EVENT_SKIP = {"onset", "duration", "trial_type", "sample"}


def events_from_tsv(path: Path) -> pd.DataFrame:
    """All rows of a BIDS ``events.tsv`` as events, keeping every column.

    The name is ``trial_type`` (else ``value``, else "event"); every other column
    becomes an attribute (``NAME|column=value|…``), so it can be filtered on.
    MNE-BIDS keeps only ``trial_type`` when it turns events into annotations.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "onset" not in df:
        return pd.DataFrame(columns=["onset", "duration", "description"])
    df = df[df["onset"].str.lower() != "n/a"]
    onset = pd.to_numeric(df["onset"], errors="coerce")
    dur = pd.to_numeric(df.get("duration", pd.Series("0", index=df.index)),
                        errors="coerce").fillna(0.0)
    if "trial_type" in df:
        names = df["trial_type"]
    elif "value" in df:
        names = df["value"]
    else:
        names = pd.Series("event", index=df.index)
    attrs = [c for c in df.columns if c not in _EVENT_SKIP and not
             (c == "value" and "trial_type" not in df)]

    def safe(v: str) -> str:
        return str(v).replace("|", "/").replace("=", ":").strip()

    desc = []
    for i in df.index:
        name = safe(names[i]) if names[i] not in ("", "n/a") else "event"
        parts = [f"{c}={safe(df.at[i, c])}" for c in attrs if df.at[i, c] not in ("", "n/a")]
        desc.append("|".join([name] + parts))
    out = pd.DataFrame({"onset": onset.to_numpy(float), "duration": dur.to_numpy(float),
                        "description": desc})
    return out[np.isfinite(out["onset"])].reset_index(drop=True)


def _apply_channels_tsv(raw: mne.io.BaseRaw, eeg: Path) -> None:
    """Channel types and bad channels from channels.tsv (when MNE-BIDS could not)."""
    ch = channel_status(eeg)
    if not len(ch):
        return
    bids_to_mne = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "MISC": "misc",
                   "TRIG": "stim", "STIM": "stim", "RESP": "resp", "GSR": "gsr",
                   "TEMP": "temperature", "PPG": "bio", "HEOG": "eog", "VEOG": "eog"}
    types = {r.name: bids_to_mne[r.type.upper()] for r in ch.itertuples()
             if r.name in raw.ch_names and r.type.upper() in bids_to_mne}
    if types:
        raw.set_channel_types(types, on_unit_change="ignore", verbose="error")
    raw.info["bads"] = [r.name for r in ch.itertuples()
                        if r.status.lower() == "bad" and r.name in raw.ch_names]
