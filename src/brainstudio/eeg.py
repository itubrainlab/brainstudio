"""EEG processing helpers built on MNE, including gaze-locked analyses."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd


@dataclass
class FilterParams:
    l_freq: float | None = 1.0
    h_freq: float | None = 40.0
    notch: float | None = None  # e.g. 50 or 60 Hz
    reference: str | None = None  # None (keep), "average", or a channel name


def eeg_picks(raw: mne.io.BaseRaw) -> np.ndarray:
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True,
                           misc=False, stim=False, exclude=[])
    if not len(picks):  # e.g. unknown channel types from XDF
        picks = mne.pick_types(raw.info, misc=True, exclude=[])
    return picks


def preprocess(raw: mne.io.BaseRaw, p: FilterParams) -> mne.io.BaseRaw:
    """Return a filtered copy of ``raw`` (original is left untouched)."""
    out = raw.copy().load_data()
    picks = mne.pick_types(out.info, eeg=True, eog=True, ecg=True, emg=True, exclude=[])
    if not len(picks):
        return out
    nyq = out.info["sfreq"] / 2
    h = p.h_freq if p.h_freq and p.h_freq < nyq else None
    if p.l_freq or h:
        out.filter(p.l_freq or None, h, picks=picks, verbose="error")
    if p.notch and p.notch < nyq:
        freqs = np.arange(p.notch, nyq, p.notch)
        out.notch_filter(freqs, picks=picks, verbose="error")
    if p.reference == "average":
        out.set_eeg_reference("average", projection=False, verbose="error")
    elif p.reference:
        out.set_eeg_reference([p.reference], verbose="error")
    return out


def compute_psd(raw: mne.io.BaseRaw, fmax: float = 60.0, tmin=None, tmax=None):
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if not len(picks):
        picks = eeg_picks(raw)
    fmax = min(fmax, raw.info["sfreq"] / 2)
    return raw.compute_psd(fmax=fmax, picks=picks, tmin=tmin, tmax=tmax, verbose="error")


def events_to_mne(onsets: np.ndarray, raw: mne.io.BaseRaw, code: int = 1) -> np.ndarray:
    """Convert onsets on the recording time base to an MNE events array."""
    samples = np.round(np.asarray(onsets) * raw.info["sfreq"]).astype(int) + raw.first_samp
    ok = (samples >= raw.first_samp) & (samples <= raw.last_samp)
    samples = np.unique(samples[ok])
    return np.column_stack([samples, np.zeros_like(samples), np.full_like(samples, code)])


def fixation_related_epochs(raw: mne.io.BaseRaw, fixations: pd.DataFrame,
                            tmin: float = -0.2, tmax: float = 0.6,
                            min_duration: float = 0.0,
                            baseline: tuple | None = (None, 0),
                            reject_uv: float | None = None) -> mne.Epochs:
    """Epoch EEG on fixation onsets (fixation-related potentials, FRPs)."""
    fx = fixations[fixations["duration"] >= min_duration]
    events = events_to_mne(fx["onset"].to_numpy(), raw)
    if not len(events):
        raise ValueError("No fixations fall inside the EEG recording")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if not len(picks):
        picks = eeg_picks(raw)
    reject = dict(eeg=reject_uv * 1e-6) if reject_uv else None
    return mne.Epochs(raw, events, event_id={"fixation": 1}, tmin=tmin, tmax=tmax,
                      baseline=baseline, picks=picks, reject=reject, preload=True,
                      verbose="error")


def event_related_epochs(raw: mne.io.BaseRaw, events_df: pd.DataFrame, description: str,
                         tmin: float = -0.2, tmax: float = 0.8,
                         baseline: tuple | None = (None, 0)) -> mne.Epochs:
    """Epoch EEG on markers whose description equals ``description``."""
    desc = events_df["description"].astype(str)
    # exact match, or structured markers ("NAME|key=value|...") matched by NAME
    sel = events_df[(desc == description) | (desc.str.split("|").str[0] == description)]
    events = events_to_mne(sel["onset"].to_numpy(), raw)
    if not len(events):
        raise ValueError(f"No '{description}' events inside the EEG recording")
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if not len(picks):
        picks = eeg_picks(raw)
    return mne.Epochs(raw, events, event_id={description: 1}, tmin=tmin, tmax=tmax,
                      baseline=baseline, picks=picks, preload=True, verbose="error")


def add_gaze_annotations(raw: mne.io.BaseRaw, ev, kinds=("blinks", "saccades")) -> None:
    """Annotate blinks/saccades/fixations on ``raw`` (e.g. to reject segments)."""
    onsets, durs, descs = [], [], []
    label = {"blinks": "BAD_blink", "saccades": "saccade", "fixations": "fixation"}
    for k in kinds:
        df = getattr(ev, k)
        onsets += list(df["onset"])
        durs += list(df["duration"])
        descs += [label[k]] * len(df)
    if not onsets:
        return
    ann = mne.Annotations(np.asarray(onsets) + raw.first_time, durs, descs,
                          orig_time=raw.annotations.orig_time)
    if raw.annotations.orig_time is None:
        ann = mne.Annotations(onsets, durs, descs)
    raw.set_annotations(raw.annotations + ann)


def set_standard_montage(raw: mne.io.BaseRaw) -> None:
    """Apply a standard 10-05 montage to channels whose names match (in place)."""
    available = mne.channels.get_builtin_montages()
    name = "colin27_1005" if "colin27_1005" in available else "standard_1005"
    try:
        raw.set_montage(name, on_missing="ignore", match_case=False, verbose="error")
    except Exception:
        pass
