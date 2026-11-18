"""Band power, baselines and cognitive-state indices (MNE spectral estimates).

Band power per epoch comes from ``Epochs.compute_psd`` integrated over each band
(µV²). The **baseline** is the same measure in a window *after a baseline event*
(e.g. 5–55 s after ``BASELINE_START``), cut into segments as long as the analysis
window so both have the same spectral resolution. Results can be expressed
relative to the baseline (dB, % change, ratio, z) using either the mean over all
baseline segments or the baseline event that most recently precedes each epoch.

Indices (computed per epoch from absolute band power):

* engagement  β / (α + θ)  over Cz, Pz, P3, P4 (Pope, Bogart & Bartolome, 1995)
* workload    frontal-midline θ / parietal α (Gevins & Smith, 2003)
* θ/α         theta / alpha over all channels (load-related ratio)
* θ/β         frontal theta / beta (attentional control)
* FAA         ln α(F4) − ln α(F3) (frontal alpha asymmetry, Davidson)
* α ERD       % change of parietal alpha vs baseline (alpha suppression)
* FMθ ERS     % change of frontal-midline theta vs baseline
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import pandas as pd

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


# =========================================================================== power
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
                             tmax=tmax, picks="eeg", exclude=(), verbose="error", **kw)
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
                    baseline=None, picks=picks, preload=True, reject_by_annotation=True,
                    verbose="error")
    owner = owner[[i for i, d in enumerate(ep.drop_log) if not d]]
    if not len(ep):
        raise ValueError("All baseline segments overlap BAD annotations")
    return ep, owner


def baseline_band_power(raw: mne.io.BaseRaw, base: BandBaseline, settings: BandSettings,
                        ch_names: list[str], seg_len: float, ref=None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Band power of baseline segments: (n_segments, n_ch, n_bands) and segment owners.

    ``ref`` (see :func:`brainstudio.eeg.resolve_erp_reference`) re-references the
    segments the same way as the epochs they are compared with."""
    from .eeg import reference_epochs
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


# =========================================================================== ROIs
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


# =========================================================================== indices
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


# =========================================================================== time course
def morlet_band(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """(n, n_ch, n_times) V → Morlet power averaged over the band's frequencies and the
    channels: (n, n_times) µV²."""
    lo, hi = band
    freqs = np.arange(lo, hi + 1e-9, max((hi - lo) / 6, 0.5))
    if len(freqs) < 2:
        freqs = np.array([lo, hi])
    n_cycles = np.maximum(freqs / 2.0, 3.0)
    p = mne.time_frequency.tfr_array_morlet(data, sfreq, freqs, n_cycles=n_cycles,
                                            output="power", verbose="error")
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
    from .eeg import reference_epochs
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


def baseline_timecourse_level(raw, base, band, picks, seg_len=4.0, ref=None, ch_names=None
                              ) -> tuple[float, float]:
    """Pooled level (all baseline segments); see :func:`baseline_timecourse_levels`."""
    b = BandBaseline(base.onsets, base.t0, base.t1, base.mode, base.label, None)
    return baseline_timecourse_levels(raw, b, band, picks, seg_len, ref, ch_names)[""]


# =========================================================================== gaze metrics
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
