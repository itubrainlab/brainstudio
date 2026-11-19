"""Preprocessing stage (MNE): bad-channel detection, filtering, ICA blink removal.

The stage is described by :class:`PreprocParams` and applied with
:func:`run_preprocessing`, which returns a new Raw and never modifies its input:

1. mark bad channels (e.g. from :func:`detect_bad_channels`, peak-to-peak based)
2. band-pass and notch filter
3. remove ICA components (blinks), fitted with :func:`fit_ica`
4. optionally interpolate bad channels (needs electrode positions)
5. re-reference (bad channels are left out of an average reference)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import pandas as pd

from . import eeg as E


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
    return idx if len(idx) else E.eeg_picks(raw)


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
    fit_raw = _filtered(raw, max(p.l_freq, 0.1), h, notch, picks=chans, tmin=tmin)
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
    if p.clean == "autoreject" and len(seg) >= 20 and E.has_positions(seg):
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
    if not E.has_positions(raw.copy().pick(chans)):
        raise ValueError("the EEG channels have no electrode positions (load a montage)")
    nyq = raw.info["sfreq"] / 2
    lab_raw = _filtered(raw, 1.0, min(100.0, nyq - 1), notch, picks=chans, tmin=tmin)
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


def blink_regressor(times: np.ndarray, blinks: pd.DataFrame, pad: float = 0.05) -> np.ndarray:
    """1 inside eye-tracker blinks (padded), 0 elsewhere, on the given time grid."""
    r = np.zeros(len(times))
    if blinks is None or not len(blinks):
        return r
    on = blinks["onset"].to_numpy(float) - pad
    off = on + blinks["duration"].to_numpy(float) + 2 * pad
    i0 = np.searchsorted(times, on)
    i1 = np.searchsorted(times, off)
    for a, b in zip(i0, i1):
        r[a:b] = 1.0
    return r


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
    out = _filtered(raw, p.l_freq, p.h_freq, p.notch)
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
    ref = E.parse_reference(p.reference, out)
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


_filtered = filtered_copy  # older name
