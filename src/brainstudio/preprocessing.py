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
    l_freq: float | None = 1.0    # band-pass before measuring, so drifts and line
    h_freq: float | None = 40.0   # noise don't count as "noisy"
    notch: float | None = None


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
    n_components: int = 20
    method: str = "picard"        # picard | fastica | infomax
    l_freq: float = 1.0           # ICA is fitted on ≥ 1 Hz high-passed data
    reject_uv: float | None = 500.0  # skip 2 s segments above this ptp when fitting
    target_sfreq: float = 128.0   # decimate the fitting data to about this rate
    seed: int = 97


#: below this high-pass cut-off an FIR filter gets very long: use a zero-phase IIR
IIR_BELOW_HZ = 0.5
IIR_PARAMS = dict(order=4, ftype="butter", output="sos")


def highpass_method(l_freq) -> dict:
    """Filter arguments for a high-pass: FIR, or zero-phase Butterworth below 0.5 Hz."""
    if l_freq and l_freq < IIR_BELOW_HZ:
        return dict(method="iir", iir_params=IIR_PARAMS)
    return {}


def _filtered(raw: mne.io.BaseRaw, l_freq, h_freq, notch, picks=None,
              tmin: float | None = None) -> mne.io.BaseRaw:
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
        out.filter(l_freq, None, picks=fp, verbose="error", **highpass_method(l_freq))
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
    fit_raw = _filtered(raw, max(p.l_freq, 0.1), h_freq, notch, picks=chans, tmin=tmin)
    fit_raw.recording_t0 = _start_sample(raw, tmin) / raw.info["sfreq"]
    decim = max(1, int(fit_raw.info["sfreq"] // p.target_sfreq))
    if decim > 1:
        fit_raw.resample(fit_raw.info["sfreq"] / decim, verbose="error")
    n = int(min(p.n_components, len(chans) - 1))
    method = p.method
    if method == "picard":
        try:
            import picard  # noqa: F401
        except ImportError:
            method = "infomax"
    ica = mne.preprocessing.ICA(n_components=n, method=method, rng=p.seed,
                                max_iter="auto",
                                fit_params=dict(ortho=False, extended=True)
                                if method == "picard" else None)
    ica.fit(fit_raw, reject=dict(eeg=p.reject_uv * 1e-6) if p.reject_uv else None,
            tstep=2.0, verbose="error")
    return ica, fit_raw


FRONTAL = ("Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4")


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
                           z_thresh: float = 3.0) -> pd.DataFrame:
    """Rank ICA components as blink candidates.

    * ``eog_score``: |correlation| with EOG channels via ``ICA.find_bads_eog``.
      Without EOG channels, frontal electrodes (Fp1/Fp2/…) act as proxies.
    * ``blink_t``: how strongly the component's blink-locked average (eye-tracker
      blink onsets, −0.2…0.5 s) stands out: max |mean| / SEM, robust to jitter.
    Components whose scores are z-outliers (> ``z_thresh``) are suggested.
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
        eog = [lower[c.lower()] for c in FRONTAL if c.lower() in lower][:2]
        proxy = True
    flagged = set()
    if eog:
        idx, scores = ica.find_bads_eog(fit_raw, ch_name=eog, threshold=z_thresh,
                                        measure="zscore", verbose="error")
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
    df["suggested"] = df["component"].isin(flagged)
    df.attrs["eog_channels"] = eog
    df.attrs["proxy"] = proxy
    return df


# =========================================================================== pipeline
@dataclass
class PreprocParams:
    bads: list[str] = field(default_factory=list)
    l_freq: float | None = 1.0
    h_freq: float | None = 40.0
    notch: float | None = None
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
