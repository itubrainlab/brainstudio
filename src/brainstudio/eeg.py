"""EEG processing helpers built on MNE, including gaze-locked analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd


@dataclass
class FilterParams:
    l_freq: float | None = 1.0
    h_freq: float | None = 40.0
    notch: float | None = None  # e.g. 50 or 60 Hz
    # None (keep as recorded), "average", or channel name(s): "Cz" / "M1, M2"
    reference: str | list[str] | None = None


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
    ref = parse_reference(p.reference, out)
    if ref == "average":
        out.set_eeg_reference("average", projection=False, verbose="error")
    elif ref:
        # re-reference to the (mean of the) given channel(s); they become flat
        out.set_eeg_reference(ref, projection=False, verbose="error")
    return out


def parse_reference(ref, raw: mne.io.BaseRaw):
    """Normalise a reference spec → None, "average" or a list of EEG channel names.

    Accepts None/"keep", "average"/"avg"/"car", a channel name, a comma- or
    space-separated string of names, or a list. Matching is case-insensitive.
    """
    if ref is None:
        return None
    if isinstance(ref, str):
        r = ref.strip()
        if not r or r.lower() in ("keep", "none", "as recorded"):
            return None
        if r.lower() in ("average", "avg", "car", "common average"):
            return "average"
        names = [n for n in r.replace(";", ",").replace(",", " ").split() if n]
    else:
        names = list(ref)
    eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    lower = {n.lower(): n for n in eeg}
    out, missing = [], []
    for n in names:
        (out if n.lower() in lower else missing).append(lower.get(n.lower(), n))
    if missing:
        raise ValueError(f"Reference channel(s) not found among EEG channels: "
                         f"{', '.join(missing)}")
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


# =========================================================================== montage
#: Folders with channel-layout montages: text files listing electrode names in amplifier
#: input order (``<number> <name>`` or just ``<name>`` per line, ``#`` comments). Positions
#: come from an MNE template, set with a ``# positions: <template>`` line (default
#: standard_1005). Files in the user folder show up as built-ins too.
MONTAGE_DIRS = (Path(__file__).parent / "montages", Path.home() / ".brainstudio" / "montages")


def layout_montages() -> dict[str, Path]:
    """Channel-layout montages shipped with BrainStudio or in ~/.brainstudio/montages."""
    out: dict[str, Path] = {}
    for d in MONTAGE_DIRS:
        if d.is_dir():
            out.update({f.stem: f for f in sorted(d.glob("*.txt"))})
    return out


def builtin_montages() -> list[str]:
    """Layout montages first, then MNE's built-in montages."""
    extra = layout_montages()
    return list(extra) + [m for m in mne.channels.get_builtin_montages() if m not in extra]


def read_channel_layout(path) -> tuple[list[str], str]:
    """Electrode names in input order and the MNE template that gives their positions."""
    names, template = [], "standard_1005"
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("#"):
            key, _, val = s.lstrip("# ").partition(":")
            if key.strip().lower() == "positions" and val.strip():
                template = val.strip()
            continue
        if s:
            names.append(s.split()[-1])
    return names, template


def montage_from_layout(names: list[str], template: str = "standard_1005"
                        ) -> mne.channels.DigMontage:
    """DigMontage whose channel order is ``names``, positioned from ``template``."""
    if template == "standard_1005" and "colin27_1005" in mne.channels.get_builtin_montages():
        template = "colin27_1005"  # MNE's new name for the 10-05 template
    pos = mne.channels.make_standard_montage(template).get_positions()
    lookup = {k.upper(): v for k, v in pos["ch_pos"].items()}
    missing = [n for n in names if n.upper() not in lookup]
    if missing:
        raise ValueError(f"No position in {template} for: {', '.join(missing)}")
    return mne.channels.make_dig_montage(
        ch_pos={n: lookup[n.upper()] for n in names}, nasion=pos["nasion"],
        lpa=pos["lpa"], rpa=pos["rpa"], coord_frame=pos["coord_frame"])


def montage_from_source(source: str) -> mne.channels.DigMontage:
    """Montage name (layout montage or ``mne.channels.get_builtin_montages``) or file."""
    layouts = layout_montages()
    if source in layouts:
        return montage_from_layout(*read_channel_layout(layouts[source]))
    p = Path(source)
    if not p.exists():  # a montage name (incl. deprecated aliases MNE still accepts)
        try:
            return mne.channels.make_standard_montage(source)
        except ValueError as e:
            raise ValueError(f"'{source}' is neither a file nor a built-in montage") from e
    if p.suffix.lower() == ".fif":
        return mne.channels.read_dig_fif(p)
    if p.name.endswith("_electrodes.tsv"):  # BIDS electrodes file
        import pandas as pd
        df = pd.read_csv(p, sep="\t", na_values=["n/a"]).dropna(subset=["x", "y", "z"])
        pos = df[["x", "y", "z"]].to_numpy(float)
        if np.nanmax(np.abs(pos)) > 1.0:  # mm / cm → m
            pos = pos / (1000.0 if np.nanmax(np.abs(pos)) > 30 else 100.0)
        return mne.channels.make_dig_montage(dict(zip(df["name"].astype(str), pos)),
                                             coord_frame="head")
    return mne.channels.read_custom_montage(p)


def guess_channel_type(name: str) -> str:
    n = name.upper()
    for key, kind in (("EOG", "eog"), ("ECG", "ecg"), ("EKG", "ecg"), ("EMG", "emg"),
                      ("TRIG", "stim"), ("STI", "stim")):
        if key in n:
            return kind
    return "eeg"


def apply_montage(raw: mne.io.BaseRaw, montage: mne.channels.DigMontage | None,
                  new_names: list[str] | None = None, set_types: bool = True) -> dict:
    """Rename channels (in recording order), set types and apply ``montage`` in place.

    Returns a report: n_renamed, n_positioned, missing (EEG channels without position).
    """
    report = {"n_renamed": 0}
    if new_names:
        mapping = {}
        for old, new in zip(raw.ch_names, new_names):
            new = new.strip()
            if new and new != old:
                mapping[old] = new
        if len(set(mapping.values())) != len(mapping) or \
                set(mapping.values()) & (set(raw.ch_names) - set(mapping)):
            raise ValueError("Channel names must be unique")
        raw.rename_channels(mapping, verbose="error")
        report["n_renamed"] = len(mapping)
    if set_types:
        types = {ch: guess_channel_type(ch) for ch in raw.ch_names}
        changed = {ch: t for ch, t in types.items()
                   if t != "eeg" and raw.get_channel_types([ch])[0] == "eeg"}
        if changed:
            raw.set_channel_types(changed, verbose="error")
    if montage is not None:
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="error")
    eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
    has = [i for i in eeg if np.isfinite(raw.info["chs"][i]["loc"][:3]).all()
           and np.abs(raw.info["chs"][i]["loc"][:3]).sum() > 0]
    report["n_positioned"] = len(has)
    report["missing"] = [raw.ch_names[i] for i in eeg if i not in has]
    return report


# =========================================================================== event-table epochs
def epochs_from_table(raw: mne.io.BaseRaw, table, tmin: float = -0.2, tmax: float = 0.8,
                      baseline=(None, 0), reject_uv: float | None = None,
                      max_epochs: int | None = 2000, seed: int = 0,
                      picks="eeg") -> mne.Epochs:
    """MNE Epochs locked to the rows of an event table, with the table as metadata.

    Event IDs are the table's ``name`` column, so standard MNE selection works:
    ``epochs["fixation"]``, ``epochs['cond == "AI"']`` (metadata query), etc.
    Rows outside the recording, or falling on an already used sample, are
    dropped; ``max_epochs`` randomly subsamples (reproducibly) to bound memory.
    """
    import pandas as pd

    tab = table.reset_index(drop=True).copy()
    sf = raw.info["sfreq"]
    samples = np.round(tab["onset"].to_numpy(float) * sf).astype(int) + raw.first_samp
    lo = raw.first_samp + int(np.ceil(-tmin * sf))
    hi = raw.last_samp - int(np.ceil(tmax * sf))
    ok = (samples >= lo) & (samples <= hi)
    tab, samples = tab[ok], samples[ok]
    _, first = np.unique(samples, return_index=True)  # MNE needs unique samples
    keep = np.sort(first)
    tab, samples = tab.iloc[keep], samples[keep]
    if max_epochs and len(tab) > max_epochs:
        sel = np.sort(np.random.default_rng(seed).choice(len(tab), max_epochs, replace=False))
        tab, samples = tab.iloc[sel], samples[sel]
    if not len(tab):
        raise ValueError("No selected events fall inside the EEG recording")
    names = sorted(tab["name"].astype(str).unique())
    event_id = {n: i + 1 for i, n in enumerate(names)}
    codes = tab["name"].astype(str).map(event_id).to_numpy()
    events = np.column_stack([samples, np.zeros_like(samples), codes]).astype(int)
    picks_idx = mne.pick_types(raw.info, eeg=True, exclude="bads") if picks == "eeg" else picks
    if picks == "eeg" and not len(picks_idx):
        picks_idx = eeg_picks(raw)
    reject = dict(eeg=reject_uv * 1e-6) if reject_uv else None
    meta = tab.reset_index(drop=True)
    meta = meta.astype({c: object for c in meta.columns
                        if isinstance(meta[c].dtype, pd.StringDtype)})
    return mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                      baseline=baseline, picks=picks_idx, reject=reject, preload=True,
                      metadata=meta, verbose="error")


# =========================================================================== contrasts
def contrast_evokeds(epochs_a: mne.Epochs, epochs_b: mne.Epochs,
                     labels=("A", "B")) -> dict[str, mne.Evoked]:
    """Averages per group and their difference (A − B) via ``mne.combine_evoked``."""
    ev_a, ev_b = epochs_a.average(), epochs_b.average()
    ev_a.comment, ev_b.comment = labels
    diff = mne.combine_evoked([ev_a, ev_b], weights=[1, -1])
    diff.comment = f"{labels[0]} − {labels[1]}"
    return {labels[0]: ev_a, labels[1]: ev_b, diff.comment: diff}


def _cluster_masks(clusters, shape) -> list[np.ndarray]:
    """MNE returns clusters as boolean masks *or* index tuples (e.g. ``(slice,)``
    for 1-D data) depending on version and data shape: always return masks."""
    out = []
    for c in clusters:
        if isinstance(c, np.ndarray) and c.dtype == bool and c.shape == shape:
            out.append(c)
        else:
            m = np.zeros(shape, bool)
            m[c] = True
            out.append(m)
    return out


def _flat(data: np.ndarray, axis) -> np.ndarray:
    """True for signals with (numerically) zero variance, e.g. the reference channel."""
    return np.nanstd(data, axis=axis) < 1e-15


def cluster_contrast(epochs_a: mne.Epochs, epochs_b: mne.Epochs, roi=None,
                     n_permutations: int = 1000, tmin=None, tmax=None, seed: int = 0,
                     spatial: bool = False):
    """Cluster-based permutation test between two groups of epochs (MNE).

    ``spatial=False``: ``mne.stats.permutation_cluster_test`` on the ROI average
    (``roi`` = channel names, default all). ``spatial=True``:
    ``mne.stats.spatio_temporal_cluster_test`` over all channels using the
    montage adjacency. Flat channels (e.g. a single-channel reference) are
    excluded. Returns a dict with times, F, clusters (boolean masks of shape
    (n_times,) or (n_times, n_channels)), p_values and the channels used.
    """
    if min(len(epochs_a), len(epochs_b)) < 2:
        raise ValueError(f"A cluster test needs at least 2 epochs per group (got "
                         f"{len(epochs_a)} and {len(epochs_b)}): pick events that occur "
                         "more than once, e.g. fixations within a phase")
    a = epochs_a.copy().crop(tmin, tmax)
    b = epochs_b.copy().crop(tmin, tmax)
    kw = dict(n_permutations=n_permutations, rng=seed, verbose="error")
    if spatial:
        adjacency, ch_names = mne.channels.find_ch_adjacency(a.info, ch_type="eeg")
        a.pick(ch_names)
        b.pick(ch_names)
        Xa = a.get_data(copy=False).transpose(0, 2, 1)  # (epochs, times, channels)
        Xb = b.get_data(copy=False).transpose(0, 2, 1)
        flat = _flat(np.concatenate([Xa, Xb]), axis=(0, 1))
        keep = np.flatnonzero(~flat)
        if len(keep) < 2:
            raise ValueError("Not enough non-flat channels for a spatio-temporal test")
        Xa, Xb = Xa[:, :, keep], Xb[:, :, keep]
        adjacency = adjacency.tocsr()[keep][:, keep]
        ch_names = [ch_names[i] for i in keep]
        F, clusters, p, _ = mne.stats.spatio_temporal_cluster_test(
            [Xa, Xb], adjacency=adjacency, **kw)
        return {"times": a.times, "F": F, "p_values": np.asarray(p),
                "clusters": _cluster_masks(clusters, F.shape), "ch_names": ch_names,
                "excluded": [n for n, f in zip(a.ch_names, flat) if f], "spatial": True}
    roi = [c for c in (roi or a.ch_names) if c in a.ch_names]
    if not roi:
        raise ValueError("None of the ROI channels are in the epochs (renamed or bad?)")
    Xa, Xb = a.get_data(picks=roi).mean(1), b.get_data(picks=roi).mean(1)
    if _flat(np.concatenate([Xa, Xb]), axis=None):
        raise ValueError("The ROI signal is flat (is the ROI only the reference channel?)")
    F, clusters, p, _ = mne.stats.permutation_cluster_test([Xa, Xb], **kw)
    return {"times": a.times, "F": F, "p_values": np.asarray(p),
            "clusters": _cluster_masks(clusters, F.shape), "roi": roi, "spatial": False}


# =========================================================================== ERP measures
def erp_measures(epochs: mne.BaseEpochs, roi: list[str], window: tuple[float, float]
                 ) -> pd.DataFrame:
    """Per-epoch ERP measures on the ROI mean inside ``window`` (µV, s):
    mean amplitude, positive/negative peak amplitude and their latencies.
    Apply the ERP baseline to ``epochs`` first."""
    roi = [c for c in roi if c in epochs.ch_names] or list(epochs.ch_names)
    m = (epochs.times >= window[0]) & (epochs.times <= window[1])
    if not m.any():
        raise ValueError("The measurement window lies outside the epochs")
    x = epochs.get_data(picks=roi).mean(1)[:, m] * 1e6
    t = epochs.times[m]
    return pd.DataFrame({"mean amplitude µV": x.mean(1),
                         "peak + µV": x.max(1), "peak + latency s": t[x.argmax(1)],
                         "peak − µV": x.min(1), "peak − latency s": t[x.argmin(1)]})


def evoked_peak(evoked: mne.Evoked, roi: list[str], window: tuple[float, float],
                polarity: str = "pos") -> tuple[float, float]:
    """Latency (s) and amplitude (µV) of the ROI-mean evoked peak inside ``window``."""
    roi = [c for c in roi if c in evoked.ch_names] or list(evoked.ch_names)
    m = (evoked.times >= window[0]) & (evoked.times <= window[1])
    y = evoked.copy().pick(roi).data.mean(0)[m] * 1e6
    t = evoked.times[m]
    i = int(np.argmax(y) if polarity == "pos" else np.argmin(y) if polarity == "neg"
            else np.argmax(np.abs(y)))
    return float(t[i]), float(y[i])


# =========================================================================== trigger channel
TRIGGER_WORDS = ("trig", "status", "marker", "event", "stim", "sti ")
_UNIT_BACK = {"uV": 1e6, "µV": 1e6, "μV": 1e6, "mV": 1e3, "nV": 1e9, "V": 1.0}


def guess_trigger_channel(raw: mne.io.BaseRaw) -> str | None:
    """A stim channel, else one named like a trigger (TRIG, Status, …), else None."""
    stims = [raw.ch_names[i] for i in mne.pick_types(raw.info, stim=True)]
    if stims:
        return stims[0]
    return next((c for c in raw.ch_names if any(w in c.lower() + " " for w in TRIGGER_WORDS)),
                None)


def trigger_codes(raw: mne.io.BaseRaw, channel: str, mask: int | None = None,
                  mode: str = "codes") -> pd.DataFrame:
    """Events on a trigger channel: onset (s) and code.

    ``mode="codes"``: the channel holds integer codes (EDF/BDF trigger or
    Status channels). The reader's unit scaling is undone and values rounded,
    then ``mne.find_events`` reports every change to a non-zero code (an
    optional bit ``mask`` keeps only some bits). ``mode="threshold"``: an
    analog pulse channel (e.g. photodiode) is binarised half-way between its
    resting level (median) and its pulse extreme, and each pulse onset is code 1.
    """
    if channel not in raw.ch_names:
        raise ValueError(f"No channel named {channel!r}")
    x = raw.get_data(picks=[channel])[0]
    if mode == "threshold":
        base = float(np.median(x))  # pulses are rare: the median is the resting level
        up, down = float(x.max()) - base, base - float(x.min())
        if max(up, down) <= 0:
            raise ValueError(f"{channel} is flat")
        thr = base + up / 2 if up >= down else base - down / 2  # positive or negative pulses
        data = ((x > thr) if up >= down else (x < thr)).astype(float)
    else:
        is_stim = raw.get_channel_types([channel])[0] == "stim"
        unit = getattr(raw, "_orig_units", {}).get(channel, "V")
        data = np.round(x * (1.0 if is_stim else _UNIT_BACK.get(unit, 1.0)))
    info = mne.create_info(["TRIG"], raw.info["sfreq"], "stim")
    trig = mne.io.RawArray(data[None], info, first_samp=raw.first_samp, verbose="error")
    ev = mne.find_events(trig, stim_channel="TRIG", shortest_event=1, consecutive=True,
                         initial_event=True, mask=mask or None, mask_type="and",
                         verbose="error")
    onsets = (ev[:, 0] - raw.first_samp) / raw.info["sfreq"]
    return pd.DataFrame({"onset": onsets, "code": ev[:, 2].astype(int)})


# =========================================================================== ERP components
@dataclass
class Component:
    """An ERP component: time window, channel cluster and polarity."""
    name: str
    tmin: float
    tmax: float
    channels: tuple[str, ...]
    polarity: int  # +1 positive-going (peak = maximum), -1 negative-going


#: Conventional windows and sites (starting points: they depend on paradigm and
#: population). The same set as the lab's Rhino ERP.
COMPONENT_PRESETS = {
    "P1": Component("P1", 0.08, 0.13, ("O1", "O2", "Oz", "PO7", "PO8"), +1),
    "N1": Component("N1", 0.08, 0.15, ("Fz", "FCz", "Cz"), -1),
    "N170": Component("N170", 0.13, 0.20, ("PO7", "PO8", "P7", "P8"), -1),
    "P2": Component("P2", 0.15, 0.25, ("Fz", "FCz", "Cz"), +1),
    "N2": Component("N2", 0.20, 0.35, ("Fz", "FCz", "Cz"), -1),
    "MMN": Component("MMN", 0.10, 0.25, ("Fz", "FCz", "Cz"), -1),
    "P300": Component("P300", 0.25, 0.50, ("Pz", "CPz", "Cz"), +1),
    "N400": Component("N400", 0.30, 0.50, ("Cz", "CPz", "Pz"), -1),
    "LPP": Component("LPP", 0.40, 0.80, ("Pz", "CPz", "Cz"), +1),
}


def component_analysis(groups: dict[str, mne.BaseEpochs], comp: Component
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure ``comp`` in every group (epochs already baseline-corrected).

    Per epoch: mean amplitude in the window on the channel-cluster mean (µV).
    Per group: n, window mean ± SEM, and peak amplitude/latency of the average
    ERP (``Evoked.get_peak`` on the cluster mean, in the component's polarity).
    With two groups the window means are compared with a two-sided Welch t-test
    (plus Mann-Whitney U and Hedges' g) and the peak latencies are subtracted.

    Returns (summary table, per-epoch table).
    """
    from scipy import stats as sst

    from . import stats as ST

    if comp.tmin >= comp.tmax:
        raise ValueError("The component window must start before it ends")
    first = next(iter(groups.values()))
    chans = [c for c in comp.channels if c in first.ch_names]
    if not chans:
        raise ValueError(f"None of {', '.join(comp.channels)} are among the epoch channels "
                         "(renamed, or marked bad?)")
    rows, per = [], []
    vals = {}
    for g, ep in groups.items():
        m = (ep.times >= comp.tmin) & (ep.times <= comp.tmax)
        if not m.any():
            raise ValueError("The component window lies outside the epochs")
        x = ep.get_data(picks=chans).mean(1) * 1e6
        amps = x[:, m].mean(1)
        vals[g] = amps
        roi = mne.channels.combine_channels(
            ep.average(), {"roi": [ep.ch_names.index(c) for c in chans]}, method="mean",
            verbose="error")
        _, lat, amp = roi.get_peak(tmin=comp.tmin, tmax=comp.tmax,
                                   mode="neg" if comp.polarity < 0 else "pos",
                                   return_amplitude=True, strict=False)
        n = len(amps)
        rows.append({"group": g, "n": n, "window mean µV": float(amps.mean()),
                     "sem": float(amps.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                     "peak µV": float(amp) * 1e6, "peak latency ms": float(lat) * 1000})
        per.append(pd.DataFrame({"group": g, "onset": ep.metadata["onset"].to_numpy(float)
                                 if ep.metadata is not None and "onset" in ep.metadata
                                 else np.nan, "window mean µV": amps}))
    summary = pd.DataFrame(rows)
    if len(groups) == 2:
        (ga, a), (gb, b) = vals.items()
        if min(len(a), len(b)) >= 2:
            w = sst.ttest_ind(a, b, equal_var=False)
            u = sst.mannwhitneyu(a, b, alternative="two-sided")
            la, lb = summary["peak latency ms"]
            pa, pb = summary["peak µV"]
            summary = pd.concat([summary, pd.DataFrame([{
                "group": f"{ga} − {gb}", "n": np.nan,
                "window mean µV": float(a.mean() - b.mean()), "sem": np.nan,
                "peak µV": pa - pb, "peak latency ms": la - lb,
                "Welch t": float(w.statistic), "df": float(w.df), "p": float(w.pvalue),
                "p (Mann-Whitney)": float(u.pvalue), "hedges g": ST.hedges_g(a, b)}])],
                ignore_index=True)
    elif len(groups) > 2:
        f = sst.f_oneway(*[v for v in vals.values() if len(v) >= 2])
        summary = pd.concat([summary, pd.DataFrame([{
            "group": "all groups", "ANOVA F": float(f.statistic), "p": float(f.pvalue)}])],
            ignore_index=True)
    summary.insert(0, "component", f"{comp.name} {comp.tmin * 1000:.0f}–{comp.tmax * 1000:.0f} ms "
                   f"{'+' if comp.polarity > 0 else '−'} @ {'+'.join(chans)}")
    return summary, pd.concat(per, ignore_index=True)


def rejection_causes(epochs: mne.BaseEpochs) -> dict[str, int]:
    """How many epochs each channel caused to be rejected, worst first."""
    counts: dict[str, int] = {}
    for reasons in epochs.drop_log:
        for r in reasons:
            if r in epochs.ch_names:
                counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
