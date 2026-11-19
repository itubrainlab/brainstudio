"""Epoching stage: events mapped to named groups → one ``mne.Epochs`` per group.

Epochs are built *without* baseline correction. The ERP baseline is chosen in the
analysis stage, because band-power analysis uses a different baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
import pandas as pd

from . import eeg as E


@dataclass
class EpochSettings:
    """Defaults follow the ERP guide (neuraldatascience.io): −0.1 … 1.0 s, no fixed
    amplitude threshold, AutoReject interpolates bad channels per epoch and drops the
    epochs it cannot repair."""
    tmin: float = -0.1
    tmax: float = 1.0
    reject_uv: float | None = None    # peak-to-peak rejection (EEG)
    autoreject: bool = True           # AutoReject (local): interpolate, then drop
    flat_uv: float | None = None      # reject epochs flatter than this
    drop_blinks: bool = False         # drop epochs overlapping an eye-tracker blink
    delay_ms: float = 0.0             # EEG recording delay: + moves events later

    def describe(self) -> str:
        s = f"{self.tmin:g} … {self.tmax:g} s"
        if self.reject_uv:
            s += f", reject ptp > {self.reject_uv:g} µV"
        if self.flat_uv:
            s += f", flat < {self.flat_uv:g} µV"
        if self.autoreject:
            s += ", AutoReject"
        if self.drop_blinks:
            s += ", no blinks"
        if self.delay_ms:
            s += f", EEG delay {self.delay_ms:+g} ms"
        return s


def overlaps(onsets: np.ndarray, tmin: float, tmax: float,
             intervals: pd.DataFrame | None) -> np.ndarray:
    """True for each onset whose window [onset+tmin, onset+tmax] overlaps an interval."""
    out = np.zeros(len(onsets), bool)
    if intervals is None or not len(intervals):
        return out
    s = intervals["onset"].to_numpy(float)
    e = s + intervals["duration"].to_numpy(float)
    order = np.argsort(s)
    s, e = s[order], e[order]
    emax = np.maximum.accumulate(e)
    a, b = onsets + tmin, onsets + tmax
    k = np.searchsorted(s, b, side="right") - 1  # last interval starting before b
    valid = k >= 0
    out[valid] = emax[k[valid]] > a[valid]
    return out


def build_group_epochs(raw: mne.io.BaseRaw, groups: dict[str, pd.DataFrame],
                       settings: EpochSettings, blinks: pd.DataFrame | None = None
                       ) -> tuple[dict[str, mne.Epochs], pd.DataFrame]:
    """One ``mne.Epochs`` per group (metadata = event rows + ``group`` column).

    Returns the epochs and a summary table per group: selected; dropped before
    epoching because of a blink, the window falling outside the recording or another
    event on the same sample; rejected by amplitude
    (peak-to-peak/flat) or by a BAD annotation; kept; and the channels behind
    amplitude rejections.
    """
    out, rows = {}, []
    for name, table in groups.items():
        tab = table.copy()
        if settings.delay_ms:  # the EEG lags the events: cut epochs later
            tab["onset"] = tab["onset"] + settings.delay_ms / 1000.0
        n_sel = len(tab)
        n_blink = 0
        if settings.drop_blinks and blinks is not None and len(tab):
            hit = overlaps(tab["onset"].to_numpy(float), settings.tmin, settings.tmax, blinks)
            n_blink = int(hit.sum())
            tab = tab[~hit]
        row = {"group": name, "selected": n_sel, "blink": n_blink,
               "outside recording": 0, "same sample": 0,
               "rejected (amplitude)": 0, "rejected (BAD annotation)": 0, "kept": 0,
               "rejected by": ""}
        if len(tab):
            tab = tab.assign(group=name)
            try:
                ep = E.epochs_from_table(raw, tab, settings.tmin, settings.tmax, baseline=None,
                                         reject_uv=settings.reject_uv)
            except ValueError as e:
                ep = None
                row.update(getattr(e, "counts", {"outside recording": len(tab)}))
            if ep is not None and settings.flat_uv:
                ep.drop_bad(flat=dict(eeg=settings.flat_uv * 1e-6), verbose="error")
            if ep is not None:
                row.update(ep.selection_counts)
                why = E.drop_reasons(ep)
                row["rejected (amplitude)"] = why["amplitude"] + why["other"]
                row["rejected (BAD annotation)"] = why["BAD annotation"]
                row["kept"] = len(ep)
                causes = E.rejection_causes(ep)
                row["rejected by"] = ", ".join(f"{c} ({n})" for c, n in list(causes.items())[:6])
                if len(ep):
                    out[name] = ep
        rows.append(row)
    summary = pd.DataFrame(rows)
    if settings.autoreject and out:
        out, summary = apply_autoreject(out, summary)
    return out, summary


def apply_autoreject(epochs: dict[str, mne.Epochs], summary: pd.DataFrame,
                     max_fit: int = 600, seed: int = 42
                     ) -> tuple[dict[str, mne.Epochs], pd.DataFrame]:
    """AutoReject (``n_interpolate=[1, 2, 4]``, ``random_state=42``), as in the ERP guide:
    fitted once on the epochs of every group (at most ``max_fit`` of them), then each
    group is cleaned — bad channels are interpolated within an epoch and the epochs that
    cannot be repaired are dropped. Needs electrode positions and ≥ 10 epochs."""
    import warnings

    summary = summary.copy()
    summary["rejected (AutoReject)"] = 0
    summary["interpolated (AutoReject)"] = 0
    every = list(epochs.values())
    n = sum(len(e) for e in every)
    if not E.has_positions(every[0]):
        summary.attrs["autoreject"] = "not applied: the EEG channels have no positions"
        return epochs, summary
    if n < 10:
        summary.attrs["autoreject"] = "not applied: fewer than 10 epochs"
        return epochs, summary
    from autoreject import AutoReject
    with warnings.catch_warnings(), mne.utils.use_log_level("error"):
        warnings.simplefilter("ignore")
        # fitting needs only the data: one array of every group's epochs (their event
        # codes may clash, which mne.concatenate_epochs refuses)
        sizes = [len(e) for e in every]
        pick = np.arange(n)
        if n > max_fit:
            pick = np.sort(np.random.default_rng(seed).choice(n, max_fit, False))
        starts = np.cumsum([0] + sizes)
        data = np.concatenate([every[k].get_data(item=pick[(pick >= starts[k]) &
                                                         (pick < starts[k + 1])] - starts[k])
                               for k in range(len(every))])
        allep = mne.EpochsArray(data, every[0].info, tmin=every[0].tmin, baseline=None,
                                verbose="error")
        ar = AutoReject(n_interpolate=[1, 2, 4], random_state=seed, n_jobs=1,
                        verbose=False, picks=mne.pick_types(allep.info, eeg=True, eog=False))
        ar.fit(allep)
        out = {}
        for g, ep in epochs.items():
            ep = ep.copy().load_data()
            log = ar.get_reject_log(ep)
            clean = ar.transform(ep)
            i = summary.index[summary["group"] == g][0]
            summary.loc[i, "rejected (AutoReject)"] = int(np.sum(log.bad_epochs))
            summary.loc[i, "interpolated (AutoReject)"] = int(
                (log.labels[~log.bad_epochs] == 2).sum())
            summary.loc[i, "kept"] = len(clean)
            if len(clean):
                out[g] = clean
    summary.attrs["autoreject"] = "applied"
    return out, summary
