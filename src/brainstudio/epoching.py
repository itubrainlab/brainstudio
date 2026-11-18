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
    tmin: float = -0.2
    tmax: float = 0.8
    reject_uv: float | None = 150.0   # peak-to-peak rejection (EEG)
    flat_uv: float | None = None      # reject epochs flatter than this
    drop_blinks: bool = False         # drop epochs overlapping an eye-tracker blink
    max_epochs: int | None = 2000     # per group, reproducible random subsample
    delay_ms: float = 0.0             # EEG recording delay: + moves events later

    def describe(self) -> str:
        s = f"{self.tmin:g} … {self.tmax:g} s"
        if self.reject_uv:
            s += f", reject ptp > {self.reject_uv:g} µV"
        if self.flat_uv:
            s += f", flat < {self.flat_uv:g} µV"
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

    Returns the epochs and a summary table (selected / outside / blink / rejected / kept).
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
        row = {"group": name, "selected": n_sel, "blink": n_blink, "outside/dup": 0,
               "rejected": 0, "kept": 0, "rejected by": ""}
        if len(tab):
            tab = tab.assign(group=name)
            try:
                ep = E.epochs_from_table(raw, tab, settings.tmin, settings.tmax, baseline=None,
                                         reject_uv=settings.reject_uv,
                                         max_epochs=settings.max_epochs)
            except ValueError:
                ep = None
            if ep is not None and settings.flat_uv:
                ep.drop_bad(flat=dict(eeg=settings.flat_uv * 1e-6), verbose="error")
            if ep is not None:
                n_in = len(ep.drop_log)
                row["outside/dup"] = len(tab) - n_in
                row["rejected"] = n_in - len(ep)
                row["kept"] = len(ep)
                causes = E.rejection_causes(ep)
                row["rejected by"] = ", ".join(f"{c} ({n})" for c, n in list(causes.items())[:6])
                if len(ep):
                    out[name] = ep
            else:
                row["outside/dup"] = len(tab)
        rows.append(row)
    return out, pd.DataFrame(rows)
