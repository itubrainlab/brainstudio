"""Shared state of the EEG pipeline: preprocessing → epoching → analysis.

Each stage reads the output of the previous one. When a stage produces new
output, every later stage that already has results is marked *stale* until it is
run again; the tabs show this and offer a re-run.
"""

from __future__ import annotations

import mne
import pandas as pd
from PySide6 import QtCore

from .. import events as EV
from .. import preprocessing as P
from ..epoching import EpochSettings

STAGES = ("preprocessing", "epoching", "analysis")


class EEGPipeline(QtCore.QObject):
    #: emitted with the name of what changed: recording, events, preprocessing,
    #: epoching, analysis
    changed = QtCore.Signal(str)
    stateChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rec = None
        self.gaze_events = None
        self.event_table = pd.DataFrame(columns=EV.BASE_COLUMNS)
        self.reset()

    def reset(self):
        self.params = P.PreprocParams(reference=None)
        self.clean: mne.io.BaseRaw | None = None
        self.bad_stats: pd.DataFrame | None = None
        self.ica = None
        self.ica_fit_raw = None
        self.ica_scores: pd.DataFrame | None = None
        self.ica_key = None
        self.psd_before = None
        self.psd_after = None
        self.epoch_settings = EpochSettings()
        self.epochs: dict[str, mne.Epochs] = {}
        self.epoch_summary: pd.DataFrame | None = None
        self.group_desc: dict[str, str] = {}
        self.state = {s: "pending" for s in STAGES}

    # ------------------------------------------------------------------ inputs
    @property
    def raw(self) -> mne.io.BaseRaw | None:
        return self.rec.raw if self.rec is not None else None

    def set_recording(self, rec):
        self.rec = rec
        self.reset()
        self.changed.emit("recording")
        self.stateChanged.emit()

    def set_gaze_events(self, ev):
        self.gaze_events = ev

    def set_event_table(self, table: pd.DataFrame):
        self.event_table = table
        self.changed.emit("events")
        self._stale_from("epoching")

    # ------------------------------------------------------------------ outputs
    def set_clean(self, clean, params: P.PreprocParams, psd_before=None, psd_after=None):
        self.clean = clean
        self.params = params
        self.psd_before, self.psd_after = psd_before, psd_after
        self.state["preprocessing"] = "ok"
        self._stale_from("epoching")
        self.changed.emit("preprocessing")

    def set_epochs(self, epochs, summary, settings, group_desc):
        self.epochs, self.epoch_summary = epochs, summary
        self.epoch_settings, self.group_desc = settings, group_desc
        self.state["epoching"] = "ok"
        self._stale_from("analysis")
        self.changed.emit("epoching")

    def set_analysis_done(self):
        self.state["analysis"] = "ok"
        self.stateChanged.emit()

    def invalidate_ica(self):
        self.ica = self.ica_fit_raw = self.ica_scores = self.ica_key = None

    def mark_stale(self, stage: str):
        """Settings of ``stage`` changed: it and every later stage are out of date."""
        self._stale_from(stage)

    def _stale_from(self, stage: str):
        for s in STAGES[STAGES.index(stage):]:
            if self.state[s] == "ok":
                self.state[s] = "stale"
        self.stateChanged.emit()

    # ------------------------------------------------------------------ helpers
    def eeg_input(self) -> mne.io.BaseRaw | None:
        """Data for epoching: the preprocessed recording (raw if not run yet)."""
        return self.clean if self.clean is not None else self.raw
