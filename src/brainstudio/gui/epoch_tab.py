"""Pipeline stage 2: map events into named groups and cut epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import workflow as W
from ..epoching import EpochSettings, build_group_epochs
from .eventselect import EventSelector
from .theme import THEME
from .widgets import StageHeader, _spin, _themed_plot, fill_table, hint


MAX_EPOCH_S = 3600.0
#: ask before building epochs larger than this in memory
MEMORY_WARN_BYTES = 2e9


def epochs_memory(raw, groups: dict, settings) -> float:
    """Bytes the epochs will take: events × EEG channels × samples × 8."""
    import mne
    n_ch = len(mne.pick_types(raw.info, eeg=True, exclude="bads")) or len(raw.ch_names)
    n_samp = int((settings.tmax - settings.tmin) * raw.info["sfreq"]) + 1
    n_ev = sum(min(len(t), settings.max_epochs or len(t)) for t in groups.values())
    return float(n_ev) * n_ch * n_samp * 8


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


class EpochTab(QtWidgets.QWidget):
    def __init__(self, pipeline, worker, parent=None):
        super().__init__(parent)
        self.pl, self.worker = pipeline, worker
        outer = QtWidgets.QVBoxLayout(self)
        self.header = StageHeader("② Epoching", "Build epochs ▶")
        self.header.run.clicked.connect(lambda: self.run())
        outer.addWidget(self.header)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)

        # ---------------------------------------------------------------- groups
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Event groups")
        bv = QtWidgets.QVBoxLayout(box)
        self.groups = QtWidgets.QListWidget()
        self.groups.setMaximumHeight(130)
        self.groups.currentRowChanged.connect(self._select_group)
        bv.addWidget(self.groups)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("+ Group", self.add_group), ("Duplicate", self.duplicate_group),
                          ("Remove", self.remove_group)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        bv.addLayout(row)
        bv.addWidget(hint("Each group is one condition: tick its event types and narrow them "
                          "with “+ Filter by column” (e.g. label = human, answering_time < 2). "
                          "The label is used in all plots and tables."))
        lv.addWidget(box)
        self.stack = QtWidgets.QStackedWidget()
        lv.addWidget(self.stack, 1)
        left.setMinimumWidth(380)
        left.setMaximumWidth(520)
        split.addWidget(left)

        # ---------------------------------------------------------------- window
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        box = QtWidgets.QGroupBox("Epoch window and rejection")
        g = QtWidgets.QGridLayout(box)
        st = EpochSettings()
        # any epoch length: up to an hour before/after the event
        self.tmin = _spin(-MAX_EPOCH_S, MAX_EPOCH_S, st.tmin, 0.05, 3, " s")
        self.tmax = _spin(-MAX_EPOCH_S, MAX_EPOCH_S, st.tmax, 0.05, 3, " s")
        self.tmin.setToolTip("Epoch start relative to the event (negative = before)")
        self.tmax.setToolTip("Epoch end relative to the event")
        self.reject = _spin(0, 1e5, st.reject_uv or 0, 10, 0, " µV")
        self.reject.setSpecialValueText("off")
        self.reject.setToolTip("Drop epochs whose peak-to-peak exceeds this on any good EEG "
                               "channel")
        self.flat = _spin(0, 1000, 0, 0.5, 1, " µV")
        self.flat.setSpecialValueText("off")
        self.blinks = QtWidgets.QCheckBox("Drop epochs containing an eye-tracker blink")
        self.autoreject = QtWidgets.QCheckBox("AutoReject: interpolate bad channels per "
                                              "epoch, drop epochs it cannot repair")
        self.autoreject.setChecked(st.autoreject)
        self.autoreject.setToolTip("AutoReject (local), as in the ERP guide. Needs electrode "
                                   "positions; fitted on at most 600 epochs of all groups.")
        self.max_ep = QtWidgets.QSpinBox()
        self.max_ep.setRange(10, 1_000_000)
        self.max_ep.setValue(st.max_epochs)
        self.max_ep.setToolTip("Random, reproducible subsample above this many events per group")
        self.delay = _spin(-500, 500, 0, 1, 0, " ms")
        self.delay.setToolTip("EEG recording delay: a positive value means the EEG lags the "
                              "events (e.g. wireless amplifiers), so epochs are cut later")
        for i, (lab, w) in enumerate((("start", self.tmin), ("end", self.tmax),
                                      ("reject ptp >", self.reject), ("flat <", self.flat),
                                      ("max epochs / group", self.max_ep),
                                      ("EEG delay", self.delay))):
            g.addWidget(QtWidgets.QLabel(lab), i // 2, 2 * (i % 2))
            g.addWidget(w, i // 2, 2 * (i % 2) + 1)
        g.addWidget(self.blinks, 3, 0, 1, 4)
        g.addWidget(self.autoreject, 5, 0, 1, 4)
        g.addWidget(hint("Epochs are not baseline-corrected here: the ERP baseline and the "
                         "band-power baseline are chosen in Analysis. Make epochs long "
                         "enough for both (e.g. −0.5…2 s for band power)."), 4, 0, 1, 4)
        rv.addWidget(box)

        # recordings: subject / session / run / task / file of each event
        self.rec_box = QtWidgets.QGroupBox("Recordings — use events from")
        rl = QtWidgets.QHBoxLayout(self.rec_box)
        rl.setContentsMargins(6, 4, 6, 4)
        self._rec_lists: dict[str, QtWidgets.QListWidget] = {}
        self._rec_layout = rl
        self.rec_box.setVisible(False)
        rv.addWidget(self.rec_box)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Drop log (MNE)…", self.plot_drop_log),
                          ("Browse epochs (MNE)…", self.plot_epochs),
                          ("Save epochs…", self.save_epochs)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        rv.addLayout(row)
        self.summary = QtWidgets.QTableWidget()
        self.summary.setMaximumHeight(170)
        rv.addWidget(self.summary)
        self.timeline = _themed_plot(bottom=("Recording time", "s"))
        self.timeline.setMouseEnabled(y=False)
        self.views = QtWidgets.QTabWidget()
        tw = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tw)
        tl.setContentsMargins(0, 4, 0, 0)
        tl.addWidget(hint("Each bar is one epoch window (event + start … event + end); faint = "
                          "selected, solid = kept as an epoch, tick = the event (time 0). "
                          "Zoom with the mouse wheel."))
        tl.addWidget(self.timeline, 1)
        self.views.addTab(tw, "Timeline")
        ew = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(ew)
        el.setContentsMargins(0, 4, 0, 0)
        self.rows_info = hint("")
        el.addWidget(self.rows_info)
        self.rows = QtWidgets.QTableWidget()
        self.rows.setSortingEnabled(False)
        el.addWidget(self.rows, 1)
        self.views.addTab(ew, "Events of this group (all columns)")
        rv.addWidget(self.views, 1)
        split.addWidget(right)
        split.setSizes([440, 900])

        self._selectors: list[EventSelector] = []
        self._sigs: dict[int, tuple] = {}
        self.add_group(default=("fixation",))
        for w in (self.tmin, self.tmax, self.reject, self.flat):
            w.valueChanged.connect(lambda *_: self.pl.mark_stale("epoching"))
        self.max_ep.valueChanged.connect(lambda *_: self.pl.mark_stale("epoching"))
        self.delay.valueChanged.connect(lambda *_: self.pl.mark_stale("epoching"))
        self.blinks.toggled.connect(lambda *_: self.pl.mark_stale("epoching"))
        self.autoreject.toggled.connect(lambda *_: self.pl.mark_stale("epoching"))
        self.pl.changed.connect(self._on_pipeline)
        self.pl.stateChanged.connect(self._on_state)

    # ================================================================== groups
    def add_group(self, _=False, default=(), label=None):
        i = len(self._selectors)
        sel = EventSelector(label or f"Group {chr(65 + i) if i < 26 else i + 1}",
                            color_index=i, default_names=default)
        sel.set_table(self.pl.event_table)
        sel.changed.connect(lambda s=sel: self._group_changed(s))
        self._selectors.append(sel)
        self.stack.addWidget(sel)
        self.groups.addItem("")
        self._group_changed(sel)
        self.groups.setCurrentRow(i)
        return sel

    def duplicate_group(self):
        cur = self.current()
        if cur is None:
            return
        sel = self.add_group(default=cur.checked_names(), label=cur.label() + " copy")
        sel.text.setText(cur.text.text())
        sel.query.setText(cur.query.text())
        sel.set_conditions(cur.conditions())

    def remove_group(self):
        i = self.groups.currentRow()
        if i < 0 or len(self._selectors) <= 1:
            return
        sel = self._selectors.pop(i)
        self.stack.removeWidget(sel)
        sel.deleteLater()
        self.groups.takeItem(i)
        for k, s in enumerate(self._selectors):  # keep colours in list order
            s.color_index = k
            s.restyle()
        self.pl.mark_stale("epoching")

    def current(self) -> EventSelector | None:
        i = self.groups.currentRow()
        return self._selectors[i] if 0 <= i < len(self._selectors) else None

    def _select_group(self, i):
        if 0 <= i < len(self._selectors):
            self.stack.setCurrentWidget(self._selectors[i])
            self._show_rows()

    def _show_rows(self):
        """The current group's events with every column that has a value."""
        sel = self.current()
        if sel is None:
            return
        try:
            df = self.rows_for(sel)
        except ValueError as e:
            self.rows.clear()
            self.rows.setRowCount(0)
            self.rows_info.setText(f"Filter error: {e}")
            return
        cols = [c for c in df.columns if c not in ("description",)
                and df[c].notna().any() and (df[c].astype(str) != "").any()]
        view = df[cols].copy()
        for c in view.columns:
            if view[c].dtype == object:
                view[c] = view[c].fillna("")
        n = len(view)
        fill_table(self.rows, view.iloc[:3000], fmt="{:.4g}")
        self.rows_info.setText(f"{sel.label()}: {n} events"
                               + (" (first 3000 shown)" if n > 3000 else "")
                               + " — the columns are what the filters can use")

    def _group_changed(self, sel: EventSelector):
        if sel not in self._selectors:
            return
        i = self._selectors.index(sel)
        try:
            n = len(self.rows_for(sel))
        except ValueError:
            n = "?"
        item = self.groups.item(i)
        if item is not None:
            item.setText(f"{sel.label()}   ({n} events)")
            item.setForeground(pg.mkColor(THEME.group(i)))
        if sel is self.current():
            self._show_rows()
        sig = (sel.label(), tuple(sorted(sel.checked_names())), sel.text.text(),
               sel.query.text(), tuple(sel.conditions()), n)
        if self._sigs.get(id(sel)) != sig:  # only real changes make epochs out of date
            self._sigs[id(sel)] = sig
            self.pl.mark_stale("epoching")

    def labels(self) -> list[str]:
        return [s.label() for s in self._selectors]

    # ================================================================== pipeline
    # ================================================================== recordings
    REC_COLUMNS = (("subject", "subject"), ("session", "session"), ("run", "run"),
                   ("bids_task", "task"), ("file", "file"))

    def _fill_recordings(self):
        t = self.pl.event_table
        # remember what was unticked: values that appear later start ticked
        unticked = {k: {lst.item(i).data(QtCore.Qt.UserRole) for i in range(lst.count())
                        if lst.item(i).checkState() != QtCore.Qt.Checked}
                    for k, lst in self._rec_lists.items()}
        while self._rec_layout.count():
            it = self._rec_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self._rec_lists = {}
        shown = 0
        for col, label in self.REC_COLUMNS:
            if col not in t or not len(t):
                continue
            vals = t[col].fillna("").astype(str)
            counts = vals[vals != ""].value_counts().sort_index()
            if not len(counts):
                continue
            n_none = int((vals == "").sum())
            if n_none:  # events from files without this entity
                counts = pd.concat([counts, pd.Series({"": n_none})])
            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(QtWidgets.QLabel(f"<b>{label}</b>"))
            lst = QtWidgets.QListWidget()
            lst.setMaximumHeight(90)
            for val, n in counts.items():
                it = QtWidgets.QListWidgetItem(f"{val or '(none)'}  ({n})")
                it.setData(QtCore.Qt.UserRole, val)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(QtCore.Qt.Unchecked if val in unticked.get(col, set())
                                 else QtCore.Qt.Checked)
                lst.addItem(it)
            lst.itemChanged.connect(self._recordings_changed)
            v.addWidget(lst)
            self._rec_layout.addWidget(w, 1)
            self._rec_lists[col] = lst
            shown += 1
        self.rec_box.setVisible(shown > 0)

    def rec_selection(self, col) -> set[str] | None:
        lst = self._rec_lists.get(col)
        if lst is None:
            return None
        return {lst.item(i).data(QtCore.Qt.UserRole) for i in range(lst.count())
                if lst.item(i).checkState() == QtCore.Qt.Checked}

    def group_spec(self, sel: EventSelector) -> W.GroupSpec:
        """A group as plain settings: ticked names, filters and the ticked
        subjects/sessions/runs/files."""
        recs = {}
        for col, lst in self._rec_lists.items():
            keep = self.rec_selection(col)
            if keep is not None and lst.count():
                recs[col] = sorted(keep)
        return W.GroupSpec(label=sel.label(), names=sorted(sel.checked_names()),
                           text=sel.text.text(), query=sel.query.text(),
                           conditions=[list(c) for c in sel.conditions()], recordings=recs)

    def rows_for(self, sel: EventSelector):
        """A group's events, limited to the ticked subjects/sessions/runs/files
        (raises ValueError for invalid filters)."""
        return W.select_events(sel.table, self.group_spec(sel))

    def _recordings_changed(self, *_):
        for s in self._selectors:
            self._group_changed(s)
        self._show_rows()

    def _on_pipeline(self, what):
        if what in ("events", "recording"):
            for s in self._selectors:
                s.set_table(self.pl.event_table)
            self._fill_recordings()
            for s in self._selectors:
                self._group_changed(s)
        if what in ("epoching", "recording"):
            self._show_results()
        self._on_state()

    def _on_state(self):
        src = self.pl.clean
        if self.pl.raw is None:
            self.header.set_input("No EEG loaded.")
        elif src is None:
            self.header.set_input("Input: unprocessed recording (run ① Preprocessing)")
        else:
            self.header.set_input(f"Input: preprocessed data — {self.pl.params.describe()}")
        self.header.set_state(self.pl.state["epoching"])

    def settings(self) -> EpochSettings:
        return EpochSettings(tmin=self.tmin.value(), tmax=self.tmax.value(),
                             reject_uv=self.reject.value() or None,
                             flat_uv=self.flat.value() or None,
                             drop_blinks=self.blinks.isChecked(),
                             autoreject=self.autoreject.isChecked(),
                             max_epochs=self.max_ep.value(),
                             delay_ms=self.delay.value())

    def run(self, then=None) -> bool:
        raw = self.pl.eeg_input()
        if raw is None:
            return False
        st = self.settings()
        if st.tmax <= st.tmin:
            QtWidgets.QMessageBox.warning(self, "Epochs", "The epoch end must be after its start.")
            return False
        labels = self.labels()
        if len(set(labels)) != len(labels):
            QtWidgets.QMessageBox.warning(self, "Epochs", "Group labels must be unique.")
            return False
        groups, desc, specs = {}, {}, []
        for s in self._selectors:
            try:
                specs.append(self.group_spec(s))
                rows = self.rows_for(s)
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, "Event filter", f"{s.label()}: {e}")
                return False
            groups[s.label()] = rows
            desc[s.label()] = s.describe()
        if not any(len(r) for r in groups.values()):
            QtWidgets.QMessageBox.warning(self, "Epochs", "No events selected in any group.")
            return False
        need = epochs_memory(raw, groups, st)
        if need > MEMORY_WARN_BYTES:
            ans = QtWidgets.QMessageBox.question(
                self, "Large epochs",
                f"These epochs need about {_human(need)} of memory "
                f"({st.tmax - st.tmin:g} s × {sum(len(t) for t in groups.values())} events, "
                f"capped at {st.max_epochs} per group).\n\nLower ‘max epochs / group’ or "
                "shorten the window to use less. Build them anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if ans != QtWidgets.QMessageBox.Yes:
                return False
        blinks = self.pl.gaze_events.blinks if self.pl.gaze_events is not None else None
        self._selected = groups
        # what these epochs come from, for scripts that reproduce them
        recipe = self.pl.snapshot_epoch_recipe(raw, specs, st)

        def done(res):
            epochs, summary = res
            self.pl.set_epochs(epochs, summary, st, desc, recipe=recipe)
            if not epochs:
                QtWidgets.QMessageBox.warning(self, "Epochs", "All epochs were dropped — check "
                                              "the rejection threshold and the selection.")
            if then and epochs:
                then()

        self.header.set_state("running")
        return self.worker.run("Building epochs…",
                               lambda: build_group_epochs(raw, groups, st, blinks), done,
                               failed=lambda m: (self._on_state(),
                                                 QtWidgets.QMessageBox.warning(
                                                     self, "Epochs", m)))

    # ================================================================== display
    def restyle(self):
        self.timeline.setBackground(THEME.bg)
        for s in self._selectors:
            s.restyle()
        self._show_results()

    def _show_results(self):
        df = self.pl.epoch_summary
        if df is None:
            self.summary.clear()
            self.summary.setRowCount(0)
        else:
            fill_table(self.summary, df, fmt="{:g}")
            note = df.attrs.get("autoreject", "")
            self.summary.setToolTip(f"AutoReject {note}" if note else "")
            if note.startswith("not"):
                self.window().statusBar().showMessage(f"AutoReject {note}", 8000)
        self._plot_timeline()

    def _plot_timeline(self):
        """Each group on its own row: every epoch as a bar spanning its window
        (onset + start … onset + end); faint = selected, solid = kept as an epoch.
        A tick marks the event itself (time 0 of the epoch)."""
        p = self.timeline
        p.clear()
        sel = getattr(self, "_selected", {})
        labels = list(sel) or list(self.pl.epochs)
        st = self.pl.epoch_settings
        width = st.tmax - st.tmin
        delay = st.delay_ms / 1000.0
        for i, lab in enumerate(labels):
            y = len(labels) - 1 - i
            rows = sel.get(lab)
            if rows is not None and len(rows):
                on = rows["onset"].to_numpy(float) + delay
                p.addItem(pg.BarGraphItem(x0=on + st.tmin, width=width, y0=y - 0.3,
                                          height=0.6, brush=THEME.s(i, 35), pen=pg.mkPen(None)))
            ep = self.pl.epochs.get(lab)
            if ep is not None and ep.metadata is not None:
                on = ep.metadata["onset"].to_numpy(float)  # already shifted by the delay
                p.addItem(pg.BarGraphItem(x0=on + ep.tmin, width=ep.tmax - ep.tmin,
                                          y0=y - 0.3, height=0.6, brush=THEME.s(i, 110),
                                          pen=pg.mkPen(THEME.s(i), width=1)))
                p.plot(on, np.full(len(on), y), pen=None, symbol="|", symbolSize=18,
                       symbolPen=pg.mkPen(THEME.fg, width=1))
        p.getAxis("left").setTicks([[(len(labels) - 1 - i, lab) for i, lab in
                                     enumerate(labels)]])
        p.setYRange(-0.7, max(len(labels) - 0.3, 0.3), padding=0)

    def _need(self):
        if not self.pl.epochs:
            QtWidgets.QMessageBox.information(self, "Epochs", "Build epochs first.")
            return None
        lab = self.current().label() if self.current() else None
        return self.pl.epochs.get(lab) or next(iter(self.pl.epochs.values()))

    def plot_drop_log(self):
        ep = self._need()
        if ep is not None:
            ep.plot_drop_log(show=True, subject=ep.metadata["group"].iloc[0]
                             if ep.metadata is not None and len(ep) else None)

    def plot_epochs(self):
        ep = self._need()
        if ep is not None:
            ep.plot(n_epochs=10, show=True, block=False, events=False)

    def save_epochs(self):
        if not self.pl.epochs:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Save epochs (one -epo.fif per group)")
        if not d:
            return
        from pathlib import Path
        for lab, ep in self.pl.epochs.items():
            safe = "".join(c if c.isalnum() else "_" for c in lab)
            ep.save(Path(d) / f"{safe}-epo.fif", overwrite=True, verbose="error")
        self.window().statusBar().showMessage(f"Saved {len(self.pl.epochs)} epoch files to {d}",
                                              6000)

