"""Pipeline stage 1: bad channels (peak-to-peak), filtering, reference, ICA blinks."""

from __future__ import annotations

import mne
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import eeg as E
from .. import preprocessing as P
from .icaview import ICAComponentView
from .theme import THEME
from .widgets import (StageHeader, _spin, _themed_plot, has_positions, hint,
                      NO_POS)

UNCHECKED, CHECKED = QtCore.Qt.Unchecked, QtCore.Qt.Checked


class PreprocessTab(QtWidgets.QWidget):
    def __init__(self, pipeline, worker, parent=None):
        super().__init__(parent)
        self.pl, self.worker = pipeline, worker
        self._stats = None
        self._blink_before = self._blink_after = None
        outer = QtWidgets.QVBoxLayout(self)
        self.header = StageHeader("① Preprocessing", "Apply preprocessing ▶")
        self.header.run.clicked.connect(lambda: self.run())
        outer.addWidget(self.header)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        outer.addWidget(split, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        left = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(left)
        scroll.setWidget(left)
        scroll.setMinimumWidth(540)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        split.addWidget(scroll)

        # ---------------------------------------------------------------- channels
        box = QtWidgets.QGroupBox("Channels — automated peak-to-peak rejection")
        v = QtWidgets.QVBoxLayout(box)
        g = QtWidgets.QGridLayout()
        bp = P.BadChannelParams()
        self.ptp_max = _spin(1, 1e5, bp.ptp_max_uv, 10, 0, " µV")
        self.ptp_min = _spin(0, 1000, bp.ptp_min_uv, 0.5, 1, " µV")
        self.ptp_win = _spin(0.1, 30, bp.window, 0.5, 1, " s")
        self.ptp_frac = _spin(1, 100, 100 * bp.max_bad_fraction, 5, 0, " %")
        self.ptp_z = _spin(0, 20, bp.z_thresh, 0.5, 1, "")
        self.ptp_z.setSpecialValueText("off")
        for i, (lab, w, tip) in enumerate((
                ("noisy above", self.ptp_max, "A window whose peak-to-peak exceeds this is noisy"),
                ("flat below", self.ptp_min, "A window below this peak-to-peak is flat"),
                ("window", self.ptp_win, "Window length"),
                ("bad if more than", self.ptp_frac, "…of the windows are noisy or flat"),
                ("outlier |z| >", self.ptp_z, "Robust z of the channel's median peak-to-peak "
                                              "relative to the other channels"))):
            w.setToolTip(tip)
            g.addWidget(QtWidgets.QLabel(lab), i // 2, 2 * (i % 2))
            g.addWidget(w, i // 2, 2 * (i % 2) + 1)
        v.addLayout(g)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Detect bad channels")
        b.clicked.connect(self.detect_bads)
        row.addWidget(b)
        for label, fn in (("All good", lambda: self._set_all(True)),
                          ("Invert", self._invert)):
            bb = QtWidgets.QPushButton(label)
            bb.setFlat(True)
            bb.clicked.connect(fn)
            row.addWidget(bb)
        row.addStretch()
        self.n_bad = QtWidgets.QLabel("")
        row.addWidget(self.n_bad)
        v.addLayout(row)
        self.ch_table = QtWidgets.QTableWidget()
        self.ch_table.setMinimumHeight(220)
        self.ch_table.itemChanged.connect(self._ch_item_changed)
        v.addWidget(self.ch_table)
        self.interp = QtWidgets.QCheckBox("Interpolate bad channels (spherical spline, needs "
                                          "positions)")
        v.addWidget(self.interp)
        v.addWidget(hint("Untick a channel to mark it bad. Bad channels are excluded from ICA, "
                         "the average reference, epochs and statistics. Measured on the "
                         "band-passed/notched signal below."))
        lay.addWidget(box)

        # ---------------------------------------------------------------- filters
        box = QtWidgets.QGroupBox("Filtering && reference")
        f = QtWidgets.QFormLayout(box)
        self.hp = _spin(0, 100, 1.0, 0.1, 2, " Hz")
        self.lp = _spin(0, 1000, 40.0, 1, 1, " Hz")
        self.hp.setSpecialValueText("off")
        self.lp.setSpecialValueText("off")
        self.notch = QtWidgets.QComboBox()
        self.notch.addItems(["off", "50 Hz", "60 Hz"])
        self.ref = QtWidgets.QComboBox()
        self.ref.setEditable(True)
        self.ref.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.ref.setToolTip("keep = as recorded · average = common average (bad channels "
                            "left out) · a channel, or several separated by commas (their "
                            "mean, e.g. M1, M2)")
        self.ref.addItems(["keep", "average"])
        f.addRow("High-pass", self.hp)
        f.addRow("Low-pass", self.lp)
        f.addRow("Notch", self.notch)
        f.addRow("Reference", self.ref)
        lay.addWidget(box)

        # ---------------------------------------------------------------- ICA
        box = QtWidgets.QGroupBox("ICA — blink removal")
        v = QtWidgets.QVBoxLayout(box)
        g = QtWidgets.QHBoxLayout()
        ip = P.ICAParams()
        self.ica_on = QtWidgets.QCheckBox("Remove components")
        self.ica_on.setChecked(True)
        self.ica_n = QtWidgets.QSpinBox()
        self.ica_n.setRange(2, 256)
        self.ica_n.setValue(ip.n_components)
        self.ica_method = QtWidgets.QComboBox()
        self.ica_method.addItems(["picard", "fastica", "infomax"])
        self.ica_rej = _spin(0, 1e5, ip.reject_uv, 50, 0, " µV")
        self.ica_rej.setSpecialValueText("no rejection")
        self.ica_rej.setToolTip("Skip 2 s segments above this peak-to-peak while fitting")
        for lab, w in (("components", self.ica_n), ("method", self.ica_method),
                       ("fit reject", self.ica_rej)):
            g.addWidget(QtWidgets.QLabel(lab))
            g.addWidget(w)
        g.addStretch()
        v.addLayout(g)
        row = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Fit ICA")
        self.btn_fit.clicked.connect(self.fit_ica)
        row.addWidget(self.btn_fit)
        for label, fn in (("Use suggested", self._use_suggested),
                          ("Components (MNE)…", self.plot_components),
                          ("Sources (MNE)…", self.plot_sources),
                          ("Properties (MNE)…", self.plot_properties)):
            bb = QtWidgets.QPushButton(label)
            bb.clicked.connect(fn)
            row.addWidget(bb)
        row.addStretch()
        v.addLayout(row)
        self.ica_info = hint("Fit ICA on the good channels (≥ 1 Hz high-passed copy). "
                             "Components are scored against the eye tracker's blinks and "
                             "against EOG or, without EOG, the frontal electrodes.")
        v.addWidget(self.ica_info)
        self.ica_table = QtWidgets.QTableWidget()
        self.ica_table.setMinimumHeight(180)
        v.addWidget(self.ica_table)
        lay.addWidget(box)
        lay.addStretch()

        # ---------------------------------------------------------------- right
        right = QtWidgets.QTabWidget()
        self.psd_plot = _themed_plot(bottom=("Frequency", "Hz"), left="Power (dB µV²/Hz)")
        self.psd_plot.addLegend()
        right.addTab(self.psd_plot, "Spectrum")
        self.ptp_plot = _themed_plot(bottom="channel", left="noisy / flat windows (%)")
        right.addTab(self.ptp_plot, "Channels")
        self.blink_plot = _themed_plot(bottom=("Time from blink onset", "s"),
                                       left="Frontal EEG (µV)")
        self.blink_plot.addLegend()
        right.addTab(self.blink_plot, "Blink check")
        self.ica_view = ICAComponentView()
        right.addTab(self.ica_view, "ICA component")
        right.setTabToolTip(0, "Spectrum before / after preprocessing")
        right.setTabToolTip(1, "Noisy / flat windows per channel")
        self.right = right
        split.addWidget(right)
        split.setSizes([560, 860])

        self.pl.changed.connect(self._on_pipeline)
        self.pl.stateChanged.connect(self._on_state)
        stale = lambda *_: self.pl.mark_stale("preprocessing")  # noqa: E731
        for w in (self.hp, self.lp):
            w.valueChanged.connect(stale)
        for w in (self.notch, self.ref):
            w.currentTextChanged.connect(stale)
        for w in (self.interp, self.ica_on):
            w.toggled.connect(stale)
        self.ica_table.itemChanged.connect(stale)
        self.ica_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ica_table.currentCellChanged.connect(lambda r, *_: self.show_component(r))
        self.ica_table.cellClicked.connect(lambda r, *_: self.show_component(r))
        self.ica_table.cellDoubleClicked.connect(lambda *_: self.plot_properties())
        self.ica_table.itemChanged.connect(self._ica_item_changed)

    # ================================================================== pipeline
    def _on_pipeline(self, what):
        if what == "recording":
            self._stats = None
            self._blink_before = self._blink_after = None
            self.refresh_channels()
            self.refresh_reference_choices()
            self._fill_ica_table()
            self._plot_all()
        elif what == "preprocessing":
            self._plot_all()
        self._on_state()

    def _on_state(self):
        raw = self.pl.raw
        if raw is None:
            self.header.set_input("No EEG loaded.")
        else:
            n = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
            self.header.set_input(f"Input: recording — {n} EEG ch, "
                                  f"{raw.info['sfreq']:g} Hz, {raw.times[-1]:.0f} s"
                                  + (f"   ·   Output: {self.pl.params.describe()}"
                                     if self.pl.clean is not None else ""))
        self.header.set_state(self.pl.state["preprocessing"])

    def params(self) -> P.PreprocParams:
        notch = {"off": None, "50 Hz": 50.0, "60 Hz": 60.0}[self.notch.currentText()]
        exclude = self.ica_excluded() if self.ica_on.isChecked() else []
        return P.PreprocParams(bads=self.bads(), l_freq=self.hp.value() or None,
                               h_freq=self.lp.value() or None, notch=notch,
                               reference=self.ref.currentText(), ica_exclude=exclude,
                               interpolate_bads=self.interp.isChecked())

    def run(self, then=None) -> bool:
        raw = self.pl.raw
        if raw is None:
            return False
        p = self.params()
        try:
            E.parse_reference(p.reference, raw)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Reference", str(e))
            return False
        ref_names = E.parse_reference(p.reference, raw)
        if isinstance(ref_names, list) and set(ref_names) & set(p.bads):
            QtWidgets.QMessageBox.warning(self, "Reference",
                                          "The reference channel is marked bad.")
            return False
        if p.interpolate_bads and p.bads and not has_positions(raw):
            QtWidgets.QMessageBox.information(self, "Interpolation", NO_POS)
            p.interpolate_bads = False
        ica = self.pl.ica if p.ica_exclude else None
        if ica is not None and set(ica.ch_names) & set(p.bads):
            QtWidgets.QMessageBox.information(
                self, "ICA", "Some channels used to fit the ICA are now marked bad: refit "
                             "ICA for the best result. Applying the current ICA anyway.")
        blinks = self.pl.gaze_events.blinks if self.pl.gaze_events is not None else None
        fp = self._frontal(raw, p.bads)

        def job():
            clean = P.run_preprocessing(raw, p, ica)
            good = [c for c in clean.ch_names if c not in clean.info["bads"]]
            eeg_good = [c for c in good if c in
                        [clean.ch_names[i] for i in mne.pick_types(clean.info, eeg=True)]]
            before = E.compute_psd(raw.copy().pick(eeg_good) if eeg_good else raw, 60.0)
            after = E.compute_psd(clean, 60.0)
            bb = P.blink_locked_average(P._filtered(raw, p.l_freq, p.h_freq, p.notch, picks=fp)
                                        if fp else raw, blinks, fp)
            ba = P.blink_locked_average(clean, blinks, fp)
            return clean, before, after, bb, ba

        def done(res):
            clean, before, after, self._blink_before, self._blink_after = res
            self.pl.set_clean(clean, p, before, after)
            if then:
                then()

        self.header.set_state("running")
        return self.worker.run("Preprocessing…", job, done,
                               failed=lambda m: (self._on_state(),
                                                 QtWidgets.QMessageBox.warning(
                                                     self, "Preprocessing", m)))

    # ================================================================== channels
    def refresh_channels(self):
        raw = self.pl.raw
        self.ch_table.blockSignals(True)
        self.ch_table.clear()
        if raw is None:
            self.ch_table.setRowCount(0)
            self.ch_table.blockSignals(False)
            return
        names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])] \
            or [raw.ch_names[i] for i in E.eeg_picks(raw)]
        cols = ["use", "channel", "median ptp µV", "noisy %", "flat %", "z", "reason"]
        self.ch_table.setColumnCount(len(cols))
        self.ch_table.setHorizontalHeaderLabels(cols)
        self.ch_table.setRowCount(len(names))
        st = self._stats.set_index("channel") if self._stats is not None else None
        bads = set(self.pl.params.bads) | set(raw.info["bads"])
        for i, n in enumerate(names):
            use = QtWidgets.QTableWidgetItem("")
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            use.setCheckState(UNCHECKED if n in bads else CHECKED)
            use.setData(QtCore.Qt.UserRole, n)
            self.ch_table.setItem(i, 0, use)
            vals = [n]
            if st is not None and n in st.index:
                r = st.loc[n]
                vals += [f"{r.median_ptp_uv:.1f}", f"{r.noisy_pct:.0f}", f"{r.flat_pct:.0f}",
                         f"{r.z:+.1f}", r.reason]
            for j, txt in enumerate(vals, start=1):
                it = QtWidgets.QTableWidgetItem(str(txt))
                it.setFlags(QtCore.Qt.ItemIsEnabled)
                self.ch_table.setItem(i, j, it)
        self.ch_table.resizeColumnsToContents()
        self.ch_table.blockSignals(False)
        self._update_bad_count()

    def bads(self) -> list[str]:
        out = []
        for i in range(self.ch_table.rowCount()):
            it = self.ch_table.item(i, 0)
            if it is not None and it.checkState() == UNCHECKED:
                out.append(it.data(QtCore.Qt.UserRole))
        return out

    def _ch_item_changed(self, item):
        if item.column() == 0:
            self._update_bad_count()
            self._plot_ptp()
            self.pl.mark_stale("preprocessing")

    def _update_bad_count(self):
        b = self.bads()
        self.n_bad.setText(f"{len(b)} bad" + (f": {', '.join(b[:6])}{' …' if len(b) > 6 else ''}"
                                               if b else ""))

    def _set_all(self, good: bool):
        self.ch_table.blockSignals(True)
        for i in range(self.ch_table.rowCount()):
            self.ch_table.item(i, 0).setCheckState(CHECKED if good else UNCHECKED)
        self.ch_table.blockSignals(False)
        self._ch_item_changed(self.ch_table.item(0, 0) or QtWidgets.QTableWidgetItem())

    def _invert(self):
        self.ch_table.blockSignals(True)
        for i in range(self.ch_table.rowCount()):
            it = self.ch_table.item(i, 0)
            it.setCheckState(CHECKED if it.checkState() == UNCHECKED else UNCHECKED)
        self.ch_table.blockSignals(False)
        if self.ch_table.rowCount():
            self._ch_item_changed(self.ch_table.item(0, 0))

    def detect_bads(self):
        raw = self.pl.raw
        if raw is None:
            return
        notch = {"off": None, "50 Hz": 50.0, "60 Hz": 60.0}[self.notch.currentText()]
        p = P.BadChannelParams(ptp_max_uv=self.ptp_max.value(), ptp_min_uv=self.ptp_min.value(),
                               window=self.ptp_win.value(),
                               max_bad_fraction=self.ptp_frac.value() / 100,
                               z_thresh=self.ptp_z.value() or np.inf,
                               l_freq=self.hp.value() or None, h_freq=self.lp.value() or None,
                               notch=notch)
        ref = self.ref.currentText()

        def done(df):
            self._stats = df
            try:  # never flag the chosen reference channel(s) for being flat
                refs = E.parse_reference(ref, raw)
            except ValueError:
                refs = None
            bads = df.loc[df["bad"], "channel"].tolist()
            if isinstance(refs, list):
                bads = [b for b in bads if b not in refs]
            self.pl.params.bads = bads
            self.refresh_channels()
            self._plot_ptp()
            self.right.setCurrentIndex(1)
            self.pl.mark_stale("preprocessing")

        self.worker.run("Measuring peak-to-peak per channel…",
                        lambda: P.detect_bad_channels(raw, p), done)

    # ================================================================== reference
    def refresh_reference_choices(self):
        cur = self.ref.currentText()
        raw = self.pl.raw
        names = ([raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
                 if raw is not None else [])
        self.ref.blockSignals(True)
        self.ref.clear()
        self.ref.addItems(["keep", "average"])
        if names:
            self.ref.insertSeparator(2)
            self.ref.addItems(names)
        try:
            ok = raw is not None and E.parse_reference(cur, raw) is not None
        except ValueError:
            ok = False
        self.ref.setCurrentText(cur if ok or cur in ("keep", "average") else "keep")
        self.ref.blockSignals(False)

    # ================================================================== ICA
    def _frontal(self, raw, bads):
        lower = {c.lower(): c for c in raw.ch_names}
        return [lower[c.lower()] for c in P.FRONTAL
                if c.lower() in lower and lower[c.lower()] not in bads][:2]

    def fit_ica(self):
        raw = self.pl.raw
        if raw is None:
            return
        bads = self.bads()
        notch = {"off": None, "50 Hz": 50.0, "60 Hz": 60.0}[self.notch.currentText()]
        ip = P.ICAParams(n_components=self.ica_n.value(), method=self.ica_method.currentText(),
                         reject_uv=self.ica_rej.value() or None)
        h = self.lp.value() or None
        blinks = self.pl.gaze_events.blinks if self.pl.gaze_events is not None else None
        eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True)]

        def job():
            ica, fit_raw = P.fit_ica(raw, bads, h, notch, ip)
            scores = P.score_blink_components(ica, fit_raw, blinks, eog)
            sources = ica.get_sources(fit_raw).get_data()
            return ica, fit_raw, scores, sources

        def done(res):
            self.pl.ica, self.pl.ica_fit_raw, self.pl.ica_scores, self.pl.ica_sources = res
            self.pl.ica_key = (tuple(bads), h, notch)
            self._fill_ica_table(use_suggested=True)
            self.pl.mark_stale("preprocessing")
            if self.ica_table.rowCount():
                self.ica_table.selectRow(0)
                self.show_component(0)

        self.worker.run(f"Fitting ICA ({ip.method}, {ip.n_components} components)…", job, done)

    def _fill_ica_table(self, use_suggested=False):
        sc = self.pl.ica_scores
        self.ica_table.blockSignals(True)
        self.ica_table.clear()
        self.ica_table.blockSignals(False)
        if sc is None:
            self.ica_table.setRowCount(0)
            self.ica_view._args = None
            self.ica_view.restyle()
            return
        cols = ["exclude", "component", "EOG/frontal |r|", "blink-locked t", "suggested"]
        self.ica_table.setColumnCount(len(cols))
        self.ica_table.setHorizontalHeaderLabels(cols)
        self.ica_table.setRowCount(len(sc))
        prev = set(self.pl.ica.exclude or []) if self.pl.ica is not None else set()
        for i, r in enumerate(sc.itertuples()):
            ex = QtWidgets.QTableWidgetItem("")
            ex.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            on = r.suggested if use_suggested else r.component in prev
            ex.setCheckState(CHECKED if on else UNCHECKED)
            ex.setData(QtCore.Qt.UserRole, int(r.component))
            self.ica_table.setItem(i, 0, ex)
            for j, txt in enumerate((f"ICA{r.component:03d}",
                                     "" if np.isnan(r.eog_score) else f"{r.eog_score:.2f}",
                                     "" if np.isnan(r.blink_t) else f"{r.blink_t:.1f}",
                                     "✓" if r.suggested else ""), start=1):
                it = QtWidgets.QTableWidgetItem(txt)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.ica_table.setItem(i, j, it)
        self.ica_table.resizeColumnsToContents()
        n_sug = int(sc["suggested"].sum())
        proxy = sc.attrs.get("proxy")
        eog = ", ".join(sc.attrs.get("eog_channels") or []) or "none"
        self.ica_info.setText(
            f"{len(sc)} components on {len(self.pl.ica.ch_names)} channels. "
            f"{n_sug} suggested as blink components "
            f"({'frontal proxies' if proxy else 'EOG'}: {eog}; eye-tracker blinks: "
            f"{'yes' if np.isfinite(sc['blink_t']).any() else 'no'}). "
            "Inspect with ‘Components’ / ‘Properties’ and tick what to remove."
            + ("" if n_sug else " None stands out: if blinks are not visible in the EEG, "
                                "removing nothing is the right call."))

    def show_component(self, row: int):
        """Display the component of a table row in the 'ICA component' panel."""
        pl = self.pl
        if pl.ica is None or getattr(pl, "ica_sources", None) is None or row < 0 \
                or row >= self.ica_table.rowCount():
            return
        it = self.ica_table.item(row, 0)
        k = int(it.data(QtCore.Qt.UserRole))
        blinks = pl.gaze_events.blinks if pl.gaze_events is not None else None
        try:
            self.ica_view.show_component(pl.ica, pl.ica_fit_raw, pl.ica_sources, k, blinks,
                                         pl.ica_scores, it.checkState() == CHECKED)
        except Exception as e:  # a display problem must not break the table
            self.ica_view._message(f"Cannot display ICA{k:03d}: {e}")
        self.right.setCurrentWidget(self.ica_view)

    def _ica_item_changed(self, item):
        if item.column() == 0 and item.row() == self.ica_table.currentRow():
            self.show_component(item.row())  # recolour when (un)ticked

    def ica_excluded(self) -> list[int]:
        out = []
        for i in range(self.ica_table.rowCount()):
            it = self.ica_table.item(i, 0)
            if it is not None and it.checkState() == CHECKED:
                out.append(int(it.data(QtCore.Qt.UserRole)))
        return out

    def _use_suggested(self):
        if self.pl.ica_scores is not None:
            self._fill_ica_table(use_suggested=True)

    def _need_ica(self) -> bool:
        if self.pl.ica is None:
            QtWidgets.QMessageBox.information(self, "ICA", "Fit ICA first.")
            return False
        return True

    def plot_components(self):
        if not self._need_ica():
            return
        if not has_positions(self.pl.ica_fit_raw):
            QtWidgets.QMessageBox.information(self, "ICA components", NO_POS)
            return
        self.pl.ica.exclude = self.ica_excluded()
        self.pl.ica.plot_components(inst=self.pl.ica_fit_raw, show=True)

    def plot_sources(self):
        if self._need_ica():
            self.pl.ica.exclude = self.ica_excluded()
            self.pl.ica.plot_sources(self.pl.ica_fit_raw, show=True, block=False)

    def plot_properties(self):
        if not self._need_ica():
            return
        rows = sorted({i.row() for i in self.ica_table.selectedIndexes()})
        picks = [int(self.ica_table.item(r, 0).data(QtCore.Qt.UserRole)) for r in rows] \
            or self.ica_excluded()[:3] or [0]
        if not has_positions(self.pl.ica_fit_raw):
            QtWidgets.QMessageBox.information(self, "ICA properties", NO_POS)
            return
        self.pl.ica.plot_properties(self.pl.ica_fit_raw, picks=picks[:5], show=True,
                                    verbose="error")

    # ================================================================== plots
    def restyle(self):
        for p in (self.psd_plot, self.ptp_plot, self.blink_plot):
            p.setBackground(THEME.bg)
        self.ica_view.restyle()
        self._plot_all()

    def _plot_all(self):
        self._plot_psd()
        self._plot_ptp()
        self._plot_blink()

    def _plot_psd(self):
        self.psd_plot.clear()
        for psd, lab, col, width in ((self.pl.psd_before, "before", THEME.muted, 1.6),
                                     (self.pl.psd_after, "after", THEME.s(0), 2.4)):
            if psd is None:
                continue
            data, freqs = psd.get_data(return_freqs=True)
            db = 10 * np.log10(np.maximum(data * 1e12, 1e-20))
            if lab == "after":
                for i, n in enumerate(psd.ch_names):
                    c = self.psd_plot.plot(freqs, db[i], pen=pg.mkPen(THEME.s(0, 45)))
                    c.setToolTip(n)
            self.psd_plot.plot(freqs, np.median(db, 0), pen=pg.mkPen(col, width=width),
                               name=f"{lab} (median of channels)")

    def _plot_ptp(self):
        self.ptp_plot.clear()
        st = self._stats
        if st is None:
            return
        bads = set(self.bads())
        x = np.arange(len(st))
        noisy = st["noisy_pct"].to_numpy()
        flat = st["flat_pct"].to_numpy()
        cols = [THEME.event if c in bads else THEME.s(0) for c in st["channel"]]
        self.ptp_plot.addItem(pg.BarGraphItem(x=x, height=noisy, width=0.8,
                                              brushes=cols, pen=None))
        if flat.any():
            self.ptp_plot.addItem(pg.BarGraphItem(x=x, height=-flat, width=0.8,
                                                  brush=THEME.rgba("muted", 160), pen=None))
        self.ptp_plot.addItem(pg.InfiniteLine(self.ptp_frac.value(), angle=0,
                                              pen=pg.mkPen(THEME.fg, style=QtCore.Qt.DashLine)))
        ax = self.ptp_plot.getAxis("bottom")
        step = max(1, int(np.ceil(len(st) / 20)))
        ax.setTicks([[(i, st["channel"].iloc[i]) for i in range(0, len(st), step)]])
        self.ptp_plot.setLabel("left", "noisy windows % (up) · flat % (down)")

    def _plot_blink(self):
        self.blink_plot.clear()
        for res, lab, col in ((self._blink_before, "before", THEME.muted),
                              (self._blink_after, "after", THEME.s(0))):
            if res is not None:
                self.blink_plot.plot(res[0], res[1], pen=pg.mkPen(col, width=2), name=lab)
        if self._blink_before is None and self._blink_after is None:
            self.blink_plot.addItem(pg.TextItem("Needs eye-tracker blinks and a frontal "
                                                "channel (Fp1/Fp2…)", color=THEME.muted))
        self.blink_plot.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(THEME.muted)))
