"""Analysis tabs (all EEG analysis runs in MNE):

* Gaze statistics + PyGaze AOIs
* Spectrum (``Raw.compute_psd``)
* Event-related potentials: ``mne.Epochs`` locked to any selection of events
  from the unified event table (carried as epochs metadata)
* Contrast: two event groups → evokeds, ``mne.combine_evoked`` difference,
  cluster-based permutation tests from ``mne.stats``.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import eeg as E
from .. import gaze as G
from .eventselect import EventSelector
from .theme import THEME


def fill_table(table: QtWidgets.QTableWidget, df: pd.DataFrame, fmt="{:.3g}"):
    table.clear()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i, row in enumerate(df.itertuples(index=False)):
        for j, v in enumerate(row):
            if isinstance(v, (float, np.floating)):
                txt = "" if not np.isfinite(v) else fmt.format(v)
            else:
                txt = str(v)
            item = QtWidgets.QTableWidgetItem(txt)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            table.setItem(i, j, item)
    table.resizeColumnsToContents()


def has_positions(inst) -> bool:
    """True if at least 4 EEG channels of an MNE object have sensor locations."""
    locs = np.array([ch["loc"][:3] for ch in inst.info["chs"]
                     if ch["kind"] == 2])  # FIFFV_EEG_CH
    return len(locs) > 0 and int((np.isfinite(locs).all(1) & (np.abs(locs).sum(1) > 0)).sum()) >= 4


NO_POS = ("The EEG channels have no electrode positions. Use File ▸ Channel names & "
          "montage… to load one; until then scalp topographies cannot be drawn.")


def _spin(lo, hi, val, step=0.05, dec=3, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def _themed_plot(**labels) -> pg.PlotWidget:
    p = pg.PlotWidget(background=THEME.bg)
    for side, text in labels.items():
        if isinstance(text, tuple):
            p.setLabel(side, *text)
        else:
            p.setLabel(side, text)
    p.showGrid(x=True, y=True, alpha=0.3)
    return p


class EpochParams(QtWidgets.QWidget):
    """tmin / tmax / baseline / rejection / epoch cap shared by ERP and contrasts."""

    def __init__(self, tmin=-0.2, tmax=0.8, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tmin = _spin(-5, 0, tmin, 0.05, 3, " s")
        self.tmax = _spin(0.01, 10, tmax, 0.05, 3, " s")
        self.baseline = QtWidgets.QCheckBox("Baseline (tmin…0)")
        self.baseline.setChecked(True)
        self.reject = _spin(0, 5000, 0, 10, 0, " µV")
        self.reject.setSpecialValueText("no rejection")
        self.reject.setToolTip("Peak-to-peak rejection threshold (MNE reject=dict(eeg=…))")
        self.max_epochs = QtWidgets.QSpinBox()
        self.max_epochs.setRange(10, 100000)
        self.max_epochs.setValue(2000)
        self.max_epochs.setToolTip("Random (reproducible) subsample above this many events, "
                                   "to bound memory")
        for lab, w in (("tmin", self.tmin), ("tmax", self.tmax), (None, self.baseline),
                       ("reject", self.reject), ("max epochs", self.max_epochs)):
            if lab:
                lay.addWidget(QtWidgets.QLabel(lab))
            lay.addWidget(w)
        lay.addStretch()

    def kwargs(self) -> dict:
        return dict(tmin=self.tmin.value(), tmax=self.tmax.value(),
                    baseline=(None, 0) if self.baseline.isChecked() else None,
                    reject_uv=self.reject.value() or None,
                    max_epochs=self.max_epochs.value())


class AnalysisPanel(QtWidgets.QTabWidget):
    """``state`` is the main window, providing rec, display_raw, gaze_events, screen."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.evoked = None
        self.epochs = None
        self.psd = None
        self.contrast = None  # dict with epochs, evokeds, stats
        self._aoi_df = None
        self._build_gaze_tab()
        self._build_psd_tab()
        self._build_erp_tab()
        self._build_contrast_tab()

    # ================================================================ common
    def set_event_table(self, table: pd.DataFrame):
        for sel in (self.erp_sel, self.con_a, self.con_b):
            sel.set_table(table)

    def restyle(self):
        for sel in (self.erp_sel, self.con_a, self.con_b):
            sel.restyle()
        self._plot_gaze_stats()
        self._plot_psd()
        self._plot_erp()
        self._plot_contrast()

    def _warn(self, title, msg):
        QtWidgets.QMessageBox.warning(self, title, str(msg))

    def _epochs_for(self, selector: EventSelector, params: EpochParams) -> mne.Epochs | None:
        raw = self.state.display_raw
        if raw is None:
            self._warn("Epochs", "No EEG loaded.")
            return None
        try:
            rows = selector.selected()
        except ValueError as e:
            self._warn("Event filter", e)
            return None
        if not len(rows):
            self._warn("Epochs", f"No events selected in “{selector.label()}”: tick event "
                                 "types in the checklist and/or relax the filters.")
            return None
        try:
            with self.state._busy(f"Epoching {len(rows)} events…"):
                ep = E.epochs_from_table(raw, rows, **params.kwargs())
        except Exception as e:
            self._warn("Epochs", e)
            return None
        if not len(ep):
            self._warn("Epochs", f"All epochs of “{selector.label()}” were rejected.")
            return None
        return ep

    # ================================================================ gaze tab
    def _build_gaze_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        lay.addLayout(top)
        box = QtWidgets.QGroupBox("Oculomotor summary")
        bl = QtWidgets.QVBoxLayout(box)
        self.summary_table = QtWidgets.QTableWidget()
        bl.addWidget(self.summary_table)
        top.addWidget(box, 1)
        box2 = QtWidgets.QGroupBox("Areas of interest (PyGaze AOI)")
        b2 = QtWidgets.QVBoxLayout(box2)
        row = QtWidgets.QHBoxLayout()
        for label, kind in (("+ Rectangle", "rectangle"), ("+ Ellipse", "ellipse")):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self.state.screen.add_aoi(k))
            row.addWidget(b)
        clr = QtWidgets.QPushButton("Clear")
        clr.clicked.connect(lambda: self.state.screen.clear_aois())
        row.addWidget(clr)
        exp = QtWidgets.QPushButton("Export CSV…")
        exp.clicked.connect(self.export_aoi)
        row.addWidget(exp)
        b2.addLayout(row)
        self.aoi_table = QtWidgets.QTableWidget()
        b2.addWidget(self.aoi_table)
        hint = QtWidgets.QLabel("Drag/resize AOIs in the screen view; right-click an AOI "
                                "to remove it. Fixations get an ‘aoi’ column usable in "
                                "event filters.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        b2.addWidget(hint)
        top.addWidget(box2, 1)
        self.hist = pg.GraphicsLayoutWidget()
        self.hist.setBackground(THEME.bg)
        self.hist_fix = self.hist.addPlot(title="Fixation duration (ms)")
        self.hist_sac = self.hist.addPlot(title="Saccade amplitude (px)")
        self.hist_main = self.hist.addPlot(title="Main sequence")
        self.hist_main.setLabel("bottom", "amplitude (px)")
        self.hist_main.setLabel("left", "peak velocity (px/s)")
        lay.addWidget(self.hist, 1)
        self.addTab(w, "Gaze statistics")

    def update_gaze_stats(self):
        ev, rec = self.state.gaze_events, self.state.rec
        if ev is None or rec is None or not rec.has_gaze:
            self.summary_table.clear()
            self.aoi_table.clear()
        else:
            fill_table(self.summary_table, G.summary_table(ev, rec.gaze.duration))
        self.update_aoi_stats()
        self._plot_gaze_stats()

    def _plot_gaze_stats(self):
        ev = self.state.gaze_events
        for p in (self.hist_fix, self.hist_sac, self.hist_main):
            p.clear()
        if ev is None:
            return
        if len(ev.fixations):
            y, x = np.histogram(ev.fixations["duration"] * 1000, bins=40)
            self.hist_fix.plot(x, y, stepMode="center", fillLevel=0, brush=THEME.s(0, 170),
                               pen=pg.mkPen(THEME.s(0), width=1))
        if len(ev.saccades):
            y, x = np.histogram(ev.saccades["amplitude"], bins=40)
            self.hist_sac.plot(x, y, stepMode="center", fillLevel=0, brush=THEME.s(1, 170),
                               pen=pg.mkPen(THEME.s(1), width=1))
            self.hist_main.plot(ev.saccades["amplitude"].to_numpy(),
                                ev.saccades["peak_velocity"].to_numpy(), pen=None,
                                symbol="o", symbolSize=4, symbolBrush=THEME.rgba("muted", 140),
                                symbolPen=None)

    def update_aoi_stats(self):
        ev = self.state.gaze_events
        aois = self.state.screen.pygaze_aois() if self.state.screen.aois else {}
        if ev is None or not aois:
            self.aoi_table.clear()
            self.aoi_table.setRowCount(0)
            self._aoi_df = None
            return
        self._aoi_df = G.aoi_statistics(ev.fixations, aois)
        fill_table(self.aoi_table, self._aoi_df)

    def export_aoi(self):
        if self._aoi_df is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export AOI stats", "aoi.csv",
                                                      "CSV (*.csv)")
        if fn:
            self._aoi_df.to_csv(fn, index=False)

    # ================================================================ PSD tab
    def _build_psd_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.psd_fmax = _spin(1, 500, 60, 5, 0, " Hz")
        self.psd_window_only = QtWidgets.QCheckBox("Visible window only")
        btn = QtWidgets.QPushButton("Compute PSD")
        btn.clicked.connect(self.compute_psd)
        topo = QtWidgets.QPushButton("Band topomaps (MNE)…")
        topo.clicked.connect(self.psd_topomap)
        for x in (QtWidgets.QLabel("fmax"), self.psd_fmax, self.psd_window_only, btn, topo):
            row.addWidget(x)
        row.addStretch()
        lay.addLayout(row)
        self.psd_plot = _themed_plot(bottom=("Frequency", "Hz"), left="Power (dB µV²/Hz)")
        lay.addWidget(self.psd_plot)
        self.addTab(w, "Spectrum")

    def compute_psd(self):
        raw = self.state.display_raw
        if raw is None:
            return
        tmin = tmax = None
        if self.psd_window_only.isChecked():
            tmin, tmax = self.state.eeg_plot.window()
            tmin, tmax = max(tmin, 0), min(tmax, raw.times[-1])
        with self.state._busy("Computing PSD…"):
            self.psd = E.compute_psd(raw, self.psd_fmax.value(), tmin, tmax)
        self._plot_psd()

    def _plot_psd(self):
        self.psd_plot.clear()
        if self.psd is None:
            return
        data, freqs = self.psd.get_data(return_freqs=True)
        db = 10 * np.log10(np.maximum(data * 1e12, 1e-20))
        for i, name in enumerate(self.psd.ch_names):
            c = self.psd_plot.plot(freqs, db[i], pen=pg.mkPen(THEME.rgba("muted", 110)))
            c.setToolTip(name)
        self.psd_plot.plot(freqs, db.mean(0), pen=pg.mkPen(THEME.s(0), width=2.5))

    def psd_topomap(self):
        if self.psd is None:
            self.compute_psd()
        if self.psd is None:
            return
        if not has_positions(self.psd):
            QtWidgets.QMessageBox.information(self, "Topomap", NO_POS)
            return
        try:
            self.psd.plot_topomap(show=True)
        except Exception as e:
            self._warn("Topomap", e)

    # ================================================================ ERP tab
    def _build_erp_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        self.erp_sel = EventSelector("Lock to", default_names=("fixation",))
        self.erp_sel.setMaximumWidth(360)
        outer.addWidget(self.erp_sel)
        right = QtWidgets.QVBoxLayout()
        outer.addLayout(right, 1)
        self.erp_params = EpochParams(-0.2, 0.6)
        right.addWidget(self.erp_params)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Compute", self.compute_erp), ("Joint plot (MNE)…", self.erp_joint),
                          ("Epochs image (MNE)…", self.erp_image),
                          ("Save epochs…", self.save_epochs)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        right.addLayout(row)
        self.erp_info = QtWidgets.QLabel("")
        self.erp_info.setEnabled(False)
        self.erp_info.setWordWrap(True)
        right.addWidget(self.erp_info)
        self.erp_plot = _themed_plot(bottom=("Time", "s"), left="Amplitude (µV)")
        self.erp_plot.addLegend()
        right.addWidget(self.erp_plot, 1)
        self.addTab(w, "Event-related potentials")

    def compute_erp(self):
        ep = self._epochs_for(self.erp_sel, self.erp_params)
        if ep is None:
            return
        self.epochs = ep
        self.evoked = ep.average()
        self.evoked.comment = self.erp_sel.label()
        dropped = sum(1 for d in ep.drop_log if d)
        self.erp_info.setText(f"{self.erp_sel.describe()} → {len(ep)} epochs averaged "
                              f"({dropped} rejected), {len(self.evoked.ch_names)} channels")
        self._plot_erp()

    def _plot_erp(self):
        self.erp_plot.clear()
        if self.evoked is None:
            return
        t = self.evoked.times
        data = self.evoked.data * 1e6
        for i, name in enumerate(self.evoked.ch_names):
            c = self.erp_plot.plot(t, data[i], pen=pg.mkPen(THEME.rgba("muted", 100)))
            c.setToolTip(name)
        self.erp_plot.plot(t, data.std(0), pen=pg.mkPen(THEME.fg, width=2.5), name="GFP")
        k = 0
        for name in ("Oz", "Pz", "Cz", "Fz"):
            if name in self.evoked.ch_names:
                i = self.evoked.ch_names.index(name)
                self.erp_plot.plot(t, data[i], pen=pg.mkPen(THEME.s(k), width=2), name=name)
                k += 1
        self.erp_plot.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(THEME.muted)))

    def erp_joint(self):
        if self.evoked is None:
            self.compute_erp()
        if self.evoked is None:
            return
        try:
            if has_positions(self.evoked):
                self.evoked.plot_joint(show=True)
            else:  # butterfly without topomaps rather than guessed positions
                self.evoked.plot(spatial_colors=False, gfp=True, show=True)
        except Exception as e:
            self._warn("Joint plot", e)

    def erp_image(self):
        if self.epochs is None:
            self.compute_erp()
        if self.epochs is not None:
            picks = [c for c in ("Oz", "Pz", "Cz") if c in self.epochs.ch_names][:1] or [0]
            self.epochs.plot_image(picks=picks, show=True)

    def save_epochs(self):
        if self.epochs is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save epochs (with metadata)",
                                                      "selection-epo.fif", "FIF (*-epo.fif)")
        if fn:
            self.epochs.save(fn, overwrite=True, verbose="error")

    # ================================================================ contrast tab
    def _build_contrast_tab(self):
        w = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(w)
        groups = QtWidgets.QVBoxLayout()
        self.con_a = EventSelector("Group A", color_index=0)
        self.con_b = EventSelector("Group B", color_index=1)
        groups.addWidget(self.con_a, 1)
        groups.addWidget(self.con_b, 1)
        gw = QtWidgets.QWidget()
        gw.setLayout(groups)
        gw.setMaximumWidth(380)
        outer.addWidget(gw)

        right = QtWidgets.QVBoxLayout()
        outer.addLayout(right, 1)
        self.con_params = EpochParams(-0.2, 0.8)
        right.addWidget(self.con_params)
        row = QtWidgets.QHBoxLayout()
        self.con_roi = QtWidgets.QLineEdit()
        self.con_roi.setPlaceholderText("ROI channels, e.g. O1, Oz, O2 (empty = all)")
        self.con_perm = QtWidgets.QSpinBox()
        self.con_perm.setRange(0, 100000)
        self.con_perm.setValue(1000)
        self.con_perm.setSpecialValueText("no test")
        self.con_spatial = QtWidgets.QCheckBox("Spatio-temporal (all channels)")
        self.con_spatial.setToolTip("mne.stats.spatio_temporal_cluster_test with the montage "
                                    "adjacency (needs electrode positions)")
        self.con_twin = QtWidgets.QCheckBox("test 0…tmax only")
        self.con_twin.setChecked(True)
        for lab, x in (("ROI", self.con_roi), ("permutations", self.con_perm),
                       (None, self.con_spatial), (None, self.con_twin)):
            if lab:
                row.addWidget(QtWidgets.QLabel(lab))
            row.addWidget(x, 1 if x is self.con_roi else 0)
        right.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Compute contrast", self.compute_contrast),
                          ("Compare evokeds (MNE)…", self.contrast_compare),
                          ("Difference topomaps (MNE)…", self.contrast_topo),
                          ("Export…", self.contrast_export)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        right.addLayout(row)
        self.con_info = QtWidgets.QLabel("Tick event types for each group (use filters for "
                                         "conditions, phases, AOIs…), then Compute.")
        self.con_info.setWordWrap(True)
        self.con_info.setEnabled(False)
        right.addWidget(self.con_info)
        self.con_plot = _themed_plot(bottom=("Time", "s"), left="Amplitude (µV)")
        self.con_plot.addLegend()
        right.addWidget(self.con_plot, 1)
        self.con_table = QtWidgets.QTableWidget()
        self.con_table.setMaximumHeight(140)
        right.addWidget(self.con_table)
        self.addTab(w, "Contrast")

    def _roi(self, epochs) -> list[str]:
        txt = self.con_roi.text().replace(";", ",")
        wanted = [c.strip() for c in txt.split(",") if c.strip()]
        lower = {c.lower(): c for c in epochs.ch_names}
        roi = [lower[c.lower()] for c in wanted if c.lower() in lower]
        if wanted and not roi:
            raise ValueError(f"None of the ROI channels {wanted} exist")
        return roi or list(epochs.ch_names)

    def compute_contrast(self):
        a = self._epochs_for(self.con_a, self.con_params)
        if a is None:
            return
        b = self._epochs_for(self.con_b, self.con_params)
        if b is None:
            return
        la, lb = self.con_a.label(), self.con_b.label()
        if la == lb:
            lb += " (2)"
        try:
            roi = self._roi(a)
        except ValueError as e:
            self._warn("ROI", e)
            return
        evokeds = E.contrast_evokeds(a, b, (la, lb))
        stats = None
        notes = []
        overlap = len(set(a.events[:, 0]) & set(b.events[:, 0]))
        if overlap:
            notes.append(f"⚠ {overlap} events are in both groups (tests assume independence)")
        if self.con_perm.value():
            spatial = self.con_spatial.isChecked()
            if spatial and not has_positions(a):
                notes.append("spatio-temporal test needs electrode positions → ROI test used")
                spatial = False
            try:
                with self.state._busy("Running cluster permutation test (MNE)…"):
                    stats = E.cluster_contrast(
                        a, b, roi, self.con_perm.value(),
                        tmin=0.0 if self.con_twin.isChecked() else None, spatial=spatial)
            except Exception as e:
                notes.append(f"test failed: {e}")
            if stats and stats.get("excluded"):
                notes.append("flat channels left out of the test: "
                             + ", ".join(stats["excluded"]))
        self.contrast = {"a": a, "b": b, "labels": (la, lb), "evokeds": evokeds,
                         "roi": roi, "stats": stats}
        sig = [p for p in (stats or {}).get("p_values", []) if p < 0.05]
        test = ""
        if stats:
            test = (f"spatio-temporal cluster test ({len(stats['ch_names'])} ch)"
                    if stats["spatial"] else "cluster test on ROI mean")
        roi_txt = "all channels" if len(roi) == len(a.ch_names) else ", ".join(roi)
        self.con_info.setText(
            f"{la}: {len(a)} epochs  ·  {lb}: {len(b)} epochs  ·  ROI: {roi_txt}"
            + (f"  ·  {test}: {len(stats['p_values'])} clusters, {len(sig)} with p < 0.05"
               if stats else "") + ("\n" + "; ".join(notes) if notes else ""))
        self._plot_contrast()

    def _plot_contrast(self):
        self.con_plot.clear()
        self.con_table.clear()
        self.con_table.setRowCount(0)
        c = self.contrast
        if c is None:
            return
        la, lb = c["labels"]
        roi = c["roi"]
        stats = c["stats"]
        t = c["evokeds"][la].times
        curves = {}
        for i, lab in enumerate((la, lb)):
            ev = c["evokeds"][lab]
            y = ev.copy().pick(roi).data.mean(0) * 1e6
            ep = c["a"] if i == 0 else c["b"]
            sem = ep.get_data(picks=roi).mean(1).std(0) / np.sqrt(len(ep)) * 1e6
            col = THEME.s(i)
            lo = self.con_plot.plot(t, y - sem, pen=None)
            hi = self.con_plot.plot(t, y + sem, pen=None)
            self.con_plot.addItem(pg.FillBetweenItem(lo, hi, brush=THEME.s(i, 50)))
            curves[lab] = self.con_plot.plot(t, y, pen=pg.mkPen(col, width=2.2),
                                             name=f"{lab} (n={len(ep)})")
        diff_label = f"{la} − {lb}"
        d = c["evokeds"][diff_label].copy().pick(roi).data.mean(0) * 1e6
        self.con_plot.plot(t, d, pen=pg.mkPen(THEME.fg, width=1.6, style=QtCore.Qt.DashLine),
                           name=diff_label)
        self.con_plot.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(THEME.muted)))
        rows = []
        if stats:
            st = stats["times"]
            for k, (mask, p) in enumerate(zip(stats["clusters"], stats["p_values"])):
                tmask = mask.any(axis=1) if mask.ndim == 2 else mask
                if not tmask.any():
                    continue
                t0, t1 = float(st[tmask][0]), float(st[tmask][-1])
                n_ch = int(mask.any(axis=0).sum()) if mask.ndim == 2 else len(roi)
                rows.append({"cluster": k + 1, "tmin (s)": t0, "tmax (s)": t1,
                             "channels": n_ch, "p": float(p)})
                if p < 0.05:
                    region = pg.LinearRegionItem((t0, t1), movable=False,
                                                 brush=THEME.rgba("accent", 55),
                                                 pen=pg.mkPen(None))
                    region.setZValue(-20)
                    self.con_plot.addItem(region)
                    lab = pg.TextItem(f"p={p:.3f}", color=THEME.accent, anchor=(0.5, 1))
                    lab.setPos((t0 + t1) / 2, self.con_plot.viewRange()[1][1])
                    self.con_plot.addItem(lab)
        if rows:
            df = pd.DataFrame(rows).sort_values("p")
            fill_table(self.con_table, df, fmt="{:.4g}")

    def contrast_compare(self):
        c = self.contrast
        if c is None:
            return
        la, lb = c["labels"]
        evs = {la: c["evokeds"][la], lb: c["evokeds"][lb]}
        roi = c["roi"]
        try:
            mne.viz.plot_compare_evokeds(
                evs, picks=roi, combine="mean" if len(roi) > 1 else None,
                colors={la: THEME.group(0), lb: THEME.group(1)}, show=True,
                title=f"{la} vs {lb} — {', '.join(roi) if len(roi) <= 6 else 'ROI'}")
        except Exception as e:
            self._warn("Compare evokeds", e)

    def contrast_topo(self):
        c = self.contrast
        if c is None:
            return
        la, lb = c["labels"]
        diff = c["evokeds"][f"{la} − {lb}"]
        if not has_positions(diff):
            QtWidgets.QMessageBox.information(self, "Topomaps", NO_POS)
            return
        try:
            stats = c["stats"]
            times = "peaks"
            if stats and len(stats["p_values"]):
                best = int(np.argmin(stats["p_values"]))
                m = stats["clusters"][best]
                tm = m.any(axis=1) if m.ndim == 2 else m
                times = float(stats["times"][tm].mean())
            diff.plot_joint(times=times, title=f"{la} − {lb}", show=True)
        except Exception as e:
            self._warn("Topomaps", e)

    def contrast_export(self):
        c = self.contrast
        if c is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export contrast to folder")
        if not d:
            return
        out = Path(d)
        la, lb = c["labels"]
        safe = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)  # noqa: E731
        mne.write_evokeds(out / f"contrast_{safe(la)}_vs_{safe(lb)}-ave.fif",
                          list(c["evokeds"].values()), overwrite=True, verbose="error")
        c["a"].save(out / f"{safe(la)}-epo.fif", overwrite=True, verbose="error")
        c["b"].save(out / f"{safe(lb)}-epo.fif", overwrite=True, verbose="error")
        if c["stats"]:
            rows = []
            st = c["stats"]
            for k, (mask, p) in enumerate(zip(st["clusters"], st["p_values"])):
                tm = mask.any(axis=1) if mask.ndim == 2 else mask
                if tm.any():
                    rows.append({"cluster": k + 1, "tmin": st["times"][tm][0],
                                 "tmax": st["times"][tm][-1], "p": p})
            pd.DataFrame(rows).to_csv(out / "clusters.csv", index=False)
        (out / "contrast.txt").write_text(
            f"A = {la}: {self.con_a.describe()}\nB = {lb}: {self.con_b.describe()}\n"
            f"ROI: {', '.join(c['roi'])}\nparams: {self.con_params.kwargs()}\n")
        self.state.statusBar().showMessage(f"Contrast exported to {out}", 5000)
