"""Analysis tabs: gaze statistics/AOIs, spectrum, fixation- and event-related potentials."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import eeg as E
from .. import gaze as G


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


def _spin(lo, hi, val, step=0.05, dec=3, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setSingleStep(step)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class AnalysisPanel(QtWidgets.QTabWidget):
    """``state`` is the main window, providing rec, display_raw, gaze_events, screen."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.evoked = None
        self.epochs = None
        self.psd = None
        self._build_gaze_tab()
        self._build_psd_tab()
        self._build_erp_tab()

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
                                "to remove it. Stats use detected fixations.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        b2.addWidget(hint)
        top.addWidget(box2, 1)
        # fixation duration / saccade amplitude histograms
        self.hist = pg.GraphicsLayoutWidget()
        self.hist.setBackground("w")
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
            for p in (self.hist_fix, self.hist_sac, self.hist_main):
                p.clear()
            return
        fill_table(self.summary_table, G.summary_table(ev, rec.gaze.duration))
        self.update_aoi_stats()
        for p in (self.hist_fix, self.hist_sac, self.hist_main):
            p.clear()
        if len(ev.fixations):
            y, x = np.histogram(ev.fixations["duration"] * 1000, bins=40)
            self.hist_fix.plot(x, y, stepMode="center", fillLevel=0, brush=(70, 130, 220, 150))
        if len(ev.saccades):
            y, x = np.histogram(ev.saccades["amplitude"], bins=40)
            self.hist_sac.plot(x, y, stepMode="center", fillLevel=0, brush=(240, 140, 30, 150))
            self.hist_main.plot(ev.saccades["amplitude"].to_numpy(),
                                ev.saccades["peak_velocity"].to_numpy(), pen=None,
                                symbol="o", symbolSize=4, symbolBrush=(100, 100, 100, 120),
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
        df = getattr(self, "_aoi_df", None)
        if df is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export AOI stats", "aoi.csv",
                                                      "CSV (*.csv)")
        if fn:
            df.to_csv(fn, index=False)

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
        self.psd_plot = pg.PlotWidget(background="w")
        self.psd_plot.setLabel("bottom", "Frequency", "Hz")
        self.psd_plot.setLabel("left", "Power (dB µV²/Hz)")
        self.psd_plot.showGrid(x=True, y=True, alpha=0.3)
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
        self.psd = E.compute_psd(raw, self.psd_fmax.value(), tmin, tmax)
        data, freqs = self.psd.get_data(return_freqs=True)
        db = 10 * np.log10(np.maximum(data * 1e12, 1e-20))
        self.psd_plot.clear()
        for i, name in enumerate(self.psd.ch_names):
            c = self.psd_plot.plot(freqs, db[i], pen=pg.mkPen((120, 120, 120, 90)))
            c.setToolTip(name)
        self.psd_plot.plot(freqs, db.mean(0), pen=pg.mkPen((200, 30, 30), width=2.5))

    def psd_topomap(self):
        if self.psd is None:
            self.compute_psd()
        if self.psd is not None:
            try:
                self.psd.plot_topomap(show=True)
            except Exception as e:  # no montage etc.
                QtWidgets.QMessageBox.warning(self, "Topomap", str(e))

    # ================================================================ ERP tab
    def _build_erp_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QHBoxLayout()
        self.erp_source = QtWidgets.QComboBox()
        self.erp_tmin = _spin(-5, 0, -0.2, 0.05, 3, " s")
        self.erp_tmax = _spin(0.01, 10, 0.6, 0.05, 3, " s")
        self.erp_baseline = QtWidgets.QCheckBox("Baseline (tmin..0)")
        self.erp_baseline.setChecked(True)
        self.erp_mindur = _spin(0, 2, 0.1, 0.02, 3, " s")
        self.erp_reject = _spin(0, 2000, 0, 10, 0, " µV")
        self.erp_reject.setSpecialValueText("no rejection")
        btn = QtWidgets.QPushButton("Compute")
        btn.clicked.connect(self.compute_erp)
        joint = QtWidgets.QPushButton("Joint plot (MNE)…")
        joint.clicked.connect(self.erp_joint)
        img = QtWidgets.QPushButton("Epochs image (MNE)…")
        img.clicked.connect(self.erp_image)
        for lab, x in (("Lock to", self.erp_source), ("tmin", self.erp_tmin),
                       ("tmax", self.erp_tmax), (None, self.erp_baseline),
                       ("min fix dur", self.erp_mindur), ("reject", self.erp_reject)):
            if lab:
                form.addWidget(QtWidgets.QLabel(lab))
            form.addWidget(x)
        form.addWidget(btn)
        form.addWidget(joint)
        form.addWidget(img)
        form.addStretch()
        lay.addLayout(form)
        self.erp_info = QtWidgets.QLabel("")
        self.erp_info.setStyleSheet("color: gray")
        lay.addWidget(self.erp_info)
        self.erp_plot = pg.PlotWidget(background="w")
        self.erp_plot.setLabel("bottom", "Time", "s")
        self.erp_plot.setLabel("left", "Amplitude (µV)")
        self.erp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.erp_plot.addLegend()
        lay.addWidget(self.erp_plot)
        self.addTab(w, "Fixation / event-related potentials")

    def refresh_sources(self):
        cur = self.erp_source.currentText()
        self.erp_source.clear()
        self.erp_source.addItem("fixation onset")
        rec = self.state.rec
        if rec is not None:
            desc = rec.all_events()["description"].astype(str)
            names = set(desc.unique())
            if len(names) > 30 or desc.str.contains("|", regex=False).any():
                names |= set(desc.str.split("|").str[0].unique())  # group by prefix
            for d in sorted(names, key=lambda x: (len(x) > 40, x))[:300]:
                self.erp_source.addItem(d)
        i = self.erp_source.findText(cur)
        if i >= 0:
            self.erp_source.setCurrentIndex(i)

    def compute_erp(self):
        raw, ev, rec = self.state.display_raw, self.state.gaze_events, self.state.rec
        if raw is None:
            QtWidgets.QMessageBox.information(self, "ERP", "No EEG loaded.")
            return
        src = self.erp_source.currentText()
        base = (None, 0) if self.erp_baseline.isChecked() else None
        try:
            if src == "fixation onset":
                if ev is None:
                    raise ValueError("Run gaze event detection first.")
                self.epochs = E.fixation_related_epochs(
                    raw, ev.fixations, self.erp_tmin.value(), self.erp_tmax.value(),
                    self.erp_mindur.value(), base, self.erp_reject.value() or None)
            else:
                self.epochs = E.event_related_epochs(raw, rec.all_events(), src,
                                                     self.erp_tmin.value(),
                                                     self.erp_tmax.value(), base)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "ERP", str(e))
            return
        if not len(self.epochs):
            QtWidgets.QMessageBox.warning(self, "ERP", "All epochs were rejected.")
            return
        self.evoked = self.epochs.average()
        self.erp_info.setText(f"{src}: {len(self.epochs)} epochs averaged "
                              f"({len(self.epochs.drop_log) - len(self.epochs)} dropped), "
                              f"{len(self.evoked.ch_names)} channels")
        self.erp_plot.clear()
        t = self.evoked.times
        data = self.evoked.data * 1e6
        for i, name in enumerate(self.evoked.ch_names):
            c = self.erp_plot.plot(t, data[i], pen=pg.mkPen((140, 140, 140, 110)))
            c.setToolTip(name)
        gfp = data.std(0)
        self.erp_plot.plot(t, gfp, pen=pg.mkPen((30, 30, 30), width=2.5), name="GFP")
        for name, color in (("Oz", (200, 30, 30)), ("Pz", (30, 90, 200)),
                            ("Cz", (40, 150, 60))):
            if name in self.evoked.ch_names:
                i = self.evoked.ch_names.index(name)
                self.erp_plot.plot(t, data[i], pen=pg.mkPen(color, width=2), name=name)
        self.erp_plot.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen("k", width=1)))

    def erp_joint(self):
        if self.evoked is None:
            self.compute_erp()
        if self.evoked is not None:
            try:
                self.evoked.plot_joint(show=True)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Joint plot", str(e))

    def erp_image(self):
        if self.epochs is None:
            self.compute_erp()
        if self.epochs is not None:
            picks = [c for c in ("Oz", "Pz", "Cz") if c in self.epochs.ch_names][:1] or [0]
            self.epochs.plot_image(picks=picks, show=True)
