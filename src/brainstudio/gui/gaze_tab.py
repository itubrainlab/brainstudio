"""Gaze tab: oculomotor statistics, histograms and PyGaze AOIs."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets

from .. import gaze as G
from .theme import THEME
from .widgets import fill_table, hint


class GazeTab(QtWidgets.QWidget):
    """``state`` is the main window (rec, gaze_events, screen)."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._aoi_df = None
        lay = QtWidgets.QVBoxLayout(self)
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
        b2.addWidget(hint("AOIs are drawn in the Viewer's screen view: drag/resize them there, "
                          "right-click to remove. Fixations get an ‘aoi’ column usable in "
                          "event filters (EEG ▸ Epoching)."))
        top.addWidget(box2, 1)
        self.hist = pg.GraphicsLayoutWidget()
        self.hist.setBackground(THEME.bg)
        self.hist_fix = self.hist.addPlot(title="Fixation duration (ms)")
        self.hist_sac = self.hist.addPlot(title="Saccade amplitude (px)")
        self.hist_main = self.hist.addPlot(title="Main sequence")
        self.hist_main.setLabel("bottom", "amplitude (px)")
        self.hist_main.setLabel("left", "peak velocity (px/s)")
        lay.addWidget(self.hist, 1)

    def update_gaze_stats(self):
        ev, rec = self.state.gaze_events, self.state.rec
        if ev is None or rec is None or not rec.has_gaze:
            self.summary_table.clear()
            self.aoi_table.clear()
        else:
            fill_table(self.summary_table, G.summary_table(ev, rec.gaze.duration))
        self.update_aoi_stats()
        self._plot()

    def restyle(self):
        self.hist.setBackground(THEME.bg)
        self._plot()

    def _plot(self):
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
