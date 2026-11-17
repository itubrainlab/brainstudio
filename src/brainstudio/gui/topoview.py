"""Live MNE scalp topography of the EEG around the cursor time."""

from __future__ import annotations

import mne
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets


class TopoView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(3, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.enabled = QtWidgets.QCheckBox("Topomap at cursor (±")
        self.enabled.setChecked(True)
        self.half = QtWidgets.QSpinBox()
        self.half.setRange(0, 500)
        self.half.setValue(25)
        self.half.setSuffix(" ms)")
        row.addWidget(self.enabled)
        row.addWidget(self.half)
        row.addStretch()
        lay.addLayout(row)
        lay.addWidget(self.canvas, 1)
        self.raw: mne.io.BaseRaw | None = None
        self._t = 0.0
        self._timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self._draw)
        self.enabled.toggled.connect(lambda _: self._draw())
        self.half.valueChanged.connect(lambda _: self.request(self._t))
        self._picks = None
        self._info = None

    def set_raw(self, raw: mne.io.BaseRaw | None):
        self.raw = raw
        self._picks = self._info = None
        if raw is not None:
            picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            info = mne.pick_info(raw.info, picks)
            pos = np.array([ch["loc"][:3] for ch in info["chs"]])
            ok = np.isfinite(pos).all(1) & (np.abs(pos).sum(1) > 0)
            if ok.sum() >= 4:
                self._picks = picks[ok]
                self._info = mne.pick_info(raw.info, self._picks)
        self.request(self._t)

    def request(self, t: float):
        """Throttled redraw (keeps up during playback without flooding)."""
        self._t = t
        if not self._timer.isActive():
            self._timer.start()

    def _draw(self):
        self.ax.clear()
        self.ax.set_axis_off()
        if not self.enabled.isChecked() or self.raw is None:
            self.canvas.draw_idle()
            return
        if self._info is None:
            self.ax.text(0.5, 0.5, "No EEG channel positions", ha="center", va="center",
                         transform=self.ax.transAxes, color="gray")
            self.canvas.draw_idle()
            return
        sf = self.raw.info["sfreq"]
        half = self.half.value() / 1000.0
        i0 = int(np.clip((self._t - half) * sf, 0, len(self.raw.times) - 1))
        i1 = int(np.clip((self._t + half) * sf, i0 + 1, len(self.raw.times)))
        data = self.raw.get_data(picks=self._picks, start=i0, stop=i1).mean(1) * 1e6
        data = data - data.mean()
        lim = max(np.abs(data).max(), 1e-3)
        mne.viz.plot_topomap(data, self._info, axes=self.ax, show=False, contours=4,
                             vlim=(-lim, lim), cmap="RdBu_r", sensors=True)
        self.ax.set_title(f"{self._t:.3f} s   (±{lim:.1f} µV, avg ref)", fontsize=9)
        self.canvas.draw_idle()
