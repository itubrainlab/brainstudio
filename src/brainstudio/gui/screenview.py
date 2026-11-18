"""Screen-space gaze view: stimulus/heatmap, scanpath, live gaze and AOIs."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore

from .. import gaze as G
from .theme import _SERIES_DARK, THEME

# The display area is always drawn dark (like a monitor), so on-screen items
# use the dark-scheme series colours in both light and dark app themes.
ON_SCREEN = _SERIES_DARK


def _c(i: int, alpha: int = 255):
    q = pg.mkColor(ON_SCREEN[i % len(ON_SCREEN)])
    q.setAlpha(alpha)
    return q


class ScreenView(pg.PlotWidget):
    aoisChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, background=THEME.bg)
        self.plotItem.invertY(True)
        self.plotItem.setAspectLocked(True)
        self.plotItem.showGrid(x=False, y=False)
        self.plotItem.setLabel("bottom", "x", "px")
        self.plotItem.setLabel("left", "y", "px")
        self.plotItem.hideButtons()
        self.screen = (1920, 1080)
        self.frame = pg.QtWidgets.QGraphicsRectItem(0, 0, *self.screen)
        self.frame.setZValue(-30)
        self.addItem(self.frame)
        self.image = pg.ImageItem()
        self.image.setZValue(-20)
        self.addItem(self.image)
        self.heat = pg.ImageItem()
        self.heat.setZValue(-10)
        self.heat.setOpacity(0.7)
        self.addItem(self.heat)
        self.scan_line = self.plot(pen=pg.mkPen((235, 235, 240, 130), width=1.5))
        self.fix_scatter = pg.ScatterPlotItem(pen=pg.mkPen((235, 235, 240), width=1))
        self.addItem(self.fix_scatter)
        self.trail = self.plot(pen=pg.mkPen(_c(5, 170), width=2))
        self.point = pg.ScatterPlotItem(size=18, pen=pg.mkPen((245, 245, 245), width=2),
                                        brush=pg.mkBrush(_c(3, 220)))
        self.point.setZValue(50)
        self.addItem(self.point)
        self.fix_labels: list[pg.TextItem] = []
        self.aois: dict[str, pg.ROI] = {}
        self.show_scanpath = True
        self.show_heatmap = False
        self.trail_len = 0.5
        self.set_screen(self.screen)
        self.restyle()

    def restyle(self):
        self._theme_bg = "bg"
        self.setBackground(THEME.bg)
        self.frame.setBrush(pg.mkBrush(THEME.screen_bg))
        self.frame.setPen(pg.mkPen(THEME.muted, width=2))

    # ----------------------------------------------------------------- setup
    def set_screen(self, size):
        self.screen = tuple(int(s) for s in size)
        self.frame.setRect(0, 0, *self.screen)
        self.reset_view()

    def reset_view(self):
        # Set both axes in one call: with a locked aspect ratio, separate
        # X/Y calls fight each other and end up zoomed on a corner.
        self.plotItem.disableAutoRange()
        self.plotItem.getViewBox().setRange(QtCore.QRectF(0, 0, *self.screen), padding=0.03)

    def set_stimulus(self, img: np.ndarray | None):
        if img is None:
            self.image.clear()
            return
        # image arrays are (rows, cols, [rgb]); ImageItem expects (x, y)
        arr = np.swapaxes(img, 0, 1)
        self.image.setImage(arr, autoLevels=img.dtype != np.uint8)
        self.image.setRect(QtCore.QRectF(0, 0, *self.screen))

    def set_heatmap(self, hm: np.ndarray | None):
        if hm is None or not self.show_heatmap or not np.any(hm):
            self.heat.clear()
            return
        lut = pg.colormap.get("inferno").getLookupTable(nPts=256, alpha=True)
        lut[:, 3] = np.clip(np.linspace(0, 400, 256), 0, 230).astype(np.uint8)
        self.heat.setImage(hm.T, lut=lut, levels=(0, hm.max()))
        self.heat.setRect(QtCore.QRectF(0, 0, *self.screen))

    # ----------------------------------------------------------------- dynamic
    def update_view(self, gaze, events, t: float, t0: float, t1: float):
        """Show the scanpath of the window [t0, t1] and gaze at cursor t."""
        for lab in self.fix_labels:
            self.removeItem(lab)
        self.fix_labels = []
        if gaze is None:
            self.point.setData([], [])
            self.trail.setData([], [])
            return
        x, y = gaze.at(t)
        self.point.setData([x] if np.isfinite(x) else [], [y] if np.isfinite(y) else [])
        sl = gaze.window(t - self.trail_len, t)
        self.trail.setData(gaze.x[sl], gaze.y[sl], connect="finite")
        if events is not None and self.show_scanpath and len(events.fixations):
            f = events.fixations
            f = f[(f["offset"] >= t0) & (f["onset"] <= t1)]
            if len(f) > 400:
                f = f.iloc[:400]
            past = f["onset"].to_numpy() <= t
            sizes = np.clip(np.sqrt(f["duration"].to_numpy() * 1000) * 1.6, 6, 60)
            brushes = [pg.mkBrush(_c(1, 150)) if p else pg.mkBrush(160, 160, 170, 70)
                       for p in past]
            self.fix_scatter.setData(f["x"].to_numpy(), f["y"].to_numpy(), size=sizes,
                                     brush=brushes)
            self.scan_line.setData(f["x"].to_numpy(), f["y"].to_numpy())
            if len(f) <= 40:
                for k, (fx, fy) in enumerate(f[["x", "y"]].to_numpy(), start=1):
                    lab = pg.TextItem(str(k), color=(20, 20, 20), anchor=(0.5, 0.5))
                    lab.setPos(fx, fy)
                    self.addItem(lab)
                    self.fix_labels.append(lab)
        else:
            self.fix_scatter.setData([], [])
            self.scan_line.setData([], [])

    # ----------------------------------------------------------------- AOIs
    def add_aoi(self, kind="rectangle", name=None, pos=None, size=None):
        name = name or f"AOI{len(self.aois) + 1}"
        while name in self.aois:
            name += "'"
        color = _c(len(self.aois) + 2)
        size = size or (self.screen[0] / 5, self.screen[1] / 5)
        pos = pos or (self.screen[0] / 2 - size[0] / 2, self.screen[1] / 2 - size[1] / 2)
        pen = pg.mkPen(color, width=2)
        cls = pg.RectROI if kind == "rectangle" else pg.EllipseROI
        roi = cls(pos, size, pen=pen, removable=True)
        roi.aoi_kind = kind
        roi.aoi_name = name
        label = pg.TextItem(name, color=color, anchor=(0, 1))
        label.setParentItem(roi)
        roi.sigRegionChangeFinished.connect(lambda *_: self.aoisChanged.emit())
        roi.sigRemoveRequested.connect(lambda r=roi: self.remove_aoi(r.aoi_name))
        self.addItem(roi)
        self.aois[name] = roi
        self.aoisChanged.emit()
        return name

    def remove_aoi(self, name):
        roi = self.aois.pop(name, None)
        if roi is not None:
            self.removeItem(roi)
            self.aoisChanged.emit()

    def clear_aois(self):
        for name in list(self.aois):
            self.remove_aoi(name)

    def aoi_specs(self) -> dict:
        """AOIs as ``{name: (kind, position, size)}`` in PyGaze's convention
        (rectangles: top-left corner; ellipses: centre)."""
        out = {}
        for name, roi in self.aois.items():
            (x, y), (w, h) = roi.pos(), roi.size()
            if roi.aoi_kind == "rectangle":
                out[name] = ("rectangle", (float(x), float(y)), (int(w), int(h)))
            else:  # PyGaze ellipses are centred
                out[name] = ("ellipse", (float(x + w / 2), float(y + h / 2)), (int(w), int(h)))
        return out

    def pygaze_aois(self) -> dict:
        """Convert ROIs into PyGaze AOI objects."""
        from .. import workflow as W
        return W.make_aois(self.aoi_specs())
