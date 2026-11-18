"""Time-locked plots: stacked EEG traces, gaze/pupil traces, overview bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore

from ..model import GazeEvents
from .theme import THEME


def event_pen():
    return pg.mkPen(THEME.rgba("event", 190), width=1, style=QtCore.Qt.DashLine)


def cursor_pen():
    return pg.mkPen(THEME.accent, width=2)


def overlay_brush(kind: str):
    a = 55 if THEME.dark else 45
    return {"fixations": pg.mkBrush(THEME.s(0, a)),
            "saccades": pg.mkBrush(THEME.s(1, a + 25)),
            "blinks": pg.mkBrush(THEME.rgba("muted", a + 45))}[kind]


def type_color(ch_type: str, bad: bool = False):
    if bad:
        return THEME.rgba("muted", 140)
    idx = {"eog": 0, "ecg": 3, "emg": 2, "misc": 4, "stim": 6}
    return THEME.s(idx[ch_type]) if ch_type in idx else THEME.rgba("trace")


class TimeCursorMixin:
    """Adds a draggable vertical cursor emitting ``cursorMoved(float)``."""

    def _init_cursor(self, plot: pg.PlotItem):
        self.cursor = pg.InfiniteLine(0, angle=90, movable=True, pen=cursor_pen())
        self.cursor.setZValue(100)
        plot.addItem(self.cursor, ignoreBounds=True)
        self.cursor.sigPositionChanged.connect(
            lambda line: self.cursorMoved.emit(float(line.value())))

    def set_cursor(self, t: float):
        self.cursor.blockSignals(True)
        self.cursor.setValue(t)
        self.cursor.blockSignals(False)


class EventLinesMixin:
    def _set_event_lines(self, plot: pg.PlotItem, events: pd.DataFrame, labels=True):
        for item in getattr(self, "_event_items", []):
            plot.removeItem(item)
        self._event_items = []
        if events is None or not len(events):
            return
        for on, desc in events[["onset", "description"]].itertuples(index=False):
            line = pg.InfiniteLine(
                on, angle=90, pen=event_pen(),
                label=str(desc).split("|")[0] if labels else None,
                labelOpts={"position": 0.985, "color": THEME.event,
                           "anchors": [(-0.05, 0.5), (-0.05, 0.5)]})
            plot.addItem(line, ignoreBounds=True)
            self._event_items.append(line)


class SegmentLinesMixin:
    """A labelled line where each recording (subject / session / run) starts."""

    def _set_segment_lines(self, plot: pg.PlotItem, segments: list[dict], labels=True):
        from ..io.multi import segment_label
        for item in getattr(self, "_segment_items", []):
            plot.removeItem(item)
        self._segment_items = []
        if not segments or len(segments) < 2:
            return
        for k, seg in enumerate(segments):
            line = pg.InfiniteLine(
                seg["start"], angle=90, pen=pg.mkPen(THEME.accent, width=2),
                label=segment_label(seg) if labels else None,
                labelOpts={"position": 0.03, "color": THEME.accent,
                           "fill": pg.mkBrush(THEME.rgba("bg", 220)),
                           "anchors": [(0, 1), (0, 1)]})
            line.setZValue(5)
            plot.addItem(line, ignoreBounds=True)
            self._segment_items.append(line)


class GazeOverlayMixin:
    """Shaded spans for fixations/saccades/blinks, rebuilt for the visible window."""

    def _init_overlays(self, plot: pg.PlotItem):
        self._overlay_plot = plot
        self._overlay_items: dict[str, pg.BarGraphItem] = {}
        self.gaze_events: GazeEvents | None = None
        self.overlay_kinds = {"fixations": False, "saccades": True, "blinks": True}

    def _update_overlays(self, t0: float, t1: float):
        for item in self._overlay_items.values():
            self._overlay_plot.removeItem(item)
        self._overlay_items = {}
        if self.gaze_events is None:
            return
        y0, y1 = self._overlay_yrange()
        for kind, on in self.overlay_kinds.items():
            if not on:
                continue
            df = getattr(self.gaze_events, kind)
            if not len(df):
                continue
            sel = df[(df["offset"] >= t0) & (df["onset"] <= t1)]
            if not len(sel) or len(sel) > 3000:
                continue
            item = pg.BarGraphItem(x0=sel["onset"].to_numpy(), width=sel["duration"].to_numpy(),
                                   y0=y0, height=y1 - y0, brush=overlay_brush(kind),
                                   pen=pg.mkPen(None))
            item.setZValue(-10)
            self._overlay_plot.addItem(item, ignoreBounds=True)
            self._overlay_items[kind] = item


# ============================================================================ EEG
class EEGPlot(pg.PlotWidget, TimeCursorMixin, EventLinesMixin, SegmentLinesMixin,
              GazeOverlayMixin):
    """Stacked multichannel EEG traces (butterfly-free, MNE-browser style)."""

    cursorMoved = QtCore.Signal(float)
    windowChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent, background=THEME.bg)
        self.plotItem.setLabel("bottom", "Time", "s")
        self.plotItem.showGrid(x=True, y=False, alpha=0.2)
        self.plotItem.setMouseEnabled(x=True, y=True)
        self.plotItem.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.plotItem.hideButtons()
        self.data: np.ndarray | None = None
        self.times: np.ndarray | None = None
        self.names: list[str] = []
        self.types: list[str] = []
        self.bads: set[str] = set()
        self.scale_uv = 50.0  # µV per channel spacing
        self.n_visible = 20
        self.curves: list[pg.PlotDataItem] = []
        self._init_cursor(self.plotItem)
        self._init_overlays(self.plotItem)
        self.plotItem.sigXRangeChanged.connect(self._on_xrange)
        self._busy = False
        self._redraw_pending = False

    # ------------------------------------------------------------------ data
    def set_data(self, data: np.ndarray, times: np.ndarray, names, types, bads=()):
        self.data, self.times = data, times
        self.names, self.types, self.bads = list(names), list(types), set(bads)
        for c in self.curves:
            self.removeItem(c)
        self.curves = []
        for i, (n, t) in enumerate(zip(self.names, self.types)):
            c = pg.PlotDataItem(pen=pg.mkPen(type_color(t, n in self.bads), width=1))
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
            self.addItem(c)
            self.curves.append(c)
        ax = self.plotItem.getAxis("left")
        ax.setTicks([[(-i, n) for i, n in enumerate(self.names)]])
        n = len(self.names)
        vb = self.plotItem.getViewBox()
        vb.setLimits(xMin=times[0] - 1, xMax=times[-1] + 1, yMin=-n - 1, yMax=2,
                     minXRange=0.05)
        self.set_visible_channels(self.n_visible)
        self._redraw()

    def restyle(self):
        self.cursor.setPen(cursor_pen())
        for c, n, t in zip(self.curves, self.names, self.types):
            c.setPen(pg.mkPen(type_color(t, n in self.bads), width=1))
        for line in getattr(self, "_event_items", []):
            line.setPen(event_pen())
            if line.label is not None:
                line.label.setColor(THEME.event)
        self._redraw()

    def update_values(self, data: np.ndarray):
        """Replace trace values (e.g. after filtering) keeping layout."""
        self.data = data
        self._redraw()

    def set_visible_channels(self, n: int):
        self.n_visible = max(1, n)
        self.plotItem.setYRange(-min(self.n_visible, len(self.names)) + 0.5, 0.5 + 0.5,
                                padding=0)

    def set_scale(self, uv: float):
        self.scale_uv = max(uv, 1e-3)
        self._redraw()

    def window(self) -> tuple[float, float]:
        (t0, t1), _ = self.plotItem.viewRange()
        return t0, t1

    def set_window(self, t0: float, t1: float):
        self.plotItem.setXRange(t0, t1, padding=0)

    # ------------------------------------------------------------------ drawing
    def _on_xrange(self, *_):
        # range changes come in bursts (layout, linked views): redraw once per burst
        if not self._redraw_pending:
            self._redraw_pending = True
            QtCore.QTimer.singleShot(0, self._apply_xrange)

    def _apply_xrange(self):
        self._redraw_pending = False
        if self._busy:
            return
        self._busy = True
        try:
            self._redraw()
            t0, t1 = self.window()
            self.windowChanged.emit(t0, t1)
        finally:
            self._busy = False

    def _overlay_yrange(self):
        return -len(self.names) - 0.5, 0.5

    def _redraw(self):
        if self.data is None:
            return
        t0, t1 = self.window()
        sf = 1.0 / (self.times[1] - self.times[0])
        i0 = max(int((t0 - self.times[0]) * sf) - 2, 0)
        i1 = min(int((t1 - self.times[0]) * sf) + 3, len(self.times))
        if i1 <= i0:
            return
        seg = self.data[:, i0:i1]
        t = self.times[i0:i1]
        # remove per-channel mean of visible window so traces stay centred
        seg = seg - np.nanmean(seg, axis=1, keepdims=True)
        scale = 1.0 / (self.scale_uv * 1e-6)
        for i, c in enumerate(self.curves):
            c.setData(t, seg[i] * scale * 0.5 - i)
        self._update_overlays(t0, t1)


# ============================================================================ gaze
class GazePlot(pg.GraphicsLayoutWidget, TimeCursorMixin, EventLinesMixin, SegmentLinesMixin,
               GazeOverlayMixin):
    """Gaze x/y (px) and pupil traces."""

    cursorMoved = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground(THEME.bg)
        self.pos_plot = self.addPlot(row=0, col=0)
        self.pupil_plot = self.addPlot(row=1, col=0)
        self.ci.layout.setRowStretchFactor(0, 3)
        self.ci.layout.setRowStretchFactor(1, 2)
        for p in (self.pos_plot, self.pupil_plot):
            p.showGrid(x=True, y=True, alpha=0.2)
            p.hideButtons()
            p.getAxis("left").setWidth(55)
        self.pos_plot.setLabel("left", "Gaze", "px")
        self.pupil_plot.setLabel("left", "Pupil")
        self.pupil_plot.setLabel("bottom", "Time", "s")
        self.pupil_plot.setXLink(self.pos_plot)
        self.pos_plot.addLegend(offset=(-10, 5))
        self.cx = self.pos_plot.plot(name="x")
        self.cy = self.pos_plot.plot(name="y")
        self.cp = self.pupil_plot.plot()
        self.cv = None
        for c in (self.cx, self.cy, self.cp):
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
        self._init_cursor(self.pos_plot)
        self.cursor2 = pg.InfiniteLine(0, angle=90, movable=False, pen=cursor_pen())
        self.pupil_plot.addItem(self.cursor2, ignoreBounds=True)
        self.cursor.sigPositionChanged.connect(lambda l: self.cursor2.setValue(l.value()))
        self._init_overlays(self.pos_plot)
        self._ylim = (0, 1920)
        self.restyle()

    def restyle(self):
        self.cx.setPen(pg.mkPen(THEME.s(0), width=1.3))
        self.cy.setPen(pg.mkPen(THEME.s(1), width=1.3))
        self.cp.setPen(pg.mkPen(THEME.s(2), width=1.3))
        self.cursor.setPen(cursor_pen())
        self.cursor2.setPen(cursor_pen())
        for line in getattr(self, "_event_items", []):
            line.setPen(event_pen())
        self.refresh_overlays()

    def set_cursor(self, t: float):
        super().set_cursor(t)
        self.cursor2.setValue(t)

    def set_gaze(self, g, pupil=None):
        if g is None:
            for c in (self.cx, self.cy, self.cp):
                c.setData([], [])
            return
        self.cx.setData(g.times, g.x, connect="finite")
        self.cy.setData(g.times, g.y, connect="finite")
        p = pupil if pupil is not None else g.pupil
        if p is not None:
            self.cp.setData(g.times, p, connect="finite")
            finite = p[np.isfinite(p)]
            if len(finite):
                lo, hi = np.percentile(finite, [0.5, 99.5])
                self.pupil_plot.setYRange(lo, hi)
        else:
            self.cp.setData([], [])
        self.pupil_plot.setVisible(p is not None)
        self._ylim = (0, max(g.screen_size))
        self.pos_plot.setYRange(0, max(g.screen_size), padding=0.02)

    def _overlay_yrange(self):
        return self._ylim

    def refresh_overlays(self):
        (t0, t1), _ = self.pos_plot.viewRange()
        self._update_overlays(t0, t1)


# ============================================================================ overview
class OverviewBar(pg.PlotWidget):
    """Full-recording strip with a draggable window region."""

    regionChanged = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent, background=THEME.bg)
        self.setFixedHeight(70)
        self.plotItem.hideAxis("left")
        self.plotItem.setMouseEnabled(x=False, y=False)
        self.plotItem.hideButtons()
        self.plotItem.setMenuEnabled(False)
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)
        self.addItem(self.region)
        self.region.sigRegionChanged.connect(self._emit)
        self.validity = self.plot(pen=None, fillLevel=0, stepMode="center")
        self.ticks: list = []
        self._spans: list = []
        self._block = False
        self.restyle()

    def restyle(self):
        self.region.setBrush(pg.mkBrush(THEME.rgba("accent", 60)))
        for ln in self.region.lines:
            ln.setPen(pg.mkPen(THEME.accent, width=1.5))
        self.validity.setBrush(pg.mkBrush(THEME.s(0, 110)))
        for t in self.ticks:
            t.setBrush(pg.mkBrush(THEME.event))

    def set_subjects(self, spans: list[dict]):
        """Subjects as alternately shaded spans with their name (click one to jump)."""
        for it in self._spans:
            self.removeItem(it)
        self._spans = []
        if len(spans) < 2:
            return
        for k, sp in enumerate(spans):
            reg = pg.LinearRegionItem((sp["start"], sp["end"]), movable=False,
                                      brush=THEME.rgba("fg", 18 if k % 2 else 6),
                                      pen=pg.mkPen(THEME.rgba("accent", 160)))
            reg.setZValue(-20)
            self.addItem(reg)
            txt = pg.TextItem(sp["label"], color=THEME.fg, anchor=(0, 0.5),
                              fill=pg.mkBrush(THEME.rgba("bg", 170)))
            txt.setPos(sp["start"], 0.45)
            txt.setZValue(-10)
            self.addItem(txt)
            self._spans += [reg, txt]

    def set_recording(self, duration: float, events: pd.DataFrame, gaze=None):
        self.plotItem.setXRange(0, max(duration, 1e-3), padding=0.01)
        self.plotItem.setYRange(0, 1.05, padding=0)
        self.region.setBounds((0, duration))
        for t in self.ticks:
            self.removeItem(t)
        self.ticks = []
        if events is not None and len(events):
            tick = pg.ScatterPlotItem(events["onset"].to_numpy(), np.full(len(events), 0.9),
                                      symbol="t", size=7, brush=THEME.event, pen=None)
            self.addItem(tick)
            self.ticks.append(tick)
        if gaze is not None and gaze.n_samples:
            # fraction of valid gaze samples in 1 s bins
            edges = np.arange(0, np.ceil(duration) + 1)
            valid = np.isfinite(gaze.x).astype(float)
            cnt, _ = np.histogram(gaze.times, edges)
            ok, _ = np.histogram(gaze.times, edges, weights=valid)
            frac = np.divide(ok, cnt, out=np.zeros_like(ok), where=cnt > 0) * 0.75
            self.validity.setData(edges, frac)
        else:
            self.validity.setData([0, 1], [0])

    def set_region(self, t0, t1):
        self._block = True
        self.region.setRegion((t0, t1))
        self._block = False

    def _emit(self):
        if not self._block:
            t0, t1 = self.region.getRegion()
            self.regionChanged.emit(t0, t1)

