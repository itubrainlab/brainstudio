"""One colour scheme for the whole app, derived from the Qt palette.

``THEME`` always describes the current scheme. Charts read it at draw time;
``apply_theme`` updates pyqtgraph defaults, matplotlib rcParams (so MNE's own
figures match) and restyles existing pyqtgraph widgets. Call it again when the
system switches between light and dark.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib as mpl
import pyqtgraph as pg
from cycler import cycler
from PySide6 import QtGui, QtWidgets

# Categorical series colours, tuned per scheme for contrast on the background
# (Okabe–Ito based: colour-blind safe). Order: blue, orange, green, vermillion,
# purple, sky, yellow, grey.
_SERIES_LIGHT = ["#0b6fb8", "#d97706", "#0f8a5f", "#c2410c", "#7c3aed", "#0891b2",
                 "#a16207", "#6b7280"]
_SERIES_DARK = ["#5aa9e6", "#f5a524", "#34d399", "#fb7a54", "#b794f6", "#4fd1e8",
                "#facc15", "#9ca3af"]


@dataclass
class Theme:
    dark: bool = False
    bg: str = "#ffffff"  # chart background (Qt Base)
    panel: str = "#f4f4f5"  # surrounding window (Qt Window)
    fg: str = "#1f2328"  # text / axes
    muted: str = "#6b7280"  # secondary text, faint traces
    grid: str = "#d0d4da"
    trace: str = "#1f2328"  # EEG traces
    accent: str = "#d97706"  # cursor, highlights (Qt Highlight)
    event: str = "#c2410c"
    series: list[str] = field(default_factory=lambda: list(_SERIES_LIGHT))
    screen_bg: str = "#18181b"  # the stimulus display area

    # ---- helpers returning (r, g, b, a) tuples for pyqtgraph
    def rgba(self, name_or_hex: str, alpha: int = 255):
        c = QtGui.QColor(getattr(self, name_or_hex, name_or_hex))
        return (c.red(), c.green(), c.blue(), alpha)

    def s(self, i: int, alpha: int = 255):
        """i-th series colour."""
        return self.rgba(self.series[i % len(self.series)], alpha)

    def group(self, i: int) -> str:
        """Hex colour of contrast group i (shared by app plots and MNE figures)."""
        return self.series[i % len(self.series)]


THEME = Theme()


def _hex(c: QtGui.QColor) -> str:
    return c.name(QtGui.QColor.HexRgb)


def _mix(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    return QtGui.QColor(int(a.red() + (b.red() - a.red()) * t),
                        int(a.green() + (b.green() - a.green()) * t),
                        int(a.blue() + (b.blue() - a.blue()) * t))


def theme_from_palette(pal: QtGui.QPalette) -> Theme:
    base = pal.color(QtGui.QPalette.Base)
    window = pal.color(QtGui.QPalette.Window)
    text = pal.color(QtGui.QPalette.WindowText)
    dark = base.lightness() < 128
    hl = pal.color(QtGui.QPalette.Highlight)
    series = list(_SERIES_DARK if dark else _SERIES_LIGHT)
    return Theme(
        dark=dark,
        bg=_hex(base),
        panel=_hex(window),
        fg=_hex(text),
        muted=_hex(_mix(base, text, 0.55)),
        grid=_hex(_mix(base, text, 0.18)),
        trace=_hex(_mix(base, text, 0.85)),
        accent=_hex(hl) if hl.saturation() > 60 else series[1],
        event=series[3],
        series=series,
        screen_bg="#0e0e11" if dark else "#27272a",
    )


def apply_matplotlib(t: Theme) -> None:
    mpl.rcParams.update({
        "figure.facecolor": t.bg, "axes.facecolor": t.bg, "savefig.facecolor": t.bg,
        "axes.edgecolor": t.muted, "axes.labelcolor": t.fg, "text.color": t.fg,
        "xtick.color": t.fg, "ytick.color": t.fg, "grid.color": t.grid,
        "legend.facecolor": t.bg, "legend.edgecolor": t.grid, "legend.labelcolor": t.fg,
        "axes.prop_cycle": cycler(color=t.series),
        "lines.color": t.trace, "patch.edgecolor": t.muted,
        "figure.edgecolor": t.bg, "boxplot.flierprops.markeredgecolor": t.fg,
    })


def style_plot_item(p: pg.PlotItem, t: Theme) -> None:
    for name in ("left", "bottom", "right", "top"):
        ax = p.getAxis(name)
        ax.setPen(pg.mkPen(t.muted))
        ax.setTextPen(pg.mkPen(t.fg))
        ax.setStyle(tickTextOffset=4)
        if ax.grid is not False:
            ax.setGrid(60 if t.dark else 45)
    if p.titleLabel is not None and p.titleLabel.text:
        p.setTitle(p.titleLabel.text, color=t.fg)
    if p.legend is not None:
        p.legend.setBrush(pg.mkBrush(t.rgba("bg", 210)))
        p.legend.setPen(pg.mkPen(t.grid))
        p.legend.setLabelTextColor(t.fg)


def style_widget(w, t: Theme) -> None:
    """Restyle a pyqtgraph PlotWidget / GraphicsLayoutWidget in place."""
    if hasattr(w, "_theme_bg"):  # widgets that keep their own background
        w.setBackground(getattr(t, w._theme_bg))
    else:
        w.setBackground(t.bg)
    items = []
    if isinstance(w, pg.PlotWidget):
        items = [w.getPlotItem()]
    elif isinstance(w, pg.GraphicsLayoutWidget):
        items = [i for i in w.ci.items if isinstance(i, pg.PlotItem)]
    for p in items:
        style_plot_item(p, t)


def apply_theme(app: QtWidgets.QApplication, root: QtWidgets.QWidget | None = None) -> Theme:
    new = theme_from_palette(app.palette())
    THEME.__dict__.update(new.__dict__)
    pg.setConfigOptions(background=THEME.bg, foreground=THEME.fg)
    apply_matplotlib(THEME)
    if root is not None:
        for w in root.findChildren(pg.GraphicsView):
            if isinstance(w, (pg.PlotWidget, pg.GraphicsLayoutWidget)):
                style_widget(w, THEME)
    return THEME
