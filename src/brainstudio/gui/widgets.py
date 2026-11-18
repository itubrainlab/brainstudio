"""Small shared widgets and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .theme import THEME


def fill_table(table: QtWidgets.QTableWidget, df: pd.DataFrame, fmt="{:.3g}"):
    table.clear()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for i, row in enumerate(df.itertuples(index=False)):
        for j, v in enumerate(row):
            if isinstance(v, (bool, np.bool_)):
                txt = "✓" if v else ""
            elif isinstance(v, (float, np.floating)):
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
    return len(locs) > 0 and int((np.isfinite(locs).all(1) &
                                  (np.abs(locs).sum(1) > 0)).sum()) >= 4


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
    for ax in ("left", "bottom"):  # keep µV/s values as they are (no ×0.001 rescaling)
        p.getAxis(ax).enableAutoSIPrefix(False)
    return p


def parse_channels(text: str, available: list[str]) -> tuple[list[str], list[str]]:
    """Comma/space separated names → (found, missing), case-insensitive."""
    wanted = [c for c in text.replace(";", ",").replace(",", " ").split() if c]
    lower = {c.lower(): c for c in available}
    found = [lower[c.lower()] for c in wanted if c.lower() in lower]
    missing = [c for c in wanted if c.lower() not in lower]
    return found, missing


def hint(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setEnabled(False)
    return lab


def run_button(text: str) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.setObjectName("Primary")  # brAIn lab style: blue primary action
    b.setMinimumHeight(28)
    return b


class StageHeader(QtWidgets.QFrame):
    """Banner at the top of a pipeline stage: what the input is and whether the
    results are current, with a button to (re)run the stage."""

    def __init__(self, title: str, run_text: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        self.title = QtWidgets.QLabel(f"<b>{title}</b>")
        self.input = QtWidgets.QLabel("")
        self.input.setWordWrap(True)
        self.state = QtWidgets.QLabel("")
        self.run = run_button(run_text)
        lay.addWidget(self.title)
        lay.addWidget(self.input, 1)
        lay.addWidget(self.state)
        lay.addWidget(self.run)

    def set_input(self, text: str):
        self.input.setText(text)

    def set_state(self, state: str, detail: str = ""):
        """state: pending | ok | stale | running."""
        txt, col = {"pending": ("○ not run", THEME.muted),
                    "ok": ("● up to date", THEME.s(2)),
                    "stale": ("▲ out of date — re-run", THEME.event),
                    "running": ("… running", THEME.accent)}[state]
        self.state.setText(txt + (f"  ({detail})" if detail else ""))
        self.state.setStyleSheet(f"color:{col}; font-weight:600")
