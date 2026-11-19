"""Small shared widgets and helpers."""

from __future__ import annotations

from .. import core as C
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .theme import THEME


def _cell_texts(col: pd.Series, fmt: str) -> list[str]:
    """One column as display strings (vectorised for numbers and flags)."""
    v = col.to_numpy()
    if v.dtype.kind == "b":
        return np.where(v, "✓", "").tolist()
    if v.dtype.kind == "f":
        ok = np.isfinite(v)
        return [fmt.format(x) if k else "" for x, k in zip(v.tolist(), ok.tolist())]
    out = []
    for x in v.tolist():
        if isinstance(x, (bool, np.bool_)):
            out.append("✓" if x else "")
        elif isinstance(x, float):
            out.append("" if not np.isfinite(x) else fmt.format(x))
        else:
            out.append(str(x))
    return out


def fill_table(table: QtWidgets.QTableWidget, df: pd.DataFrame, fmt="{:.3g}"):
    table.setUpdatesEnabled(False)
    sorting = table.isSortingEnabled()
    table.setSortingEnabled(False)
    try:
        table.clear()
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        for j, c in enumerate(df.columns):
            for i, txt in enumerate(_cell_texts(df.iloc[:, j], fmt)):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setFlags(flags)
                table.setItem(i, j, item)
        # column widths from the first rows only: measuring thousands of rows is slow
        table.horizontalHeader().setResizeContentsPrecision(200)
        table.resizeColumnsToContents()
    finally:
        table.setSortingEnabled(sorting)
        table.setUpdatesEnabled(True)


has_positions = C.has_positions


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
