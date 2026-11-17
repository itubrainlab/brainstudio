"""Event selection widget: checklist of event types + text and metadata filters."""

from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .. import events as EV
from .theme import THEME

QUERY_HELP = ("Metadata filter — pandas / MNE epochs-metadata syntax, e.g.\n"
              '  phase == "READING" and cond == "AI"\n'
              "  duration > 0.2 and aoi == \"AOI1\"\n"
              "  onset > 300 and page in [1, 2]\n"
              "Columns are listed under ‘Columns’.")


class EventSelector(QtWidgets.QGroupBox):
    """Pick events from the unified event table (see :mod:`brainstudio.events`)."""

    changed = QtCore.Signal()

    def __init__(self, title: str = "Events", color_index: int | None = None,
                 default_names=(), parent=None):
        super().__init__(parent)
        self.color_index = color_index
        self.table = pd.DataFrame(columns=EV.BASE_COLUMNS)
        self._checked: set[str] = set(default_names)
        self._ready = False
        lay = QtWidgets.QVBoxLayout(self)
        head = QtWidgets.QHBoxLayout()
        self.swatch = QtWidgets.QLabel()
        self.swatch.setFixedSize(14, 14)
        self.label_edit = QtWidgets.QLineEdit(title)
        self.label_edit.setToolTip("Label used in plots and exports")
        self.label_edit.editingFinished.connect(self.changed.emit)
        if color_index is not None:
            head.addWidget(self.swatch)
        head.addWidget(self.label_edit, 1)
        lay.addLayout(head)

        self.text = QtWidgets.QLineEdit()
        self.text.setPlaceholderText("Text filter (regex on description)…")
        self.query = QtWidgets.QLineEdit()
        self.query.setPlaceholderText('Filter, e.g.  phase == "READING" and cond == "AI"')
        self.query.setToolTip(QUERY_HELP)
        for w in (self.text, self.query):
            w.setClearButtonEnabled(True)
            lay.addWidget(w)
        self._timer = QtCore.QTimer(self, singleShot=True, interval=350)
        self._timer.timeout.connect(self._refresh)
        self.text.textChanged.connect(lambda _: self._timer.start())
        self.query.textChanged.connect(lambda _: self._timer.start())

        self.list = QtWidgets.QListWidget()
        self.list.setMinimumHeight(110)
        self.list.itemChanged.connect(self._item_changed)
        lay.addWidget(self.list, 1)

        row = QtWidgets.QHBoxLayout()
        for label, fn in (("All", lambda: self._set_all(True)),
                          ("None", lambda: self._set_all(False))):
            b = QtWidgets.QPushButton(label)
            b.setFlat(True)
            b.clicked.connect(fn)
            row.addWidget(b)
        cols = QtWidgets.QToolButton()
        cols.setText("Columns")
        cols.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.cols_menu = QtWidgets.QMenu(cols)
        cols.setMenu(self.cols_menu)
        row.addWidget(cols)
        row.addStretch()
        self.status = QtWidgets.QLabel("")
        row.addWidget(self.status)
        lay.addLayout(row)
        self.restyle()

    # ------------------------------------------------------------------ public
    def set_table(self, table: pd.DataFrame):
        self.table = table
        self._fill_columns_menu()
        self._refresh()

    def label(self) -> str:
        return self.label_edit.text().strip() or self.title()

    def checked_names(self) -> set[str]:
        return set(self._checked)

    def set_checked(self, names):
        self._checked = set(names)
        self._refresh()

    def selected(self) -> pd.DataFrame:
        """Rows selected by the checklist and filters (raises ValueError)."""
        return EV.filter_events(self.table, self.text.text(), self.query.text(),
                                self._checked)

    def describe(self) -> str:
        parts = [", ".join(sorted(self._checked)) or "nothing"]
        if self.text.text().strip():
            parts.append(f"text ~ /{self.text.text().strip()}/")
        if self.query.text().strip():
            parts.append(self.query.text().strip())
        return " | ".join(parts)

    def restyle(self):
        if self.color_index is not None:
            self.swatch.setStyleSheet(
                f"background:{THEME.group(self.color_index)}; border-radius:3px")
        self._refresh()

    # ------------------------------------------------------------------ internals
    def _fill_columns_menu(self):
        self.cols_menu.clear()
        for c in self.table.columns:
            vals = self.table[c].dropna()
            vals = vals[vals.astype(str) != ""]
            if not len(vals):
                continue
            if pd.api.types.is_numeric_dtype(vals):
                info = f"{vals.min():.4g} … {vals.max():.4g}"
            else:
                u = vals.astype(str).unique()
                info = ", ".join(u[:6]) + (" …" if len(u) > 6 else "")
            act = self.cols_menu.addAction(f"{c}:  {info}"[:90])
            act.triggered.connect(lambda _=False, c=c: self._insert_column(c))

    def _insert_column(self, col: str):
        q = self.query.text().rstrip()
        self.query.setText((q + " and " if q else "") + col + " ")
        self.query.setFocus()

    def _refresh(self):
        self._ready = False
        self.list.blockSignals(True)
        self.list.clear()
        err = None
        try:
            filtered = EV.filter_events(self.table, self.text.text(), self.query.text())
        except ValueError as e:
            filtered, err = self.table.iloc[0:0], str(e)
        counts = filtered["name"].astype(str).value_counts() if len(filtered) else pd.Series(
            dtype=int)
        all_names = self.table["name"].astype(str).value_counts() if len(self.table) else counts
        # gaze events first, then markers alphabetically
        order = sorted(all_names.index, key=lambda n: (n not in ("fixation", "saccade",
                                                                 "blink"), n))
        for n in order:
            c = int(counts.get(n, 0))
            item = QtWidgets.QListWidgetItem(f"{n}  ({c})")
            item.setData(QtCore.Qt.UserRole, n)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if n in self._checked else QtCore.Qt.Unchecked)
            if c == 0:
                item.setForeground(QtGui.QColor(THEME.muted))
            self.list.addItem(item)
        self.list.blockSignals(False)
        if err:
            self.status.setText("⚠ filter error")
            self.status.setToolTip(err)
            self.status.setStyleSheet(f"color:{THEME.event}")
        else:
            n_sel = int(counts[counts.index.isin(self._checked)].sum()) if len(counts) else 0
            self.status.setText(f"{n_sel} selected")
            self.status.setToolTip("")
            self.status.setStyleSheet(f"color:{THEME.muted}")
        self._ready = True
        self.changed.emit()

    def _item_changed(self, item):
        if not self._ready:
            return
        n = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            self._checked.add(n)
        else:
            self._checked.discard(n)
        self._refresh()

    def _set_all(self, on: bool):
        self._checked = set(self.table["name"].astype(str).unique()) if on else set()
        self._refresh()
