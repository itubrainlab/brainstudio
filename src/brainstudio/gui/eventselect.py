"""Event selection widget: checklist of event types + text and metadata filters."""

from __future__ import annotations

from .. import core as C
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from .theme import THEME

QUERY_HELP = ("Metadata filter — pandas / MNE epochs-metadata syntax, e.g.\n"
              '  phase == "READING" and cond == "AI"\n'
              "  duration > 0.2 and aoi == \"AOI1\"\n"
              "  onset > 300 and page in [1, 2]\n"
              "Columns are listed under ‘Columns’.")


class _ConditionRow(QtWidgets.QWidget):
    """One column filter: [column] [operator] [value] [×]."""

    changed = QtCore.Signal()
    removed = QtCore.Signal(object)

    def __init__(self, columns, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.col = QtWidgets.QComboBox()
        self.col.setMinimumWidth(90)
        self.op = QtWidgets.QComboBox()
        self.op.addItems(list(C.OPERATORS))
        self.op.setToolTip("= ≠ < ≤ > ≥ compare numbers (or text for = ≠); contains: text "
                           "anywhere in the value; one of: comma-separated values")
        self.value = QtWidgets.QComboBox()
        self.value.setEditable(True)
        self.value.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.value.setMinimumWidth(80)
        rm = QtWidgets.QToolButton()
        rm.setText("×")
        rm.setToolTip("Remove this filter")
        rm.clicked.connect(lambda: self.removed.emit(self))
        lay.addWidget(self.col, 3)
        lay.addWidget(self.op, 0)
        lay.addWidget(self.value, 3)
        lay.addWidget(rm)
        self._table = None
        self.set_columns(columns)
        self.col.currentTextChanged.connect(self._column_changed)
        self.op.currentTextChanged.connect(lambda _: self.changed.emit())
        self.value.currentTextChanged.connect(lambda _: self.changed.emit())

    def set_columns(self, table: pd.DataFrame | list):
        self._table = table if isinstance(table, pd.DataFrame) else None
        cols = list(table.columns) if isinstance(table, pd.DataFrame) else list(table)
        cols = [c for c in cols if c != "description"]
        cur = self.col.currentText()
        self.col.blockSignals(True)
        self.col.clear()
        self.col.addItems(cols)
        if cur in cols:
            self.col.setCurrentText(cur)
        self.col.blockSignals(False)
        self._fill_values()

    def _column_changed(self, _):
        self._fill_values()
        self.changed.emit()

    def _fill_values(self):
        cur = self.value.currentText()
        self.value.blockSignals(True)
        self.value.clear()
        t, c = self._table, self.col.currentText()
        if t is not None and c in t:
            vals = t[c].dropna()
            vals = vals[vals.astype(str) != ""]
            if pd.api.types.is_numeric_dtype(vals):
                u = sorted(vals.unique())
                items = [f"{v:g}" for v in u[:300]]
                self.value.setToolTip(f"{c}: {vals.min():g} … {vals.max():g}")
            else:
                counts = vals.astype(str).value_counts()
                items = list(counts.index[:300])
                self.value.setToolTip(f"{c}: {len(counts)} different values")
            self.value.addItems(items)
        self.value.setCurrentText(cur)
        self.value.blockSignals(False)

    def condition(self) -> tuple[str, str, str]:
        return self.col.currentText(), self.op.currentText(), self.value.currentText().strip()

    def set_condition(self, col, op, value):
        self.col.setCurrentText(col)
        self.op.setCurrentText(op)
        self.value.setCurrentText(str(value))


class EventSelector(QtWidgets.QGroupBox):
    """Pick events from the unified event table (see :func:`brainstudio.core.event_table`)."""

    changed = QtCore.Signal()

    def __init__(self, title: str = "Events", color_index: int | None = None,
                 default_names=(), parent=None):
        super().__init__(parent)
        self.color_index = color_index
        self.table = pd.DataFrame(columns=C.BASE_COLUMNS)
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
        # column filters: pick a column, an operator and one of its values
        self._conds: list[_ConditionRow] = []
        self.cond_box = QtWidgets.QWidget()
        self.cond_lay = QtWidgets.QVBoxLayout(self.cond_box)
        self.cond_lay.setContentsMargins(0, 0, 0, 0)
        self.cond_lay.setSpacing(3)
        lay.addWidget(self.cond_box)
        add = QtWidgets.QPushButton("+ Filter by column")
        add.setToolTip("Keep only events whose column matches a value (all filters must match)")
        add.clicked.connect(lambda: self.add_condition())
        lay.addWidget(add)
        self.text.setPlaceholderText("Text filter (regex on the description)…")
        self.query.setPlaceholderText('Advanced: pandas query, e.g. label == "human"')
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
        cols.setToolTip("Every column of the event table with example values; pick one to "
                        "filter on it")
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
        for r in self._conds:
            r.set_columns(table)
        self._refresh()

    # ---- column filters
    def add_condition(self, col: str | None = None, op: str = "=", value: str = ""):
        row = _ConditionRow(self.table, self)
        if col:
            row.col.setCurrentText(col)
        row.op.setCurrentText(op)
        if value:
            row.value.setCurrentText(str(value))
        elif row.value.count():
            row.value.setCurrentIndex(0)
        row.changed.connect(self._timer.start)
        row.removed.connect(self._remove_condition)
        self._conds.append(row)
        self.cond_lay.addWidget(row)
        self._timer.start()
        return row

    def _remove_condition(self, row):
        self._conds.remove(row)
        self.cond_lay.removeWidget(row)  # gone from the panel now, not at the next loop
        row.hide()
        row.setParent(None)
        row.deleteLater()
        self._timer.start()

    def conditions(self) -> list[tuple[str, str, str]]:
        return [r.condition() for r in self._conds]

    def set_conditions(self, conds):
        for r in list(self._conds):
            self._remove_condition(r)
        for c, op, v in conds:
            self.add_condition(c, op, v)

    def label(self) -> str:
        return self.label_edit.text().strip() or self.title()

    def checked_names(self) -> set[str]:
        return set(self._checked)

    def set_checked(self, names):
        self._checked = set(names)
        self._refresh()

    def selected(self) -> pd.DataFrame:
        """Rows selected by the checklist and filters (raises ValueError)."""
        return C.filter_events(self.table, self.text.text(), self.query.text(),
                                self._checked, self.conditions())

    def describe(self) -> str:
        parts = [", ".join(sorted(self._checked)) or "nothing"]
        if C.describe_conditions(self.conditions()):
            parts.append(C.describe_conditions(self.conditions()))
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
        """Columns menu: add a filter on that column."""
        row = self.add_condition(col)
        row.value.setFocus()

    def _refresh(self):
        self._ready = False
        self.list.blockSignals(True)
        self.list.clear()
        err = None
        try:
            filtered = C.filter_events(self.table, self.text.text(), self.query.text(),
                                        conditions=self.conditions())
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
