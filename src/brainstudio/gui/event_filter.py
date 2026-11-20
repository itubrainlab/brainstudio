"""Conditions that filter the event list: column, test and value per row, all or any of
them met (:func:`brainstudio.core.filter_rows`)."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .. import core as C

ANY = "any column"
NO_VALUE = ("is empty", "is not empty")


class EventFilter(QtWidgets.QWidget):
    changed = QtCore.Signal()
    apply_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list[str] = []
        self._timer = QtCore.QTimer(self, singleShot=True, interval=250)
        self._timer.timeout.connect(self.changed)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self.rows = QtWidgets.QVBoxLayout()
        self.rows.setSpacing(3)
        lay.addLayout(self.rows)
        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(QtWidgets.QLabel("Match"))
        self.match_box = QtWidgets.QComboBox()
        self.match_box.addItem("all conditions", "all")
        self.match_box.addItem("any condition", "any")
        self.match_box.currentIndexChanged.connect(lambda _: self.changed.emit())
        foot.addWidget(self.match_box)
        foot.addStretch(1)
        add = QtWidgets.QToolButton()
        add.setText("+")
        add.setToolTip("Another condition")
        add.clicked.connect(lambda: self.add_condition())
        foot.addWidget(add)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setToolTip("Use only the markers shown everywhere: EEG and gaze "
                                  "views, stimuli, epoching and exported scripts (gaze "
                                  "events are kept)")
        self.apply_btn.clicked.connect(self.apply_requested)
        foot.addWidget(self.apply_btn)
        lay.addLayout(foot)
        self.state = QtWidgets.QLabel("")
        self.state.setWordWrap(True)
        lay.addWidget(self.state)
        self.add_condition()

    # ------------------------------------------------------------------ rows
    def add_condition(self, column=None, test="contains", value=""):
        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(3)
        col, op, val = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QLineEdit()
        for box in (col, op):
            box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(5)
        col.setToolTip("The column the condition looks in")
        self._fill_columns(col, column)
        op.addItems(list(C.FILTER_OPS))
        op.setCurrentText(test)
        op.setToolTip("contains: every word, part of a value is enough (any case); "
                      ">, ≥, <, ≤ compare numbers")
        val.setPlaceholderText("value")
        val.setClearButtonEnabled(True)
        val.setText(value)
        rm = QtWidgets.QToolButton()
        rm.setText("−")
        rm.setToolTip("Remove this condition")
        rm.clicked.connect(lambda: self._remove(row))
        for w in (col, op):
            w.currentIndexChanged.connect(lambda _, v=val, o=op: self._op_changed(v, o))
        val.textChanged.connect(lambda _: (self.apply_btn.setEnabled(True),
                                           self._timer.start()))
        h.addWidget(col, 2)
        h.addWidget(op, 2)
        h.addWidget(val, 3)
        h.addWidget(rm)
        row.parts = (col, op, val)
        self.rows.addWidget(row)
        self._op_changed(val, op, emit=False)
        if value or test in NO_VALUE:
            self.changed.emit()
        return row

    def _remove(self, row):
        if self.rows.count() == 1:  # the last condition is cleared, not removed
            col, op, val = row.parts
            col.setCurrentIndex(0)
            op.setCurrentText("contains")
            val.clear()
        else:
            self.rows.removeWidget(row)
            row.deleteLater()
        self.changed.emit()

    def _op_changed(self, val, op, emit=True):
        val.setEnabled(op.currentText() not in NO_VALUE)
        if emit:
            self.changed.emit()

    def _rows(self):
        return [self.rows.itemAt(i).widget() for i in range(self.rows.count())]

    # ------------------------------------------------------------------ columns
    def _fill_columns(self, box, chosen):
        box.blockSignals(True)
        box.clear()
        box.addItem(ANY, None)
        for c in self._columns:
            box.addItem(c, c)
        box.setCurrentIndex(max(box.findData(chosen), 0) if chosen else 0)
        box.blockSignals(False)

    def set_columns(self, columns):
        """The event table's columns; each condition keeps its column when still there."""
        columns = list(columns)
        if columns == self._columns:
            return
        self._columns = columns
        for row in self._rows():
            self._fill_columns(row.parts[0], row.parts[0].currentData())

    # ------------------------------------------------------------------ result
    def conditions(self) -> list[tuple]:
        return [(col.currentData(), op.currentText(), val.text())
                for col, op, val in (r.parts for r in self._rows())]

    def match(self) -> str:
        return self.match_box.currentData()

    def active(self) -> bool:
        return any(op in NO_VALUE or val.strip() for _, op, val in self.conditions())

    def show_state(self, applied: dict | None, n_used: int, n_all: int):
        """Whether the conditions shown are the ones in use, and what they keep."""
        mine = C.event_filter(self.conditions(), self.match(), None) if self.active() else None
        same = (mine or {}).get("conditions") == (applied or {}).get("conditions") and \
            (not mine or mine["match"] == applied["match"])
        self.apply_btn.setEnabled(not same)
        self.apply_btn.setDefault(not same)
        if applied:
            text = f"Filter in use: {n_used:,} of {n_all:,} markers"
            color = "#d9a441" if not same else "#6aa36f"
            text += " (changed here: Apply to use it)" if not same else ""
        else:
            text, color = ("All markers in use" + (" (Apply to use the filter)"
                                                   if not same else "")), "#9a9a9a"
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {color}")
