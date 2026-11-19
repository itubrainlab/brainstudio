"""Details of one XDF stream: everything known about it, its role, and how each
of its channels is used (include/rename/type for EEG, x/y/pupil/validity for gaze,
name/flag/attribute for marker columns)."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core import MARKER_COLUMN_ROLES as COLUMN_ROLES
from ..core import EEG_CHANNEL_TYPES, GAZE_COLUMN_ROLES, ROLES
from .widgets import hint

CHECKED, UNCHECKED = QtCore.Qt.Checked, QtCore.Qt.Unchecked


def _ro(text) -> QtWidgets.QTableWidgetItem:
    it = QtWidgets.QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
    return it


class StreamDialog(QtWidgets.QDialog):
    def __init__(self, stream: dict, role: str, options: dict | None, parent=None):
        super().__init__(parent)
        self.st = stream
        self._opts_in = options or {}
        self.setWindowTitle(f"Stream — {stream['name']}")
        self.resize(820, 680)
        lay = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"<b style='font-size:15px'>{stream['name']}</b>"
                                 f"&nbsp;&nbsp;<span style='color:#9a9a9a'>{stream['type']}"
                                 f" · id {stream['key']}</span>")
        head.addWidget(title, 1)
        head.addWidget(QtWidgets.QLabel("role"))
        self.role = QtWidgets.QComboBox()
        self.role.addItems(list(ROLES))
        self.role.setCurrentText(role)
        self.role.currentTextChanged.connect(self._build_table)
        head.addWidget(self.role)
        lay.addLayout(head)

        box = QtWidgets.QGroupBox("Stream")
        g = QtWidgets.QGridLayout(box)
        eff = stream.get("effective_srate")
        items = [
            ("format", stream["format"]), ("channels", stream["channels"]),
            ("nominal rate", f"{stream['srate']:g} Hz" if stream["srate"] else "irregular"),
            ("effective rate", f"{eff:.4f} Hz" if eff else "—"),
            ("samples", f"{stream['samples']:,}"),
            ("duration", f"{stream['duration']:.1f} s"),
            ("first time stamp", f"{stream['first']:.3f} s" if stream["first"] is not None
             else "—"),
            ("source id", stream.get("source_id") or "—"),
            ("host", stream.get("hostname") or "—"),
            ("manufacturer", stream.get("manufacturer") or "—"),
            ("automatic role", stream["auto_role"]),
        ]
        for k, (lab, val) in enumerate(items):
            g.addWidget(QtWidgets.QLabel(lab), k // 3, 2 * (k % 3))
            v = QtWidgets.QLabel(str(val))
            v.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            v.setStyleSheet("font-weight:600")
            g.addWidget(v, k // 3, 2 * (k % 3) + 1)
        if stream.get("notes"):
            n = hint("Clock: " + "; ".join(stream["notes"]))
            g.addWidget(n, (len(items) + 2) // 3, 0, 1, 6)
        lay.addWidget(box)

        self.table_box = QtWidgets.QGroupBox("Channels")
        tv = QtWidgets.QVBoxLayout(self.table_box)
        self.tools = QtWidgets.QHBoxLayout()
        tv.addLayout(self.tools)
        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setVisible(False)
        tv.addWidget(self.table, 1)
        self.help = hint("")
        tv.addWidget(self.help)
        self.end_events = QtWidgets.QCheckBox("Also add an instant “<name>_end” event when a "
                                              "start/stop pair closes")
        tv.addWidget(self.end_events)
        lay.addWidget(self.table_box, 1)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        reset = bb.addButton("Automatic", QtWidgets.QDialogButtonBox.ResetRole)
        reset.clicked.connect(lambda: self._build_table(use_auto=True))
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._build_table()

    # ------------------------------------------------------------------ defaults
    def _defaults(self, role: str, use_auto=False) -> dict:
        ch = self.st["channel_info"]
        labels = [c["label"] for c in ch]
        auto = self.st.get("auto_options", {}).get(role)
        if not use_auto and role == self.st["role"] and self._opts_in:
            return self._opts_in
        if auto:
            return auto
        if role == "EEG":
            return {"include": [True] * len(ch), "names": labels, "types": ["eeg"] * len(ch)}
        if role == "gaze":
            return {"columns": ["extra"] * len(ch)}
        if role == "markers":
            return {"names": labels, "columns": ["name"] + ["attribute"] * (len(ch) - 1),
                    "end_events": True}
        return {}

    def _clear_tools(self):
        while self.tools.count():
            it = self.tools.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

    def _build_table(self, *_, use_auto=False):
        role = self.role.currentText()
        d = self._defaults(role, use_auto)
        ch = self.st["channel_info"]
        t = self.table
        t.clear()
        self._clear_tools()
        self.end_events.setVisible(False)
        single = bool(d.get("single"))
        if role == "EEG":
            cols = ["use", "#", "name", "type", "unit", "values"]
        elif role == "gaze":
            cols = ["#", "channel", "use as", "unit", "values"]
        elif role == "markers" and not single:
            cols = ["#", "column name", "use as", "values"]
        else:
            cols = ["#", "channel", "type", "unit", "values"]
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(ch))
        for i, c in enumerate(ch):
            if role == "EEG":
                use = QtWidgets.QTableWidgetItem("")
                use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                inc = d.get("include") or [True] * len(ch)
                use.setCheckState(CHECKED if (inc[i] if i < len(inc) else True) else UNCHECKED)
                t.setItem(i, 0, use)
                t.setItem(i, 1, _ro(i + 1))
                names = d.get("names") or []
                t.setItem(i, 2, QtWidgets.QTableWidgetItem(names[i] if i < len(names)
                                                           else c["label"]))
                combo = QtWidgets.QComboBox()
                combo.addItems(list(EEG_CHANNEL_TYPES))
                types = d.get("types") or []
                combo.setCurrentText(types[i] if i < len(types) else "eeg")
                t.setCellWidget(i, 3, combo)
                t.setItem(i, 4, _ro(c["unit"]))
                t.setItem(i, 5, _ro(c["preview"]))
            elif role == "gaze":
                t.setItem(i, 0, _ro(i + 1))
                t.setItem(i, 1, _ro(c["label"]))
                combo = QtWidgets.QComboBox()
                combo.addItems(list(GAZE_COLUMN_ROLES))
                cr = d.get("columns") or []
                combo.setCurrentText(cr[i] if i < len(cr) else "extra")
                t.setCellWidget(i, 2, combo)
                t.setItem(i, 3, _ro(c["unit"]))
                t.setItem(i, 4, _ro(c["preview"]))
            elif role == "markers" and not single:
                t.setItem(i, 0, _ro(i + 1))
                names = d.get("names") or []
                t.setItem(i, 1, QtWidgets.QTableWidgetItem(names[i] if i < len(names)
                                                           else c["label"]))
                combo = QtWidgets.QComboBox()
                combo.addItems(list(COLUMN_ROLES))
                cr = d.get("columns") or []
                combo.setCurrentText(cr[i] if i < len(cr) else "attribute")
                t.setCellWidget(i, 2, combo)
                t.setItem(i, 3, _ro(c["preview"]))
            else:
                for j, v in enumerate((i + 1, c["label"], c["type"], c["unit"], c["preview"])):
                    t.setItem(i, j, _ro(v))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)

        if role == "EEG":
            for label, fn in (("All", lambda: self._check_all(True)),
                              ("None", lambda: self._check_all(False)),
                              ("Only type eeg", self._only_eeg)):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(fn)
                self.tools.addWidget(b)
            self.tools.addStretch()
            self.help.setText("Untick channels you do not want to load. Names and types can "
                              "be edited here or later in Channel names & montage (Ctrl+M).")
        elif role == "gaze":
            self.tools.addStretch()
            self.help.setText("x and y: screen coordinates (several channels are averaged, e.g. "
                              "left/right eye); pupil: pupil size; valid: a 0/1 validity "
                              "flag. Other channels stay available as extras.")
        elif role == "markers" and not single:
            self.tools.addStretch()
            self.end_events.setVisible(True)
            self.end_events.setChecked(bool(d.get("end_events", True)))
            self.help.setText("Each sample becomes an event. name: the event name (several "
                              "columns are joined with _); flag: a start/stop column — a start "
                              "and the next stop/end with the same name become one event with "
                              "a duration (phases for other events); attribute: kept as a "
                              "column you can filter on, under the name given here.")
        elif role == "markers":
            self.help.setText("Single-column marker stream: each value is an event; values "
                              "like NAME|key=value become a name plus columns.")
        else:
            self.help.setText("This stream is not loaded.")

    def _check_all(self, on):
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setCheckState(CHECKED if on else UNCHECKED)

    def _only_eeg(self):
        for i in range(self.table.rowCount()):
            on = self.table.cellWidget(i, 3).currentText() == "eeg"
            self.table.item(i, 0).setCheckState(CHECKED if on else UNCHECKED)

    # ------------------------------------------------------------------ result
    def result_options(self) -> tuple[str, dict]:
        role = self.role.currentText()
        t = self.table
        n = t.rowCount()
        if role == "EEG":
            return role, {"include": [t.item(i, 0).checkState() == CHECKED for i in range(n)],
                          "names": [t.item(i, 2).text().strip() for i in range(n)],
                          "types": [t.cellWidget(i, 3).currentText() for i in range(n)]}
        if role == "gaze":
            return role, {"columns": [t.cellWidget(i, 2).currentText() for i in range(n)]}
        if role == "markers" and t.columnCount() == 4:
            return role, {"names": [t.item(i, 1).text().strip() or f"col{i + 1}"
                                    for i in range(n)],
                          "columns": [t.cellWidget(i, 2).currentText() for i in range(n)],
                          "end_events": self.end_events.isChecked()}
        return role, {}

    def _accept(self):
        role, o = self.result_options()
        msg = None
        if role == "EEG" and not any(o["include"]):
            msg = "Select at least one channel."
        elif role == "EEG":
            names = [nm for nm, on in zip(o["names"], o["include"]) if on]
            if len(set(names)) != len(names):
                msg = "Channel names must be unique."
        elif role == "gaze" and not ("x" in o["columns"] and "y" in o["columns"]):
            msg = "Choose at least one x and one y channel."
        elif role == "markers" and o:
            if "name" not in o["columns"]:
                msg = "Choose at least one column as the event name."
            elif o["columns"].count("flag") > 1:
                msg = "Only one column can be the start/stop flag."
            used = [nm for nm, r in zip(o["names"], o["columns"]) if r == "attribute"]
            if not msg and len(set(used)) != len(used):
                msg = "Attribute column names must be unique."
        if msg:
            QtWidgets.QMessageBox.warning(self, "Stream", msg)
            return
        self.accept()
