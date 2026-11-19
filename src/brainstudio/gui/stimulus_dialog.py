"""Which stimulus each event showed: the folder with the images / videos, the event
column that names them and the mapping of its values to files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .. import core as C
from .widgets import hint

NONE = "—"
#: event-table columns that never name a stimulus
NOT_STIMULUS = ("onset", "duration", "kind", "description", "phase")


class StimulusDialog(QtWidgets.QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Stimuli")
        self.resize(760, 620)
        self.ses, self.reader = session, C.StimulusReader()
        table = session.event_table
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(hint(
            "The screen view shows, from each event's onset and for its duration (else "
            "until the next one), the image or video its column names. BIDS keeps them in "
            "the dataset's stimuli folder."))
        form = QtWidgets.QFormLayout()
        row = QtWidgets.QHBoxLayout()
        self.folder = QtWidgets.QLineEdit(str(session.stimulus_folder or ""))
        self.folder.editingFinished.connect(self._folder_changed)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow("Folder", row)
        self.files_note = QtWidgets.QLabel("")
        form.addRow("", self.files_note)
        self.column = QtWidgets.QComboBox()
        self.column.addItem("(no stimuli)", None)
        for c in table.columns if table is not None else []:
            if c not in NOT_STIMULUS and len(C._marker_values(table, c)):
                self.column.addItem(c, c)
        self.column.currentIndexChanged.connect(lambda _: self._fill(auto=True))
        form.addRow("Event column", self.column)
        lay.addLayout(form)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["value", "events", "stimulus file"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.currentCellChanged.connect(lambda r, *_: self._preview(r))
        split.addWidget(self.table)
        self.preview = QtWidgets.QLabel("")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumWidth(240)
        split.addWidget(self.preview)
        split.setSizes([480, 260])
        lay.addWidget(split, 1)
        self.summary = QtWidgets.QLabel("")
        lay.addWidget(self.summary)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        match = bb.addButton("Match by name", QtWidgets.QDialogButtonBox.ResetRole)
        match.setToolTip("Each value → the file with the same path, name or name without "
                         "extension")
        match.clicked.connect(lambda: self._fill(auto=True))
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.files = list(session.stimulus_files)
        self._show_files()
        i = self.column.findData(session.stimulus_column)
        self.column.blockSignals(True)
        self.column.setCurrentIndex(max(i, 0))
        self.column.blockSignals(False)
        self._fill(auto=False)

    # ------------------------------------------------------------------ folder
    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Stimulus folder",
                                                       self.folder.text())
        if d:
            self.folder.setText(d)
            self._folder_changed()

    def _folder_changed(self):
        self.files = C.stimulus_files(self.folder.text().strip() or None)
        self._show_files()
        self._fill(auto=True)

    def _show_files(self):
        n_img = sum(Path(f).suffix.lower() in C.STIMULUS_IMAGES for f in self.files)
        self.files_note.setText(f"{n_img} images, {len(self.files) - n_img} videos"
                                if self.files else "no images or videos in this folder")

    # ------------------------------------------------------------------ mapping
    def _fill(self, auto: bool):
        """One row per value of the column: its events and the file it shows."""
        col = self.column.currentData()
        vals = C._marker_values(self.ses.event_table, col) if col else None
        counts = vals.value_counts(sort=False) if vals is not None else {}
        keys = sorted(counts.keys(), key=lambda v: (not _is_number(v),
                                                    float(v) if _is_number(v) else 0, v))
        current = C.match_stimuli(keys, self.files) if auto or \
            col != self.ses.stimulus_column else dict(self.ses.stimulus_map)
        self.table.setRowCount(len(keys))
        for r, v in enumerate(keys):
            for c, text in ((0, v), (1, str(counts[v]))):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                self.table.setItem(r, c, it)
            box = QtWidgets.QComboBox()
            box.addItems([NONE] + self.files)
            box.setCurrentText(current.get(v) or NONE)
            box.currentIndexChanged.connect(lambda _, r=r: (self._update_summary(),
                                                            self._preview(r)))
            self.table.setCellWidget(r, 2, box)
        self.table.resizeColumnsToContents()
        self._update_summary()
        if keys:
            self.table.selectRow(0)
            self._preview(0)

    def mapping(self) -> dict[str, str]:
        out = {}
        for r in range(self.table.rowCount()):
            f = self.table.cellWidget(r, 2).currentText()
            if f != NONE:
                out[self.table.item(r, 0).text()] = f
        return out

    def _update_summary(self):
        m = self.mapping()
        n = sum(int(self.table.item(r, 1).text()) for r in range(self.table.rowCount())
                if self.table.item(r, 0).text() in m)
        self.summary.setText(f"{len(m)} of {self.table.rowCount()} values have a file; "
                             f"{n} events show a stimulus")

    def _preview(self, row: int):
        self.preview.clear()
        if row < 0 or row >= self.table.rowCount():
            return
        f = self.table.cellWidget(row, 2).currentText()
        if f == NONE:
            self.preview.setText("no file")
            return
        kind = "image" if Path(f).suffix.lower() in C.STIMULUS_IMAGES else "video"
        try:
            img = np.ascontiguousarray(self.reader.frame(Path(self.folder.text()) / f, kind))
        except Exception as e:  # noqa: BLE001
            self.preview.setText(f"cannot read {f}:\n{e}")
            return
        q = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1],
                         QtGui.QImage.Format_RGB888).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(q).scaled(
            self.preview.width() - 8, 360, QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation))
        self.preview.setToolTip(f"{f} ({kind}, {img.shape[1]}×{img.shape[0]})")

    def result_settings(self):
        """(folder, column, mapping) chosen."""
        folder = self.folder.text().strip()
        return (Path(folder) if folder else None), self.column.currentData(), self.mapping()

    def done(self, r):
        self.reader.close()
        super().done(r)


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False
