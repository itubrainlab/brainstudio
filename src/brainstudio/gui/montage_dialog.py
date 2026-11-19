"""Dialog to rename channels and apply an MNE montage."""

from __future__ import annotations

from .. import core as C
from pathlib import Path

import mne
import numpy as np
from PySide6 import QtCore, QtWidgets


MONTAGE_FILES = ("Montage files (*.elc *.sfp *.loc *.locs *.eloc *.csd *.elp *.hpts "
                 "*.bvef *.txt *.xyz *.fif *_electrodes.tsv);;All files (*)")


class MontageDialog(QtWidgets.QDialog):
    def __init__(self, raw: mne.io.BaseRaw, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel names & montage")
        self.resize(560, 640)
        self.raw = raw
        self.montage: mne.channels.DigMontage | None = None
        lay = QtWidgets.QVBoxLayout(self)

        box = QtWidgets.QGroupBox("Montage (electrode positions)")
        g = QtWidgets.QGridLayout(box)
        self.rb_builtin = QtWidgets.QRadioButton("Built-in")
        self.rb_file = QtWidgets.QRadioButton("File")
        self.rb_none = QtWidgets.QRadioButton("None (rename channels only)")
        self.rb_builtin.setChecked(True)
        self.builtin = QtWidgets.QComboBox()
        names = C.builtin_montages()
        self.layouts = C.layout_montages()
        self.builtin.addItems(names)
        for i, n in enumerate(self.layouts):
            self.builtin.setItemData(i, "Channel layout: also defines the input order "
                                        "(names are filled in when you pick it)",
                                     QtCore.Qt.ToolTipRole)
        if self.layouts:
            self.builtin.insertSeparator(len(self.layouts))
        for pref in self._preferred(raw) + ["colin27_1005", "standard_1005", "standard_1020"]:
            if pref in names:
                self.builtin.setCurrentText(pref)
                break
        self.path = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        g.addWidget(self.rb_builtin, 0, 0)
        g.addWidget(self.builtin, 0, 1, 1, 2)
        g.addWidget(self.rb_file, 1, 0)
        g.addWidget(self.path, 1, 1)
        g.addWidget(browse, 1, 2)
        g.addWidget(self.rb_none, 2, 0, 1, 3)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Channel names, in recording order (one per line)")
        v = QtWidgets.QVBoxLayout(box)
        self.names = QtWidgets.QPlainTextEdit("\n".join(raw.ch_names))
        v.addWidget(self.names)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Fill from montage order", self._fill_from_montage),
                          ("Load names from file…", self._load_names),
                          ("Reset", lambda: self.names.setPlainText("\n".join(raw.ch_names)))):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        hint = QtWidgets.QLabel(
            "Names are matched to the montage case-insensitively. Channels named "
            "*EOG*/*ECG*/*EMG* get that channel type; ACC/GYR/Battery/Counter/Validation "
            "become misc. Leave a line unchanged to keep a name.")
        hint.setWordWrap(True)
        hint.setEnabled(False)
        v.addWidget(hint)
        lay.addWidget(box, 1)

        self.preview = QtWidgets.QLabel("")
        self.preview.setWordWrap(True)
        lay.addWidget(self.preview)
        self.builtin.currentTextChanged.connect(self._builtin_changed)
        if self.builtin.currentText() in self.layouts:
            self._builtin_changed(self.builtin.currentText())
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        check = bb.addButton("Check", QtWidgets.QDialogButtonBox.ActionRole)
        plot = bb.addButton("Plot sensors (MNE)…", QtWidgets.QDialogButtonBox.ActionRole)
        check.clicked.connect(self._check)
        plot.clicked.connect(self._plot)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    # ------------------------------------------------------------------ helpers
    def _preferred(self, raw) -> list[str]:
        """A layout with this many channels, if the recording's names have no positions."""
        ref = mne.channels.make_standard_montage(
            "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages()
            else "standard_1005")
        known = {n.upper() for n in ref.ch_names}
        if sum(n.upper() in known for n in raw.ch_names) >= 4:
            return []
        out = []
        for name, path in self.layouts.items():
            if len(C.read_channel_layout(path)[0]) == len(raw.ch_names):
                out.append(name)
        return out

    def _builtin_changed(self, name: str):
        if name not in self.layouts:
            return
        self.rb_builtin.setChecked(True)
        self._fill_from_montage()

    def _browse(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Montage file", "", MONTAGE_FILES)
        if fn:
            self.path.setText(fn)
            self.rb_file.setChecked(True)

    def _load_names(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Channel names", "",
                                                      "Text (*.txt *.csv *.tsv);;All (*)")
        if fn:
            txt = Path(fn).read_text(encoding="utf-8", errors="replace")
            names = [n.strip() for n in txt.replace(",", "\n").replace("\t", "\n").splitlines()
                     if n.strip()]
            self.names.setPlainText("\n".join(names))

    def _get_montage(self):
        if self.rb_none.isChecked():
            return None
        src = self.builtin.currentText() if self.rb_builtin.isChecked() else self.path.text()
        if not src:
            raise ValueError("Choose a montage file")
        return C.montage_from_source(src)

    def _fill_from_montage(self):
        try:
            m = self._get_montage()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Montage", str(e))
            return
        if m is None:
            return
        names = list(m.ch_names)
        n = len(self.raw.ch_names)
        # keep existing names beyond the montage length
        filled = names[:n] + self.raw.ch_names[len(names):]
        self.names.setPlainText("\n".join(filled))
        src = self.builtin.currentText() if self.rb_builtin.isChecked() else ""
        if src in self.layouts:
            msg = (f"Filled {min(n, len(names))} names from the {src} layout "
                   f"(amplifier input order, {len(names)} channels).")
            if len(names) != n:
                msg += f" The recording has {n} channels: check the extra/missing ones."
        else:
            msg = (f"Filled {min(n, len(names))} names from the montage's file order. "
                   "Check that it matches your cap's channel order.")
        self.preview.setText(msg)

    def new_names(self) -> list[str]:
        names = [n.strip() for n in self.names.toPlainText().splitlines()]
        names = [n for n in names if n] if len(names) > len(self.raw.ch_names) else names
        return (names + self.raw.ch_names[len(names):])[: len(self.raw.ch_names)]

    def _trial(self):
        # a 1-sample stand-in with the same info: avoids copying the whole recording
        test = mne.io.RawArray(np.zeros((len(self.raw.ch_names), 1)), self.raw.info.copy(),
                               verbose="error")
        montage = self._get_montage()
        report = C.apply_montage(test, montage, self.new_names())
        return test, montage, report

    def _check(self):
        try:
            _, _, rep = self._trial()
        except Exception as e:
            self.preview.setText(f"⚠ {e}")
            return None
        miss = rep["missing"]
        self.preview.setText(
            f"{rep['n_renamed']} channels renamed, {rep['n_positioned']} EEG channels with "
            f"positions" + (f"; no position for: {', '.join(miss[:12])}"
                            f"{' …' if len(miss) > 12 else ''}" if miss else "."))
        return rep

    def _plot(self):
        try:
            test, _, _ = self._trial()
            test.plot_sensors(show_names=True, show=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Plot sensors", str(e))

    def _accept(self):
        try:
            self.montage = self._get_montage()
            self.montage_source = None if self.rb_none.isChecked() else (
                self.builtin.currentText() if self.rb_builtin.isChecked() else self.path.text())
            self.names_out = self.new_names()
            self._trial()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Montage", str(e))
            return
        self.accept()
