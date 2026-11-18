"""Run computations off the GUI thread (MNE compute only — never plotting)."""

from __future__ import annotations

import traceback

from PySide6 import QtCore, QtWidgets


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str, str)


class _Task(QtCore.QRunnable):
    def __init__(self, fn, signals):
        super().__init__()
        self.fn, self.signals = fn, signals

    def run(self):
        try:
            res = self.fn()
        except Exception as e:  # reported on the GUI thread
            self.signals.failed.emit(f"{type(e).__name__}: {e}", traceback.format_exc())
            return
        self.signals.done.emit(res)


class Worker(QtCore.QObject):
    """One background task at a time, with a busy indicator in the status bar."""

    busyChanged = QtCore.Signal(bool)

    def __init__(self, window: QtWidgets.QMainWindow):
        super().__init__(window)
        self.window = window
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.busy = False
        self._keep = None
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.label = QtWidgets.QLabel("")
        window.statusBar().addPermanentWidget(self.label)
        # elapsed time of the running job, so long computations visibly progress
        self._clock = QtCore.QElapsedTimer()
        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._show_elapsed)
        self._text = ""
        window.statusBar().addPermanentWidget(self.progress)

    def _show_elapsed(self):
        s = self._clock.elapsed() // 1000
        self.label.setText(f"{self._text}  {s // 60}:{s % 60:02d}")

    def run(self, label: str, fn, done, failed=None) -> bool:
        if self.busy:
            QtWidgets.QMessageBox.information(
                self.window, "Busy", "Another computation is still running — please wait.")
            return False
        self.busy = True
        self.busyChanged.emit(True)
        self._text = label
        self.label.setText(label)
        self._clock.start()
        self._tick.start()
        self.progress.setVisible(True)
        sig = _Signals()
        self._keep = sig

        def finish():
            self._tick.stop()
            self.busy = False
            self.progress.setVisible(False)
            self.label.setText("")
            self.busyChanged.emit(False)

        def ok(res):
            finish()
            done(res)

        def bad(msg, tb):
            finish()
            print(tb)
            if failed is not None:
                failed(msg)
            else:
                QtWidgets.QMessageBox.warning(self.window, label.rstrip("…"), msg)

        sig.done.connect(ok)
        sig.failed.connect(bad)
        self.pool.start(_Task(fn, sig))
        return True
