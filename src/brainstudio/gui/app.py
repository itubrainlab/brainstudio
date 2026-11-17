"""Command-line entry point: ``brainstudio [PATH] [--demo] [--screen WxH]``."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import brainstudio.gui  # noqa: F401  (sets Qt env vars before Qt imports)


def main(argv=None):
    ap = argparse.ArgumentParser(description="BrainStudio: EEG + eye-tracking viewer")
    ap.add_argument("path", nargs="?", help="BIDS dataset root, XDF, EEG or EyeLink file")
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic BIDS dataset + XDF and open it")
    ap.add_argument("--screen", default=None, help="screen resolution, e.g. 1920x1080")
    args = ap.parse_args(argv)

    from PySide6 import QtWidgets

    import pyqtgraph as pg  # noqa: F401

    import warnings

    import matplotlib.pyplot as plt

    # embedded figures are briefly tiny while tabs/splitters lay out; matplotlib then
    # skips its layout for that frame and warns — harmless, the next draw is correct
    warnings.filterwarnings("ignore", message="constrained_layout not applied",
                            category=UserWarning)

    from .main_window import MainWindow

    # Interactive mode: MNE's plt.show() must not start a nested Qt event loop
    # ("QCoreApplication::exec: The event loop is already running").
    plt.ion()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("BrainStudio")
    win = MainWindow()
    if args.screen:
        w, h = (int(v) for v in args.screen.lower().split("x"))
        win.scr_w.setValue(w)
        win.scr_h.setValue(h)
    win.show()
    path = args.path
    if args.demo:
        from ..demo import write_bids, write_xdf_demo
        out = Path(tempfile.mkdtemp(prefix="brainstudio_demo_"))
        write_bids(out / "bids")
        write_xdf_demo(out / "freeview.xdf")
        print(f"Demo data written to {out}")
        path = path or str(out / "bids")
    if path:
        p = Path(path)
        if p.is_dir():
            win.open_bids(p)
        else:
            win.open_file(p)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
