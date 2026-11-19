"""Command-line entry point:
``brainstudio [PATH] [--demo] [--screen WxH] [--version] [--self-test]``."""

from __future__ import annotations

from .. import core as C
import argparse
import sys
import tempfile
from pathlib import Path

import brainstudio.gui  # noqa: F401  (sets Qt env vars before Qt imports)


def main(argv=None):
    ap = argparse.ArgumentParser(description="brAInStudio: EEG + eye-tracking viewer")
    ap.add_argument("path", nargs="?", help="BIDS dataset root, XDF, EEG or EyeLink file")
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic BIDS dataset + XDF and open it")
    ap.add_argument("--screen", default=None, help="screen resolution, e.g. 1920x1080")
    ap.add_argument("--version", action="store_true", help="print the version and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="open the window, load the synthetic demo, preprocess it and exit "
                         "with 0 (ok) or 1 (failure); used to check built packages")
    args = ap.parse_args(argv)
    if args.version:
        from .. import __version__
        _say(f"brAInStudio {__version__}")
        return 0

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
    app.setApplicationName("brAInStudio")
    win = MainWindow()
    if args.screen:
        w, h = (int(v) for v in args.screen.lower().split("x"))
        win.scr_w.setValue(w)
        win.scr_h.setValue(h)
    win.show()
    if args.self_test:
        return _self_test(app, win)
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


def _say(msg: str) -> None:
    """Print, also from windowed builds that have no console (stdout is None)."""
    if sys.stdout is not None:
        print(msg, flush=True)


def _analysis_checks(win) -> dict[str, bool]:
    """Run each analysis once on the loaded demo (catches modules missing in builds)."""
    out = {}
    raw, table = win.pipeline.clean, win.event_df
    try:
        ica, fit_raw = C.fit_ica(win.rec.raw, (), 40.0, None, C.ICAParams(n_components=5))
        out["ICA (picard)"] = ica.n_components_ == 5 and len(
            C.score_blink_components(ica, fit_raw, win.gaze_events.blinks)) == 5
        out["ICLabel"] = fit_raw.ic_labels is not None and len(fit_raw.ic_labels) == 5
        if not out["ICLabel"]:
            _say(f"ICLabel: {fit_raw.fit_note}")
    except Exception as e:  # report and go on
        _say(f"ICA: {e!r}")
        out["ICA (picard)"] = False
        out["ICLabel"] = False
    try:
        groups = {"stim": table[table.name == "stimulus"],
                  "fix": table[table.name == "fixation"].iloc[:60]}
        eps, summ = C.build_group_epochs(raw, groups, C.EpochSettings(-0.2, 0.8, None))
        out["epochs"] = set(eps) == {"stim", "fix"}
        out["AutoReject"] = summ.attrs.get("autoreject") == "applied"
        # an exported script: plain standard code, no brAInStudio import
        import ast

        from .. import script as SC
        spec = C.ErpSpec(["stim", "fix"], C.COMPONENT_PRESETS["P300"], (-0.2, 0.0))
        step = {"kind": "load", "path": "x.xdf", "screen_size": (1920, 1080),
                "roles": None, "options": None,
                "recording": {"channels_in": list(raw.ch_names),
                              "types_in": list(raw.get_channel_types()), "streams": []}}
        recipe = {"load": {"steps": [step], "gaze": {"params": None, "aois": {}}},
                  "pre": None, "groups": [C.GroupSpec("stim", ["stimulus"])],
                  "settings": C.EpochSettings()}
        code = SC.build_script("erp", recipe, spec, {"ch_names": list(raw.ch_names),
                                                     "n_subjects": 1})
        ast.parse(code)
        out["script export"] = "def read_xdf(" in code and "brainstudio" not in code.split(
            '"""', 2)[2]
        erp = {g: e.copy().apply_baseline((-0.2, 0)) for g, e in eps.items()}
        summary, _ = C.component_analysis(erp, C.COMPONENT_PRESETS["P300"])
        out["ERP component statistics"] = "p" in summary
        bp = C.band_power(eps["stim"], C.BandSettings(tmin=0.0, tmax=0.8))
        out["band power"] = bp.shape[0] == len(eps["stim"])
    except Exception as e:
        _say(f"analysis: {e!r}")
        out.setdefault("epochs", False)
        out.setdefault("AutoReject", False)
        out.setdefault("script export", False)
        out.setdefault("ERP component statistics", False)
        out.setdefault("band power", False)
    return out


def _stimulus_check(win) -> bool:
    """The demo's stimuli are on the timeline and a video frame decodes (PyAV)."""
    st = win.session.stimuli
    try:
        v = st[st["kind"] == "video"].iloc[0]
        frame = win.stim_reader.frame(v["path"], "video", 1.0)
        return set(st["kind"]) == {"image", "video"} and frame.shape[2] == 3
    except Exception as e:  # noqa: BLE001
        _say(f"stimuli: {e!r}")
        return False


def _self_test(app, win, timeout: float = 300.0) -> int:
    """Load the synthetic demo and run the default preprocessing: 0 if it worked."""
    import time
    import traceback

    from ..demo import write_bids
    try:
        out = Path(tempfile.mkdtemp(prefix="brainstudio_selftest_"))
        write_bids(out / "bids", subjects=("01",))
        win.open_bids(out / "bids")
        t0 = time.time()
        while (win.worker.busy or win.pipeline.clean is None) and time.time() - t0 < timeout:
            app.processEvents()
            time.sleep(0.02)
        rec = win.rec
        checks = {
            "recording loaded": rec is not None,
            "EEG": rec is not None and rec.raw is not None,
            "gaze": rec is not None and rec.has_gaze,
            "gaze events": win.gaze_events is not None and len(win.gaze_events.fixations) > 0,
            "events": len(win.event_df) > 0,
            "preprocessed": win.pipeline.clean is not None,
            "montage resources": bool(C.layout_montages()),
            "stimuli (images, video)": _stimulus_check(win),
        }
        checks.update(_analysis_checks(win))
        for k, ok in checks.items():
            _say(f"{'ok  ' if ok else 'FAIL'} {k}")
        return 0 if all(checks.values()) else 1
    except Exception:
        _say(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
