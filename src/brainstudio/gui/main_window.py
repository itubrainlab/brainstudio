"""BrainStudio main window."""

from __future__ import annotations

import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import eeg as E
from .. import events as EV
from .. import gaze as G
from .. import io as bio
from ..model import Recording
from .analysis import AnalysisPanel, _spin, fill_table
from .montage_dialog import MontageDialog
from .screenview import ScreenView
from .theme import THEME, apply_theme
from .timeplots import EEGPlot, GazePlot, OverviewBar
from .topoview import TopoView

pg.setConfigOptions(antialias=False, useOpenGL=False)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BrainStudio — EEG & eye-tracking viewer")
        self.resize(1600, 950)
        self.rec: Recording | None = None
        self.display_raw: mne.io.BaseRaw | None = None
        self.gaze_events = None
        self.pupil_clean = None
        self.bids_root: Path | None = None
        self.cursor_t = 0.0
        self.gaze_offset = 0.0  # manual sync correction (s), applied to gaze
        self._raw_gaze = None

        self._build_central()
        self._build_left_dock()
        self._build_right_dock()
        self._build_menu()
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self._play_step)
        self.statusBar().showMessage("Open a BIDS dataset (File ▸ Open BIDS…) or an XDF file.")
        self.event_df = EV.build_event_table(None)
        self._apply_theme()
        hints = QtGui.QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):  # Qt >= 6.5: follow system light/dark
            hints.colorSchemeChanged.connect(lambda *_: QtCore.QTimer.singleShot(
                0, self._apply_theme))

    # ================================================================== layout
    def _build_central(self):
        self.eeg_plot = EEGPlot()
        self.gaze_plot = GazePlot()
        self.gaze_plot.pos_plot.setXLink(self.eeg_plot.plotItem)
        self.screen = ScreenView()
        self.overview = OverviewBar()

        time_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        time_split.addWidget(self.eeg_plot)
        time_split.addWidget(self.gaze_plot)
        time_split.setSizes([620, 300])
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(time_split, 1)
        ll.addWidget(self.overview)
        ll.addLayout(self._transport())

        self.topo = TopoView()
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right.addWidget(self.screen)
        right.addWidget(self.topo)
        right.setSizes([420, 380])
        h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        h.addWidget(left)
        h.addWidget(right)
        h.setSizes([1000, 560])

        self.analysis = AnalysisPanel(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(h, "Viewer")
        self.tabs.addTab(self.analysis, "Analysis")
        self.tabs.currentChanged.connect(self._on_tab)
        self.setCentralWidget(self.tabs)

        self.eeg_plot.cursorMoved.connect(self.set_cursor)
        self.gaze_plot.cursorMoved.connect(self.set_cursor)
        self.eeg_plot.windowChanged.connect(self._on_window)
        self.overview.regionChanged.connect(lambda a, b: self.eeg_plot.set_window(a, b))
        self.screen.aoisChanged.connect(self._aois_changed)
        # click in a time plot moves the cursor
        for plot in (self.eeg_plot.plotItem, self.gaze_plot.pos_plot):
            plot.scene().sigMouseClicked.connect(lambda ev, p=plot: self._click(ev, p))

    def _transport(self):
        row = QtWidgets.QHBoxLayout()
        st = self.style()
        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward))
        self.btn_prev.clicked.connect(lambda: self.page(-1))
        self.btn_play = QtWidgets.QToolButton()
        self.btn_play.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self.toggle_play)
        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaSeekForward))
        self.btn_next.clicked.connect(lambda: self.page(1))
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed.setCurrentIndex(2)
        self.time_label = QtWidgets.QLabel("t = 0.000 s")
        self.time_label.setMinimumWidth(260)
        for w in (self.btn_prev, self.btn_play, self.btn_next, QtWidgets.QLabel("speed"),
                  self.speed, self.time_label):
            row.addWidget(w)
        row.addStretch()
        return row

    def _build_left_dock(self):
        dock = QtWidgets.QDockWidget("Dataset", self)
        dock.setObjectName("dataset")
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Recording"])
        self.tree.itemActivated.connect(self._tree_activated)
        lay.addWidget(self.tree, 3)
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(150)
        lay.addWidget(self.info)
        lay.addWidget(QtWidgets.QLabel("Events (double-click to jump)"))
        self.event_table = QtWidgets.QTableWidget()
        self.event_table.cellDoubleClicked.connect(self._event_jump)
        lay.addWidget(self.event_table, 2)
        dock.setWidget(w)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
        self.left_dock = dock

    def _build_right_dock(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setObjectName("controls")
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)

        # --- view
        box = QtWidgets.QGroupBox("View")
        f = QtWidgets.QFormLayout(box)
        self.win_len = _spin(0.5, 600, 10, 1, 1, " s")
        self.win_len.valueChanged.connect(self._apply_window_len)
        self.eeg_scale = _spin(1, 5000, 50, 5, 0, " µV")
        self.eeg_scale.valueChanged.connect(lambda v: self.eeg_plot.set_scale(v))
        self.n_chan = QtWidgets.QSpinBox()
        self.n_chan.setRange(1, 512)
        self.n_chan.setValue(20)
        self.n_chan.valueChanged.connect(lambda v: self.eeg_plot.set_visible_channels(v))
        f.addRow("Window", self.win_len)
        f.addRow("EEG scale", self.eeg_scale)
        f.addRow("Channels shown", self.n_chan)
        self.ov_checks = {}
        for kind, on in (("fixations", False), ("saccades", True), ("blinks", True)):
            cb = QtWidgets.QCheckBox(f"Shade {kind}")
            cb.setChecked(on)
            cb.toggled.connect(self._overlay_toggled)
            self.ov_checks[kind] = cb
            f.addRow(cb)
        self.cb_events = QtWidgets.QCheckBox("Show event markers")
        self.cb_events.setChecked(True)
        self.cb_events.toggled.connect(self._refresh_events)
        f.addRow(self.cb_events)
        self.cb_scan = QtWidgets.QCheckBox("Scanpath in screen view")
        self.cb_scan.setChecked(True)
        self.cb_scan.toggled.connect(self._scan_toggled)
        f.addRow(self.cb_scan)
        self.cb_heat = QtWidgets.QCheckBox("Fixation heatmap")
        self.cb_heat.toggled.connect(self._heat_toggled)
        f.addRow(self.cb_heat)
        self.trail = _spin(0, 10, 0.5, 0.1, 2, " s")
        self.trail.valueChanged.connect(self._trail_changed)
        f.addRow("Gaze trail", self.trail)
        lay.addWidget(box)

        # --- EEG preprocessing
        box = QtWidgets.QGroupBox("EEG preprocessing (MNE)")
        f = QtWidgets.QFormLayout(box)
        self.hp = _spin(0, 100, 1.0, 0.1, 2, " Hz")
        self.lp = _spin(0, 1000, 40.0, 1, 1, " Hz")
        self.hp.setSpecialValueText("off")
        self.lp.setSpecialValueText("off")
        self.notch = QtWidgets.QComboBox()
        self.notch.addItems(["off", "50 Hz", "60 Hz"])
        self.ref = QtWidgets.QComboBox()
        self.ref.setEditable(True)  # pick a channel or type several: "M1, M2"
        self.ref.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.ref.setToolTip("keep = as recorded · average = common average · a channel, "
                            "or several separated by commas (their mean, e.g. M1, M2)")
        self.ref.addItems(["keep", "average"])
        f.addRow("High-pass", self.hp)
        f.addRow("Low-pass", self.lp)
        f.addRow("Notch", self.notch)
        f.addRow("Reference", self.ref)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Apply")
        b.clicked.connect(self.apply_filters)
        r = QtWidgets.QPushButton("Raw")
        r.clicked.connect(self.reset_filters)
        row.addWidget(b)
        row.addWidget(r)
        f.addRow(row)
        mb = QtWidgets.QPushButton("Open in MNE browser…")
        mb.clicked.connect(self.open_mne_browser)
        f.addRow(mb)
        lay.addWidget(box)

        # --- gaze
        box = QtWidgets.QGroupBox("Gaze event detection")
        f = QtWidgets.QFormLayout(box)
        p = G.DetectionParams()
        self.p_fix_dist = _spin(1, 500, p.fix_max_dist, 5, 0, " px")
        self.p_fix_dur = _spin(0.01, 2, p.fix_min_dur, 0.01, 3, " s")
        self.p_sac_vel = _spin(10, 1e5, p.sac_vel_thresh, 100, 0, " px/s")
        self.p_sac_dur = _spin(0.001, 0.5, p.sac_min_dur, 0.005, 3, " s")
        self.p_blink_min = _spin(0.005, 2, p.blink_min_dur, 0.01, 3, " s")
        self.p_blink_max = _spin(0.05, 10, p.blink_max_dur, 0.1, 2, " s")
        f.addRow("Fixation radius", self.p_fix_dist)
        f.addRow("Min fixation", self.p_fix_dur)
        f.addRow("Saccade velocity", self.p_sac_vel)
        f.addRow("Min saccade", self.p_sac_dur)
        f.addRow("Blink min", self.p_blink_min)
        f.addRow("Blink max", self.p_blink_max)
        self.cb_pupil_clean = QtWidgets.QCheckBox("Interpolate pupil over blinks")
        self.cb_pupil_clean.setChecked(True)
        f.addRow(self.cb_pupil_clean)
        b = QtWidgets.QPushButton("Detect events")
        b.clicked.connect(self.detect_gaze_events)
        f.addRow(b)
        self.offset = _spin(-10000, 10000, 0, 1, 1, " ms")
        self.offset.setToolTip("Shift gaze relative to EEG (sync correction)")
        self.offset.editingFinished.connect(self._offset_changed)
        f.addRow("Gaze offset", self.offset)
        self.scr_w = QtWidgets.QSpinBox()
        self.scr_h = QtWidgets.QSpinBox()
        for s, v in ((self.scr_w, 1920), (self.scr_h, 1080)):
            s.setRange(16, 16384)
            s.setValue(v)
            s.editingFinished.connect(self._screen_changed)
        sr = QtWidgets.QHBoxLayout()
        sr.addWidget(self.scr_w)
        sr.addWidget(QtWidgets.QLabel("×"))
        sr.addWidget(self.scr_h)
        f.addRow("Screen (px)", sr)
        lay.addWidget(box)
        lay.addStretch()
        scroll.setWidget(w)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.right_dock = dock

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        acts = [
            ("Open &BIDS dataset…", "Ctrl+B", self.action_open_bids),
            ("&Open file (XDF / EEG / EyeLink)…", QtGui.QKeySequence.Open, self.action_open_file),
            ("Attach &eye-tracking file…", "Ctrl+E", self.action_attach_gaze),
            ("Channel names && &montage…", "Ctrl+M", self.action_montage),
            ("Load &stimulus image…", "Ctrl+I", self.action_stimulus),
            (None, None, None),
            ("Export gaze events (CSV)…", None, self.export_gaze_events),
            ("Save screen view (PNG)…", None, self.export_screen),
            (None, None, None),
            ("&Quit", QtGui.QKeySequence.Quit, self.close),
        ]
        for label, key, fn in acts:
            if label is None:
                m.addSeparator()
                continue
            a = m.addAction(label)
            if key:
                a.setShortcut(QtGui.QKeySequence(key))
            a.triggered.connect(fn)
        v = self.menuBar().addMenu("&View")
        v.addAction(self.left_dock.toggleViewAction())
        v.addAction(self.right_dock.toggleViewAction())
        h = self.menuBar().addMenu("&Help")
        h.addAction("Keyboard shortcuts", lambda: QtWidgets.QMessageBox.information(
            self, "Shortcuts", "←/→  previous/next page\nSpace  play/pause\n"
            "Click in a time plot  move cursor\n+/-  EEG scale\n"
            "PgUp/PgDn  scroll channels"))

    # ================================================================== loading
    def action_open_bids(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open BIDS dataset root")
        if d:
            self.open_bids(d)

    def action_open_file(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", "",
            "Recordings (*.xdf *.xdfz *.vhdr *.edf *.bdf *.set *.fif *.asc);;All files (*)")
        if fn:
            self.open_file(fn)

    def action_attach_gaze(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Attach eye-tracking data", "",
            "Eye tracking (*.asc *.tsv *.tsv.gz *.xdf);;All files (*)")
        if fn and self.rec is not None:
            with self._busy("Loading eye tracking…"):
                g = bio.load_eyetracking_file(fn, self.screen_size())
                self.rec.gaze = g
                self._raw_gaze = g
                self._after_gaze_change()

    def action_stimulus(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Stimulus image", "",
                                                      "Images (*.png *.jpg *.jpeg *.bmp)")
        if fn:
            import matplotlib.image as mpimg
            img = mpimg.imread(fn)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            self.screen.set_stimulus(img)

    def open_bids(self, root):
        root = Path(root)
        with self._busy("Scanning BIDS dataset…"):
            runs = bio.scan_bids(root)
        self.bids_root = root
        self.tree.clear()
        desc = bio.bids.dataset_description(root)
        top = QtWidgets.QTreeWidgetItem([desc.get("Name", root.name)])
        self.tree.addTopLevelItem(top)
        nodes = {}
        for run in runs:
            sub = run.entities.get("subject") or "?"
            ses = run.entities.get("session")
            key = (sub,)
            if key not in nodes:
                nodes[key] = QtWidgets.QTreeWidgetItem(top, [f"sub-{sub}"])
            parent = nodes[key]
            if ses:
                k2 = (sub, ses)
                if k2 not in nodes:
                    nodes[k2] = QtWidgets.QTreeWidgetItem(parent, [f"ses-{ses}"])
                parent = nodes[k2]
            item = QtWidgets.QTreeWidgetItem(parent, [run.label])
            item.setData(0, QtCore.Qt.UserRole, run)
            item.setToolTip(0, "\n".join(str(p) for p in [run.eeg, *run.gaze] if p))
        self.tree.expandAll()
        self.left_dock.show()
        self.statusBar().showMessage(f"{len(runs)} runs found in {root}. Double-click to load.")
        if len(runs) == 1:
            self.load_run(runs[0])

    def _tree_activated(self, item, _col):
        run = item.data(0, QtCore.Qt.UserRole)
        if run is not None:
            self.load_run(run)

    def load_run(self, run):
        try:
            with self._busy(f"Loading {run.label}…"):
                rec = bio.load_bids_run(self.bids_root, run, self.screen_size_or_none())
        except Exception as e:
            self._error("Could not load run", e)
            return
        self.set_recording(rec)

    def open_file(self, path):
        path = Path(path)
        if path.is_dir():
            self.open_bids(path)
            return
        try:
            with self._busy(f"Loading {path.name}…"):
                rec = bio.load(path, self.screen_size())
        except Exception as e:
            self._error("Could not load file", e)
            return
        self.tree.clear()
        QtWidgets.QTreeWidgetItem(self.tree, [path.name])
        self.set_recording(rec)

    def screen_size(self):
        return (self.scr_w.value(), self.scr_h.value())

    def screen_size_or_none(self):
        return None if self.screen_size() == (1920, 1080) else self.screen_size()

    # ================================================================== state
    def set_recording(self, rec: Recording):
        self.stop_play()
        self.rec = rec
        self._raw_gaze = rec.gaze
        self.offset.setValue(0)
        self.gaze_offset = 0.0
        self.gaze_events = None
        self.pupil_clean = None
        if rec.has_gaze:
            self.scr_w.setValue(rec.gaze.screen_size[0])
            self.scr_h.setValue(rec.gaze.screen_size[1])
            self.screen.set_screen(rec.gaze.screen_size)
        self.display_raw = None
        self.screen.reset_view()
        if rec.raw is not None:
            self._refresh_reference_choices()
            self.apply_filters(redraw=False)
            self._load_eeg_plot()
        else:
            self.eeg_plot.data = None
            for c in self.eeg_plot.curves:
                c.setData([], [])
        self.topo.set_raw(self.display_raw)
        self.gaze_plot.set_gaze(rec.gaze)
        if rec.has_gaze:
            self.detect_gaze_events()
        self.info.setPlainText(rec.summary() + ("\n" + rec.description if rec.description else ""))
        self._refresh_events()
        self._rebuild_event_table()
        self.analysis.update_gaze_stats()
        win = min(self.win_len.value(), max(rec.duration, 0.5))
        if rec.raw is None and rec.has_gaze:
            self.eeg_plot.plotItem.getViewBox().setLimits(xMin=-1, xMax=rec.duration + 1)
        self.eeg_plot.set_window(0, win)
        self.set_cursor(0.0)
        self.setWindowTitle(f"BrainStudio — {rec.source.name if rec.source else ''}")
        self.statusBar().showMessage(rec.summary().replace("\n", "  |  "))

    def _eeg_picks(self, raw):
        return mne.pick_types(raw.info, meg=True, eeg=True, eog=True, ecg=True, emg=True,
                              misc=True, seeg=True, ecog=True, dbs=True, bio=True,
                              resp=True, fnirs=True, exclude=[])

    def _load_eeg_plot(self):
        raw = self.display_raw
        picks = self._eeg_picks(raw)
        data = raw.get_data(picks=picks).astype(np.float32)  # display only
        names = [raw.ch_names[i] for i in picks]
        types = raw.get_channel_types(picks=picks)
        self.eeg_plot.set_data(data, raw.times, names, types, raw.info["bads"])
        self.n_chan.setMaximum(max(len(names), 1))
        self.eeg_plot.set_visible_channels(min(self.n_chan.value(), len(names)))

    def _refresh_reference_choices(self):
        """Reference options: keep, average, then the recording's EEG channels."""
        cur = self.ref.currentText()
        raw = self.rec.raw if self.rec is not None else None
        names = ([raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
                 if raw is not None else [])
        self.ref.blockSignals(True)
        self.ref.clear()
        self.ref.addItems(["keep", "average"])
        if names:
            self.ref.insertSeparator(2)
            self.ref.addItems(names)
        try:  # keep the previous choice when it still exists in this recording
            ok = raw is not None and E.parse_reference(cur, raw) is not None
        except ValueError:
            ok = False
        self.ref.setCurrentText(cur if ok or cur in ("keep", "average") else "keep")
        self.ref.blockSignals(False)

    def apply_filters(self, redraw=True):
        if self.rec is None or self.rec.raw is None:
            return
        notch = {"off": None, "50 Hz": 50.0, "60 Hz": 60.0}[self.notch.currentText()]
        p = E.FilterParams(self.hp.value() or None, self.lp.value() or None, notch,
                           self.ref.currentText())
        try:
            E.parse_reference(p.reference, self.rec.raw)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Reference", f"{e}\nUsing the recorded "
                                          "reference instead.")
            self.ref.setCurrentText("keep")
            p.reference = None
        try:
            with self._busy("Filtering…"):
                self.display_raw = E.preprocess(self.rec.raw, p)
        except Exception as e:
            self._error("Filtering failed", e)
            self.display_raw = self.rec.raw
        if redraw:
            picks = self._eeg_picks(self.display_raw)
            self.eeg_plot.update_values(self.display_raw.get_data(picks=picks).astype(np.float32))
            self.topo.set_raw(self.display_raw)

    def reset_filters(self):
        if self.rec is None or self.rec.raw is None:
            return
        self.display_raw = self.rec.raw
        self.eeg_plot.update_values(
            self.display_raw.get_data(picks=self._eeg_picks(self.display_raw)).astype(np.float32))
        self.topo.set_raw(self.display_raw)

    def open_mne_browser(self):
        if self.display_raw is None:
            return
        raw = self.display_raw.copy()
        if self.gaze_events is not None:
            E.add_gaze_annotations(raw, self.gaze_events, ("blinks", "saccades"))
        t0, t1 = self.eeg_plot.window()
        raw.plot(start=max(t0, 0), duration=t1 - t0, block=False, show=True,
                 scalings=dict(eeg=self.eeg_scale.value() * 1e-6,
                               eog=self.eeg_scale.value() * 4e-6))

    def _gaze_params(self):
        return G.DetectionParams(
            blink_min_dur=self.p_blink_min.value(), blink_max_dur=self.p_blink_max.value(),
            fix_max_dist=self.p_fix_dist.value(), fix_min_dur=self.p_fix_dur.value(),
            sac_vel_thresh=self.p_sac_vel.value(), sac_min_dur=self.p_sac_dur.value())

    def detect_gaze_events(self):
        if self.rec is None or not self.rec.has_gaze:
            return
        with self._busy("Detecting fixations, saccades and blinks…"):
            self.gaze_events = G.detect_events(self.rec.gaze, self._gaze_params())
            self.rec.gaze_events = self.gaze_events
            self.pupil_clean = (G.clean_pupil(self.rec.gaze, self.gaze_events.blinks)
                                if self.cb_pupil_clean.isChecked() else None)
        self.gaze_plot.set_gaze(self.rec.gaze, self.pupil_clean)
        self.eeg_plot.gaze_events = self.gaze_events
        self.gaze_plot.gaze_events = self.gaze_events
        self._overlay_toggled()
        self._heat_toggled()
        self.analysis.update_gaze_stats()
        self._rebuild_event_table()
        self._update_screen()

    # ================================================================== events / montage
    def _rebuild_event_table(self):
        aois = self.screen.pygaze_aois() if self.screen.aois else None
        self.event_df = EV.build_event_table(self.rec, self.gaze_events, aois)
        self.analysis.set_event_table(self.event_df)

    def _aois_changed(self):
        self.analysis.update_aoi_stats()
        if self.rec is not None:
            self._rebuild_event_table()

    def action_montage(self):
        if self.rec is None or self.rec.raw is None:
            QtWidgets.QMessageBox.information(self, "Montage", "Load a recording with EEG first.")
            return
        dlg = MontageDialog(self.rec.raw, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            with self._busy("Applying montage…"):
                rep = E.apply_montage(self.rec.raw, dlg.montage, dlg.names_out)
                if self.display_raw is not None and self.display_raw is not self.rec.raw:
                    E.apply_montage(self.display_raw, dlg.montage, dlg.names_out)
        except Exception as e:
            self._error("Montage", e)
            return
        self._refresh_reference_choices()
        self._load_eeg_plot()
        self.topo.set_raw(self.display_raw)
        self.info.setPlainText(self.rec.summary() + "\n" + self.rec.description)
        self.statusBar().showMessage(
            f"Montage applied: {rep['n_renamed']} channels renamed, "
            f"{rep['n_positioned']} EEG channels with positions", 8000)

    # ================================================================== theme
    def _apply_theme(self):
        apply_theme(QtWidgets.QApplication.instance(), self)
        for w in (self.eeg_plot, self.gaze_plot, self.overview, self.screen, self.topo):
            w.restyle()
        self.analysis.restyle()
        self._heat_toggled()

    def changeEvent(self, ev):
        if ev.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange) \
                and hasattr(self, "analysis"):
            dark = QtWidgets.QApplication.instance().palette().color(
                QtGui.QPalette.Base).lightness() < 128
            if dark != THEME.dark:
                QtCore.QTimer.singleShot(0, self._apply_theme)
        super().changeEvent(ev)

    def _after_gaze_change(self):
        g = self.rec.gaze
        self.gaze_plot.set_gaze(g)
        self.screen.set_screen(g.screen_size)
        self.detect_gaze_events()
        self.overview.set_recording(self.rec.duration, self.rec.all_events(), g)
        self.info.setPlainText(self.rec.summary())

    def _offset_changed(self):
        if self.rec is None or self._raw_gaze is None:
            return
        new = self.offset.value() / 1000.0
        if new == self.gaze_offset:
            return
        self.gaze_offset = new
        self.rec.gaze = self._raw_gaze.shifted(new)
        self._after_gaze_change()

    def _screen_changed(self):
        size = self.screen_size()
        self.screen.set_screen(size)
        if self.rec is not None and self.rec.gaze is not None:
            self.rec.gaze.screen_size = size
            self._heat_toggled()

    # ================================================================== display
    def _refresh_events(self):
        if self.rec is None:
            return
        ev = self.rec.all_events()
        show = ev if self.cb_events.isChecked() else None
        if show is not None and len(show) > 500:
            show = show.iloc[:500]
        self.eeg_plot._set_event_lines(self.eeg_plot.plotItem, show)
        self.gaze_plot._set_event_lines(self.gaze_plot.pos_plot, show, labels=False)
        self.overview.set_recording(self.rec.duration, ev, self.rec.gaze)
        fill_table(self.event_table, ev[["onset", "description"]], fmt="{:.3f}")

    def _overlay_toggled(self, *_):
        kinds = {k: cb.isChecked() for k, cb in self.ov_checks.items()}
        self.eeg_plot.overlay_kinds = dict(kinds)
        self.gaze_plot.overlay_kinds = dict(kinds)
        t0, t1 = self.eeg_plot.window()
        self.eeg_plot._update_overlays(t0, t1)
        self.gaze_plot._update_overlays(t0, t1)

    def _scan_toggled(self, on):
        self.screen.show_scanpath = on
        self._update_screen()

    def _heat_toggled(self, *_):
        self.screen.show_heatmap = self.cb_heat.isChecked()
        if self.gaze_events is not None and self.rec is not None and self.rec.has_gaze:
            hm = G.heatmap(self.gaze_events.fixations, self.screen.screen)
            self.screen.set_heatmap(hm)
        else:
            self.screen.set_heatmap(None)

    def _trail_changed(self, v):
        self.screen.trail_len = v
        self._update_screen()

    def _apply_window_len(self, v):
        t0, _ = self.eeg_plot.window()
        self.eeg_plot.set_window(t0, t0 + v)

    def _on_window(self, t0, t1):
        self.overview.set_region(t0, t1)
        self.gaze_plot._update_overlays(t0, t1)
        if not self.win_len.hasFocus():
            self.win_len.blockSignals(True)
            self.win_len.setValue(t1 - t0)
            self.win_len.blockSignals(False)
        if not (t0 <= self.cursor_t <= t1) and not self.play_timer.isActive():
            self.set_cursor(t0 + 0.1 * (t1 - t0))
        else:
            self._update_screen()

    def _click(self, ev, plot):
        if ev.button() != QtCore.Qt.LeftButton or ev.double():
            return
        vb = plot.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.set_cursor(float(vb.mapSceneToView(ev.scenePos()).x()))

    def set_cursor(self, t: float):
        self.cursor_t = t
        self.eeg_plot.set_cursor(t)
        self.gaze_plot.set_cursor(t)
        txt = f"t = {t:.3f} s"
        if self.rec is not None and self.rec.has_gaze:
            x, y = self.rec.gaze.at(t)
            txt += f"   gaze = ({x:.0f}, {y:.0f}) px" if np.isfinite(x) else "   gaze = —"
        self.time_label.setText(txt)
        self.topo.request(t)
        self._update_screen()

    def _update_screen(self):
        if self.rec is None:
            return
        t0, t1 = self.eeg_plot.window()
        self.screen.update_view(self.rec.gaze, self.gaze_events, self.cursor_t, t0, t1)

    def _on_tab(self, i):
        if i == 1:
            self.analysis.update_aoi_stats()

    def _event_jump(self, row, _col):
        try:
            t = float(self.event_table.item(row, 0).text())
        except (AttributeError, ValueError):
            return
        t0, t1 = self.eeg_plot.window()
        w = t1 - t0
        self.eeg_plot.set_window(t - w / 4, t + 3 * w / 4)
        self.set_cursor(t)

    def page(self, direction: int):
        t0, t1 = self.eeg_plot.window()
        w = t1 - t0
        self.eeg_plot.set_window(t0 + direction * w, t1 + direction * w)

    # ================================================================== playback
    def toggle_play(self, on):
        st = self.style()
        if on and self.rec is not None:
            self.btn_play.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaPause))
            self._play_clock = QtCore.QElapsedTimer()
            self._play_clock.start()
            self._play_start = self.cursor_t
            self.play_timer.start()
        else:
            self.btn_play.setIcon(st.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            self.play_timer.stop()

    def stop_play(self):
        self.btn_play.setChecked(False)

    def _play_step(self):
        speed = float(self.speed.currentText().rstrip("×"))
        t = self._play_start + speed * self._play_clock.elapsed() / 1000.0
        if self.rec is None or t > self.rec.duration:
            self.stop_play()
            return
        t0, t1 = self.eeg_plot.window()
        if t > t1:
            w = t1 - t0
            self.eeg_plot.set_window(t, t + w)
        self.set_cursor(t)

    # ================================================================== export
    def export_gaze_events(self):
        if self.gaze_events is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export gaze events",
                                                      "gaze_events.csv", "CSV (*.csv)")
        if not fn:
            return
        frames = []
        for kind in ("fixations", "saccades", "blinks"):
            df = getattr(self.gaze_events, kind).copy()
            df.insert(0, "type", kind[:-1])
            frames.append(df)
        pd.concat(frames, ignore_index=True).sort_values("onset").to_csv(fn, index=False)

    def export_screen(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save screen view",
                                                      "screen.png", "PNG (*.png)")
        if fn:
            self.screen.grab().save(fn)

    # ================================================================== keys / utils
    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        k = ev.key()
        if k == QtCore.Qt.Key_Right:
            self.page(1)
        elif k == QtCore.Qt.Key_Left:
            self.page(-1)
        elif k == QtCore.Qt.Key_Space:
            self.btn_play.toggle()
        elif k in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self.eeg_scale.setValue(self.eeg_scale.value() / 1.5)
        elif k == QtCore.Qt.Key_Minus:
            self.eeg_scale.setValue(self.eeg_scale.value() * 1.5)
        elif k in (QtCore.Qt.Key_PageDown, QtCore.Qt.Key_PageUp):
            vb = self.eeg_plot.plotItem.getViewBox()
            (_, _), (y0, y1) = vb.viewRange()
            d = (y1 - y0) * (1 if k == QtCore.Qt.Key_PageUp else -1)
            vb.setYRange(y0 + d, y1 + d, padding=0)
        else:
            super().keyPressEvent(ev)

    class _BusyCtx:
        def __init__(self, win, msg):
            self.win, self.msg = win, msg

        def __enter__(self):
            self.win.statusBar().showMessage(self.msg)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QtWidgets.QApplication.processEvents()

        def __exit__(self, *exc):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.win.statusBar().clearMessage()
            return False

    def _busy(self, msg):
        return self._BusyCtx(self, msg)

    def _error(self, title, e: Exception):
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(self, title, f"{type(e).__name__}: {e}")
