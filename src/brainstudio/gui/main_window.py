"""brAInStudio main window."""

from __future__ import annotations

from .. import core as C
from .. import session as S
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import io as bio
from ..core import subject_spans
from ..model import Recording
from . import icons, style
from .eeg_analysis import AnalysisTab
from .epoch_tab import EpochTab
from .gaze_tab import GazeTab
from .montage_dialog import MontageDialog
from ..session import STAGES, Session
from .controllers import Controller
from .pipeline import EEGPipeline
from .preprocess_tab import PreprocessTab
from .screenview import ScreenView
from .stream_dialog import StreamDialog
from .theme import THEME, apply_theme
from .timeplots import EEGPlot, GazePlot, OverviewBar
from .topoview import TopoView
from .widgets import _spin, fill_table
from .worker import Worker

pg.setConfigOptions(antialias=False, useOpenGL=False)


class MainWindow(QtWidgets.QMainWindow):
    # the session's state, read by the views
    rec = property(lambda self: self.session.rec)
    gaze_events = property(lambda self: self.session.gaze_events)
    event_df = property(lambda self: self.session.event_table)
    load_steps = property(lambda self: self.session.load_steps)

    def __init__(self):
        style.apply(QtWidgets.QApplication.instance())
        super().__init__()
        self.setWindowTitle("brAInStudio — EEG & eye-tracking analysis")
        self.setWindowIcon(icons.app())
        self.resize(1600, 950)
        # model (Session), its Qt bridge (pipeline) and the controller of the actions
        self.session = Session()
        self.pipeline = EEGPipeline(self.session, self)
        self.worker = Worker(self)
        self.ctl = Controller(self.session, self.worker)
        self.display_raw: mne.io.BaseRaw | None = None
        self.pupil_clean = None
        self.bids_root: Path | None = None
        self.cursor_t = 0.0

        self._build_toolbar()
        self._build_central()
        self._build_dataset_panel()
        self._build_controls_panel()
        self._build_menu()
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(40)
        self.play_timer.timeout.connect(self._play_step)
        self.statusBar().showMessage("Open a BIDS dataset (File ▸ Open BIDS…) or an XDF file.")
        self.pipeline.changed.connect(self._on_pipeline)
        self.pipeline.stateChanged.connect(self._update_pipeline_titles)
        self._update_pipeline_titles()
        self._apply_theme()
        hints = QtGui.QGuiApplication.styleHints()
        if hasattr(hints, "colorSchemeChanged"):  # Qt >= 6.5: follow system light/dark
            hints.colorSchemeChanged.connect(lambda *_: QtCore.QTimer.singleShot(
                0, self._apply_theme))

    # ================================================================== layout
    def _build_toolbar(self):
        """brAIn lab top bar: logo, wordmark, the open recording, help."""
        bar = QtWidgets.QToolBar("brAInStudio")
        bar.setObjectName("Toolbar")
        bar.setMovable(False)
        bar.setFloatable(False)
        bar.toggleViewAction().setVisible(False)
        w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(w)
        row.setContentsMargins(8, 3, 8, 3)
        row.setSpacing(0)
        logo = QtWidgets.QLabel()
        logo.setPixmap(icons.app().pixmap(QtCore.QSize(26, 26)))
        row.addWidget(logo)
        row.addSpacing(8)
        for text, name in (("br", "Brand"), ("AI", "BrandAI"), ("n", "Brand"),
                           (" studio", "BrandSub")):
            lab = QtWidgets.QLabel(text)
            lab.setObjectName(name)
            row.addWidget(lab)
        row.addSpacing(14)
        div = QtWidgets.QFrame()
        div.setObjectName("Divider")
        div.setFixedSize(1, 22)
        row.addWidget(div)
        row.addSpacing(14)
        self.title_label = QtWidgets.QLabel("no recording open")
        self.title_label.setObjectName("ProjectName")
        row.addWidget(self.title_label)
        row.addStretch(1)
        help_ = QtWidgets.QToolButton()
        help_.setIcon(icons.question())
        help_.setIconSize(QtCore.QSize(20, 20))
        help_.setToolTip("Keyboard shortcuts")
        help_.clicked.connect(self._show_shortcuts)
        row.addWidget(help_)
        bar.addWidget(w)
        self.addToolBar(QtCore.Qt.TopToolBarArea, bar)
        self.toolbar = bar

    def _show_shortcuts(self):
        QtWidgets.QMessageBox.information(
            self, "Shortcuts", "←/→  previous/next page\nSpace  play/pause\n"
            "Click in a time plot  move cursor\n+/-  EEG scale\nPgUp/PgDn  scroll channels")

    def _build_ribbon(self) -> QtWidgets.QWidget:
        """Buttons for the File menu's load and export functions."""
        rib = QtWidgets.QWidget()
        rib.setObjectName("Ribbon")
        rib.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        g = QtWidgets.QGridLayout(rib)
        g.setContentsMargins(4, 4, 4, 4)
        g.setSpacing(2)
        items = (("BIDS", icons.folder(), "Open a BIDS dataset (Ctrl+B)", self.action_open_bids),
                 ("File", icons.file(), "Open an XDF / EEG / EyeLink file (Ctrl+O)",
                  self.action_open_file),
                 ("Eye data", icons.eye_plus(), "Attach an eye-tracking file to the "
                  "recording (Ctrl+E)", self.action_attach_gaze),
                 ("Montage", icons.cap(), "Channel names & montage (Ctrl+M)",
                  self.action_montage),
                 ("Stimulus", icons.image(), "Load a stimulus image for the screen view "
                  "(Ctrl+I)", self.action_stimulus),
                 ("Export", icons.export(), "Export gaze events (CSV)",
                  self.export_gaze_events),
                 ("Snapshot", icons.camera(), "Save the screen view (PNG)", self.export_screen))
        self.ribbon_buttons = {}
        for k, (text, icon, tip, fn) in enumerate(items):
            b = QtWidgets.QToolButton()
            b.setText(text)
            b.setIcon(icon)
            b.setIconSize(QtCore.QSize(26, 26))
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            b.clicked.connect(fn)
            g.addWidget(b, k // 4, k % 4)
            self.ribbon_buttons[text] = b
        return rib

    def _update_ribbon(self):
        rec = self.rec
        on = {"Eye data": rec is not None, "Montage": rec is not None and rec.raw is not None,
              "Export": self.gaze_events is not None,
              "Snapshot": rec is not None and rec.has_gaze}
        for k, v in on.items():
            self.ribbon_buttons[k].setEnabled(v)

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
        self.time_split = time_split
        right.addWidget(self.topo)
        right.setSizes([420, 380])
        h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.data_split = h  # Dataset panel | viewer | screen & topomap | Controls panel
        h.addWidget(left)
        h.addWidget(right)
        h.setSizes([1000, 560])

        # --- EEG: preprocessing → epoching → analysis pipeline
        self.pre_tab = PreprocessTab(self.pipeline, self.worker, self.ctl)
        self.epoch_tab = EpochTab(self.pipeline, self.worker, self.ctl)
        self.eeg_analysis = AnalysisTab(self.pipeline, self.worker, self.ctl)
        eeg = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(eeg)
        ev.setContentsMargins(4, 4, 4, 4)
        bar = QtWidgets.QHBoxLayout()
        self.pipe_label = QtWidgets.QLabel("")
        bar.addWidget(self.pipe_label, 1)
        self.btn_pipe = QtWidgets.QPushButton("Run pipeline ▶▶")
        self.btn_pipe.setObjectName("Primary")
        self.btn_pipe.setToolTip("Run every stage that is not up to date, in order")
        self.btn_pipe.clicked.connect(self.run_pipeline)
        bar.addWidget(self.btn_pipe)
        ev.addLayout(bar)
        self.eeg_tabs = QtWidgets.QTabWidget()
        self.eeg_tabs.addTab(self.pre_tab, "① Preprocessing")
        self.eeg_tabs.addTab(self.epoch_tab, "② Epoching")
        self.eeg_tabs.addTab(self.eeg_analysis, "③ Analysis")
        ev.addWidget(self.eeg_tabs, 1)

        self.gaze_tab = GazeTab(self)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(h, "Data")
        self.tabs.addTab(eeg, "EEG")
        self.tabs.addTab(self.gaze_tab, "Gaze")
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

    def _side_panel(self, title: str, body: QtWidgets.QWidget, min_width: int) -> QtWidgets.QWidget:
        """A titled side panel of the Data tab (part of its splitter, cannot be closed)."""
        panel = QtWidgets.QFrame()
        panel.setObjectName("SidePanel")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QtWidgets.QLabel(title)
        head.setObjectName("PanelTitle")
        v.addWidget(head)
        v.addWidget(body, 1)
        panel.setMinimumWidth(min_width)
        return panel

    def _build_dataset_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.addWidget(self._build_ribbon())
        inner = QtWidgets.QVBoxLayout()
        inner.setContentsMargins(6, 0, 6, 0)
        lay.addLayout(inner, 1)
        lay = inner
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Recording"])
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.itemActivated.connect(self._tree_activated)
        self.tree.itemSelectionChanged.connect(self._tree_selection)
        lay.addWidget(self.tree, 2)
        self.btn_runs = QtWidgets.QPushButton("Load selected runs together")
        self.btn_runs.setToolTip("Select several BIDS runs (⌘/Ctrl- or Shift-click) and load "
                                 "them as one recording; events keep their subject, session "
                                 "and run")
        self.btn_runs.clicked.connect(self._load_selected_runs)
        self.btn_runs.setVisible(False)
        lay.addWidget(self.btn_runs)

        # --- streams of a multi-stream file (XDF) and the role each one plays
        self.streams_box = QtWidgets.QGroupBox("Streams")
        sv = QtWidgets.QVBoxLayout(self.streams_box)
        sv.setContentsMargins(4, 4, 4, 4)
        self.stream_table = QtWidgets.QTableWidget(0, 4)
        self.stream_table.setHorizontalHeaderLabels(["stream", "role", "ch", "Hz"])
        self.stream_table.verticalHeader().setVisible(False)
        self.stream_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.stream_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.stream_table.horizontalHeader().setStretchLastSection(False)
        self.stream_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        self.stream_table.currentCellChanged.connect(lambda r, *_: self._show_stream(r))
        self.stream_table.cellDoubleClicked.connect(lambda r, *_: self._stream_details(r))
        sv.addWidget(self.stream_table)
        row = QtWidgets.QHBoxLayout()
        det = QtWidgets.QPushButton("Details…")
        det.setToolTip("Everything about the selected stream, and how each of its channels "
                       "or columns is used (double-click a stream)")
        det.clicked.connect(lambda: self._stream_details(self.stream_table.currentRow()))
        row.addWidget(det)
        self.btn_roles = QtWidgets.QPushButton("Reload")
        self.btn_roles.setObjectName("Primary")
        self.btn_roles.setToolTip("Reload the file with the roles and stream settings chosen "
                                  "here (resets montage and pipeline results)")
        self.btn_roles.clicked.connect(self._apply_stream_roles)
        self.btn_roles.setEnabled(False)
        auto = QtWidgets.QPushButton("Automatic")
        auto.setToolTip("Back to the roles brAInStudio picks by itself")
        auto.clicked.connect(self._auto_stream_roles)
        row.addWidget(self.btn_roles, 1)
        row.addWidget(auto)
        sv.addLayout(row)
        self.streams_box.setVisible(False)
        lay.addWidget(self.streams_box, 2)
        lay.addWidget(self._build_trigger_box())
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)
        self.info.setMaximumHeight(150)
        lay.addWidget(self.info)
        lay.addWidget(QtWidgets.QLabel("Events (double-click to jump)"))
        self.event_table = QtWidgets.QTableWidget()
        self.event_table.cellDoubleClicked.connect(self._event_jump)
        lay.addWidget(self.event_table, 2)
        self.left_panel = self._side_panel("Dataset", w, 330)
        self.data_split.insertWidget(0, self.left_panel)
        self._update_ribbon()

    def _build_controls_panel(self):
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
        # subjects of combined recordings: jump from one to the next
        self.subj_prev = QtWidgets.QToolButton()
        self.subj_prev.setText("◀")
        self.subj_prev.setToolTip("Previous subject")
        self.subj_next = QtWidgets.QToolButton()
        self.subj_next.setText("▶")
        self.subj_next.setToolTip("Next subject")
        self.subj_box = QtWidgets.QComboBox()
        self.subj_box.setToolTip("Jump to the start of a subject's recording")
        self.subj_box.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.subj_box.setMinimumContentsLength(6)
        self.subj_box.activated.connect(self._jump_subject)
        self.subj_prev.clicked.connect(lambda: self._step_subject(-1))
        self.subj_next.clicked.connect(lambda: self._step_subject(+1))
        self.subj_row = QtWidgets.QWidget()
        sr_l = QtWidgets.QHBoxLayout(self.subj_row)
        sr_l.setContentsMargins(0, 0, 0, 0)
        sr_l.addWidget(self.subj_prev)
        sr_l.addWidget(self.subj_next)
        sr_l.addWidget(self.subj_box, 1)
        self.subj_label = QtWidgets.QLabel("Subject")
        self._subject_spans: list[dict] = []
        f.addRow(self.subj_label, self.subj_row)
        self.subj_label.setVisible(False)
        self.subj_row.setVisible(False)
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
        self.view_form = f
        lay.addWidget(box)

        # --- EEG display (preprocessing itself lives in the EEG tab)
        box = QtWidgets.QGroupBox("EEG display")
        f = QtWidgets.QFormLayout(box)
        self.show_clean = QtWidgets.QRadioButton("Preprocessed (EEG ▸ ① Preprocessing)")
        self.show_raw = QtWidgets.QRadioButton("Raw, as recorded")
        self.show_clean.setChecked(True)
        self.show_clean.toggled.connect(lambda _: self._update_display())
        f.addRow(self.show_clean)
        f.addRow(self.show_raw)
        b = QtWidgets.QPushButton("Edit preprocessing…")
        b.clicked.connect(lambda: (self.tabs.setCurrentIndex(1),
                                   self.eeg_tabs.setCurrentIndex(0)))
        f.addRow(b)
        mb = QtWidgets.QPushButton("Open in MNE browser…")
        mb.clicked.connect(self.open_mne_browser)
        f.addRow(mb)
        lay.addWidget(box)

        # --- gaze
        box = QtWidgets.QGroupBox("Gaze event detection")
        self.gaze_box = box
        f = QtWidgets.QFormLayout(box)
        p = C.DetectionParams()
        # presets: "Automatic" picks one from the data at load (Gazepoint FPOG vs raw)
        self.p_preset = QtWidgets.QComboBox()
        self.p_preset.addItem("Automatic (from the data)", "auto")
        for key, (label, _, why) in C.PRESETS.items():
            self.p_preset.addItem(label, key)
            self.p_preset.setItemData(self.p_preset.count() - 1, why, QtCore.Qt.ToolTipRole)
        self.p_preset.addItem("Custom", "custom")
        self.p_preset.setToolTip("Detection settings for the kind of gaze data. Automatic: "
                                 "Gazepoint data with only the fixation-filtered gaze "
                                 "(FPOG) gets the Gazepoint preset, other data the "
                                 "pymovements defaults. Editing a value makes it Custom.")
        self.p_preset_note = QtWidgets.QLabel("")
        self.p_preset_note.setWordWrap(True)
        self.p_preset_note.setEnabled(False)
        self.p_fix_disp = _spin(1, 500, p.fix_dispersion, 5, 0, " px")
        self.p_fix_disp.setToolTip("I-DT dispersion threshold (pymovements): (max − min x) + "
                                   "(max − min y) of a fixation's samples")
        self.p_fix_dur = _spin(0.01, 2, p.fix_min_dur, 0.01, 3, " s")
        self.p_sac_lambda = _spin(1, 20, p.sac_threshold_factor, 0.5, 1, " λ")
        self.p_sac_lambda.setToolTip("Saccades (pymovements, Engbert & Kliegl): velocity above "
                                     "λ times the median-based velocity spread")
        self.p_sac_dur = _spin(0.001, 0.5, p.sac_min_dur, 0.002, 3, " s")
        self.p_blink_min = _spin(0.005, 2, p.blink_min_dur, 0.01, 3, " s")
        self.p_blink_max = _spin(0.05, 10, p.blink_max_dur, 0.1, 2, " s")
        self.p_velocity = QtWidgets.QComboBox()
        for key, label in (("smooth", "Savitzky-Golay (smooth)"),
                           ("neighbors", "central difference"),
                           ("preceding", "preceding sample")):
            self.p_velocity.addItem(label, key)
        self.p_velocity.setToolTip("How pymovements computes gaze velocity (Gaze.pos2vel) for "
                                   "saccade detection")
        f.addRow("Preset", self.p_preset)
        f.addRow(self.p_preset_note)
        f.addRow("Fixation dispersion", self.p_fix_disp)
        f.addRow("Min fixation", self.p_fix_dur)
        f.addRow("Saccade threshold", self.p_sac_lambda)
        f.addRow("Min saccade", self.p_sac_dur)
        f.addRow("Saccade velocity", self.p_velocity)
        f.addRow("Blink min", self.p_blink_min)
        f.addRow("Blink max", self.p_blink_max)
        self.p_preset.activated.connect(lambda _: self._preset_chosen())
        for spin in (self.p_fix_disp, self.p_fix_dur, self.p_sac_lambda, self.p_sac_dur,
                     self.p_blink_min, self.p_blink_max):
            spin.valueChanged.connect(lambda *_: self._params_edited())
        self.p_velocity.currentIndexChanged.connect(lambda *_: self._params_edited())
        self.cb_pupil_clean = QtWidgets.QCheckBox("Interpolate pupil over blinks")
        self.cb_pupil_clean.setChecked(True)
        f.addRow(self.cb_pupil_clean)
        b = QtWidgets.QPushButton("Detect events")
        b.clicked.connect(lambda: self.detect_gaze_events())
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
        self.right_panel = self._side_panel("Controls", scroll, 280)
        self.data_split.addWidget(self.right_panel)
        self.data_split.setStretchFactor(1, 1)
        self.data_split.setSizes([360, 820, 470, 300])
        for k in range(self.data_split.count()):
            self.data_split.setCollapsible(k, False)

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
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open recording(s) — select several files to combine them", "",
            "Recordings (*.xdf *.xdfz *.vhdr *.edf *.bdf *.set *.fif *.asc);;All files (*)")
        if len(fns) == 1:
            self.open_file(fns[0])
        elif fns:
            self.open_files(fns)


    def _load_async(self, label: str, step: dict, shown, error_title: str) -> bool:
        """Read a recording in the background (the window stays responsive and the
        status bar shows the elapsed time), then show it on the GUI thread."""
        if self.worker.busy:
            QtWidgets.QMessageBox.information(self, "Busy", "Wait for the running "
                                              "computation to finish.")
            return False
        return self.ctl.load(step, label, self.p_preset.currentData(), self._gaze_params(),
                             shown, lambda msg: QtWidgets.QMessageBox.critical(
                                 self, error_title, msg))


    def open_files(self, paths, roles_by_name=None, options_by_name=None):
        """Several files joined into one recording (one segment per file)."""
        paths = [Path(p) for p in paths]
        self._load_async(f"Loading {len(paths)} files…",
                         {"kind": "load_many", "paths": [str(p) for p in paths],
                          "screen_size": tuple(self.screen_size()),
                          "roles_by_name": roles_by_name, "options_by_name": options_by_name},
                         lambda rec: self._show_files(paths, rec), "Could not load the files")


    def _show_files(self, paths, rec):
        self.tree.clear()
        self.bids_root = None
        self.btn_runs.setVisible(False)
        top = QtWidgets.QTreeWidgetItem(self.tree, [f"{len(paths)} files combined"])
        for seg in rec.segments:
            ent = "  ".join(f"{k}-{seg[k]}" for k in ("subject", "session", "run") if seg.get(k))
            it = QtWidgets.QTreeWidgetItem(top, [f"{seg.get('file', '')}"
                                                 + (f"   ({ent})" if ent else "")])
            it.setToolTip(0, f"starts at {seg['start']:.1f} s on the combined time base")
        self.tree.expandAll()
        self.show_recording()


    def action_attach_gaze(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Attach eye-tracking data", "",
            "Eye tracking (*.asc *.tsv *.tsv.gz *.xdf);;All files (*)")
        if fn and self.rec is not None:
            with self._busy("Loading eye tracking…"):
                self.offset.blockSignals(True)
                self.offset.setValue(0)
                self.offset.blockSignals(False)
                self.ctl.attach_gaze(fn, self.screen_size())


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
        desc = C.dataset_description(root)
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
        self._tree_selection()
        self.statusBar().showMessage(f"{len(runs)} runs found in {root}. Double-click a run to "
                                     "load it, or select several and load them together.")
        if len(runs) == 1:
            self.load_run(runs[0])

    def _tree_activated(self, item, _col):
        run = item.data(0, QtCore.Qt.UserRole)
        if run is not None:
            self.load_run(run)

    def _selected_runs(self):
        runs = []
        for it in self.tree.selectedItems():
            stack = [it]
            while stack:  # a subject/session node selects all its runs
                node = stack.pop()
                run = node.data(0, QtCore.Qt.UserRole)
                if run is not None and run not in runs:
                    runs.append(run)
                stack.extend(node.child(i) for i in range(node.childCount()))
        return runs

    def _tree_selection(self):
        n = len(self._selected_runs()) if self.bids_root is not None else 0
        self.btn_runs.setVisible(self.bids_root is not None)
        self.btn_runs.setEnabled(n >= 1)
        self.btn_runs.setText(f"Load {n} selected runs together" if n > 1
                              else "Load selected run")

    def _load_selected_runs(self):
        runs = self._selected_runs()
        if len(runs) == 1:
            self.load_run(runs[0])
        elif runs:
            self._load_bids(runs, f"Loading {len(runs)} runs…", "Could not load the runs")


    def load_run(self, run):
        self._load_bids([run], f"Loading {run.label}…", "Could not load run")

    def _load_bids(self, runs, label, error_title):
        self._load_async(label, {"kind": "load_bids", "root": str(self.bids_root),
                                 "runs": [list(r.key) for r in runs],
                                 "screen_size": self.screen_size_or_none()},
                         lambda rec: self.show_recording(), error_title)


    def open_file(self, path, roles=None, options=None):
        path = Path(path)
        if path.is_dir():
            self.open_bids(path)
            return

        def show(rec):
            self.tree.clear()
            self.bids_root = None
            self.btn_runs.setVisible(False)
            QtWidgets.QTreeWidgetItem(self.tree, [path.name])
            self.show_recording()

        self._load_async(f"Loading {path.name}…",
                         {"kind": "load", "path": str(path),
                          "screen_size": tuple(self.screen_size()), "roles": roles,
                          "options": options}, show, "Could not load file")


    def screen_size(self):
        return (self.scr_w.value(), self.scr_h.value())

    def screen_size_or_none(self):
        return None if self.screen_size() == (1920, 1080) else self.screen_size()

    # ================================================================== state
    def show_recording(self):
        """Show the session's (newly loaded) recording."""
        # one repaint at the end instead of one per table/plot/layout change
        self.setUpdatesEnabled(False)
        try:
            self._show_recording(self.rec)
        finally:
            self.setUpdatesEnabled(True)


    def _show_recording(self, rec: Recording):
        self.stop_play()
        self.offset.blockSignals(True)
        self.offset.setValue(0)
        self.offset.blockSignals(False)
        self.pupil_clean = None
        self.eeg_plot.gaze_events = self.gaze_plot.gaze_events = None
        if rec.has_gaze:
            self.scr_w.setValue(rec.gaze.screen_size[0])
            self.scr_h.setValue(rec.gaze.screen_size[1])
            self.screen.set_screen(rec.gaze.screen_size)
        self.display_raw = rec.raw
        self.screen.reset_view()
        if rec.raw is not None:
            self._load_eeg_plot()
        else:
            self.eeg_plot.data = None
            for c in self.eeg_plot.curves:
                c.setData([], [])
        self.topo.set_raw(self.display_raw)
        self.gaze_plot.set_gaze(rec.gaze)
        self._update_gaze_visibility()
        if self.gaze_events is not None:  # detected while loading
            self._show_gaze_events(rebuilt=False)
        self.info.setPlainText(rec.summary() + ("\n" + rec.description if rec.description else ""))
        self._fill_streams()
        self._fill_trigger_channels()
        self._refresh_events()
        self._fill_event_list()
        self.gaze_tab.update_gaze_stats()
        if rec.raw is not None:  # default preprocessing (0.1–30 Hz, 50 Hz notch) for the viewer
            self.pre_tab.run()
        win = min(self.win_len.value(), max(rec.duration, 0.5))
        if rec.raw is None and rec.has_gaze:
            self.eeg_plot.plotItem.getViewBox().setLimits(xMin=-1, xMax=rec.duration + 1)
        self.eeg_plot.set_window(0, win)
        self.set_cursor(0.0)
        title, tip = self._recording_title(rec)
        self.setWindowTitle(f"brAInStudio — {title}")
        self.title_label.setText(title)
        self.title_label.setToolTip(tip)
        self._update_ribbon()
        self.statusBar().showMessage(rec.summary().replace("\n", "  |  "))


    def _recording_title(self, rec) -> tuple[str, str]:
        """Name shown in the top bar and window title, and its tooltip: the dataset
        folder for BIDS, the file name otherwise ("first + N more" for combined files)."""
        files = [p for p in (rec.sources or [rec.source]) if p]
        if self.bids_root is not None:
            runs = [s.get("file", "") for s in rec.segments if s.get("file")]
            return self.bids_root.name, f"{self.bids_root}\n" + "\n".join(runs)
        if len(files) > 1:
            return (f"{Path(files[0]).name} + {len(files) - 1} more",
                    "\n".join(str(p) for p in files))
        return (Path(files[0]).name if files else "recording"), str(files[0] if files else "")

    def _eeg_picks(self, raw):
        """Channels of the EEG scope: EEG only (EOG/ECG/EMG/misc/stim are hidden);
        all data channels when nothing is typed as EEG."""
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks):
            return picks
        return mne.pick_types(raw.info, meg=True, eog=True, ecg=True, emg=True, misc=True,
                              seeg=True, ecog=True, dbs=True, bio=True, resp=True,
                              fnirs=True, exclude=[])

    def _load_eeg_plot(self):
        raw = self.display_raw
        picks = self._eeg_picks(raw)
        if raw is self.pipeline.clean and raw is not None:
            # preprocessed view: only the channels kept in ① Preprocessing
            drop = set(self.pipeline.params.bads) | set(raw.info["bads"])
            picks = [i for i in picks if raw.ch_names[i] not in drop] or picks
        data = raw.get_data(picks=picks).astype(np.float32)  # display only
        names = [raw.ch_names[i] for i in picks]
        types = raw.get_channel_types(picks=picks)
        self.eeg_plot.set_data(data, raw.times, names, types, raw.info["bads"])
        self.n_chan.setMaximum(max(len(names), 1))
        self.eeg_plot.set_visible_channels(min(self.n_chan.value(), len(names)))

    # ================================================================== trigger channel
    def _build_trigger_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Trigger channel → events")
        box.setCheckable(True)
        box.setChecked(False)
        box.setToolTip("Turn the codes on a trigger channel (EDF/BDF TRIG, Status, …) "
                       "into events")
        v = QtWidgets.QVBoxLayout(box)
        inner = QtWidgets.QWidget()
        iv = QtWidgets.QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        box.toggled.connect(inner.setVisible)
        inner.setVisible(False)
        g = QtWidgets.QGridLayout()
        self.trig_ch = QtWidgets.QComboBox()
        self.trig_mode = QtWidgets.QComboBox()
        self.trig_mode.addItem("integer codes", "codes")
        self.trig_mode.addItem("pulses (threshold)", "threshold")
        self.trig_mode.setToolTip("codes: the channel holds trigger values (EDF/BDF); pulses: "
                                  "an analog channel such as a photodiode, each pulse = 1")
        self.trig_mask = QtWidgets.QSpinBox()
        self.trig_mask.setRange(0, 0xFFFFFF)
        self.trig_mask.setSpecialValueText("all bits")
        self.trig_mask.setToolTip("Bit mask applied to the codes (e.g. 255 keeps the lower "
                                  "8 bits of a BioSemi Status channel)")
        g.addWidget(QtWidgets.QLabel("channel"), 0, 0)
        g.addWidget(self.trig_ch, 0, 1, 1, 3)
        g.addWidget(QtWidgets.QLabel("mode"), 1, 0)
        g.addWidget(self.trig_mode, 1, 1)
        g.addWidget(QtWidgets.QLabel("mask"), 1, 2)
        g.addWidget(self.trig_mask, 1, 3)
        iv.addLayout(g)
        find = QtWidgets.QPushButton("Find events")
        find.clicked.connect(self._find_trigger_events)
        iv.addWidget(find)
        self.trig_table = QtWidgets.QTableWidget(0, 3)
        self.trig_table.setHorizontalHeaderLabels(["code", "count", "name (editable)"])
        self.trig_table.verticalHeader().setVisible(False)
        self.trig_table.horizontalHeader().setStretchLastSection(True)
        self.trig_table.setMaximumHeight(150)
        iv.addWidget(self.trig_table)
        self.trig_stim = QtWidgets.QCheckBox("Leave the trigger channel out of the EEG "
                                             "(type stim)")
        self.trig_stim.setChecked(True)
        iv.addWidget(self.trig_stim)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add as events")
        add.setObjectName("Primary")
        add.setToolTip("Ticked codes become events named as in the table (codes with the same "
                       "name are pooled); replaces earlier trigger events")
        add.clicked.connect(self._add_trigger_events)
        rm = QtWidgets.QPushButton("Remove")
        rm.setToolTip("Remove the events made from the trigger channel")
        rm.clicked.connect(lambda: self._set_trigger_events(None))
        row.addWidget(add, 1)
        row.addWidget(rm)
        iv.addLayout(row)
        v.addWidget(inner)
        self.trigger_box = box
        self._trig_found = None
        return box

    def _fill_trigger_channels(self):
        raw = self.rec.raw if self.rec is not None else None
        self.trigger_box.setVisible(raw is not None)
        self.trig_ch.clear()
        self.trig_table.setRowCount(0)
        self._trig_found = None
        if raw is None:
            return
        self.trig_ch.addItems(raw.ch_names)
        guess = C.guess_trigger_channel(raw)
        if guess:
            self.trig_ch.setCurrentText(guess)
            if not self.rec.streams:  # EDF/BDF/…: open the box when a trigger is likely
                self.trigger_box.setChecked(True)

    def _find_trigger_events(self):
        if self.rec is None or self.rec.raw is None:
            return
        ch = self.trig_ch.currentText()
        try:
            mask, mode = self.trig_mask.value() or None, self.trig_mode.currentData()
            found = C.trigger_codes(self.rec.raw, ch, mask, mode)
        except Exception as e:
            self._error("Trigger channel", e)
            return
        self._trig_found = (ch, found, mask, mode)
        counts = found["code"].value_counts().sort_index()
        t = self.trig_table
        t.setRowCount(len(counts))
        for i, (code, n) in enumerate(counts.items()):
            c = QtWidgets.QTableWidgetItem(str(code))
            c.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            c.setCheckState(QtCore.Qt.Checked)
            c.setData(QtCore.Qt.UserRole, int(code))
            t.setItem(i, 0, c)
            k = QtWidgets.QTableWidgetItem(str(n))
            k.setFlags(QtCore.Qt.ItemIsEnabled)
            t.setItem(i, 1, k)
            t.setItem(i, 2, QtWidgets.QTableWidgetItem(f"trigger_{code}"))
        t.resizeColumnsToContents()
        self.statusBar().showMessage(f"{len(found)} trigger events, {len(counts)} codes on "
                                     f"{ch}", 8000)

    def _add_trigger_events(self):
        if self._trig_found is None:
            self._find_trigger_events()
        if self._trig_found is None:
            return
        ch, found, mask, mode = self._trig_found
        names = {}
        for i in range(self.trig_table.rowCount()):
            it = self.trig_table.item(i, 0)
            if it.checkState() == QtCore.Qt.Checked:
                names[it.data(QtCore.Qt.UserRole)] = \
                    self.trig_table.item(i, 2).text().strip() or f"trigger_{it.text()}"
        self.ctl.set_trigger_events(ch, mask, mode, names, self.trig_stim.isChecked())
        self._after_trigger_events()


    def _set_trigger_events(self, ev):
        """``None`` removes the trigger events."""
        if self.rec is not None and ev is None:
            self.ctl.remove_trigger_events()
            self._after_trigger_events()

    def _after_trigger_events(self, _=None):
        self._refresh_events()
        self.info.setPlainText(self.rec.summary() + ("\n" + self.rec.description
                                                     if self.rec.description else ""))
        n = int(self.rec.events["description"].astype(str)
                .str.contains("|trigger_code=", regex=False).sum())
        self.statusBar().showMessage(f"{n} trigger events in the event table", 6000)

    # ================================================================== streams
    def _fill_streams(self):
        streams = self.rec.streams if self.rec is not None else []
        self._stream_opts = {st["key"]: dict(st.get("options") or {}) for st in streams}
        self.streams_box.setVisible(bool(streams))
        t = self.stream_table
        t.setRowCount(len(streams))
        for i, st in enumerate(streams):
            name = QtWidgets.QTableWidgetItem(st["name"] or f"stream {st['key']}")
            name.setData(QtCore.Qt.UserRole, st["key"])
            labels = ", ".join(st["labels"][:16]) + (" …" if len(st["labels"]) > 16 else "")
            name.setToolTip(f"{st['name']}  (id {st['key']}, type {st['type']})\n"
                            f"{st['samples']} samples, "
                            f"{st['format']}\nautomatic role: {st['auto_role']}\n"
                            f"channels: {labels}")
            name.setFlags(name.flags() & ~QtCore.Qt.ItemIsEditable)
            t.setItem(i, 0, name)
            combo = QtWidgets.QComboBox()
            combo.addItems(list(C.ROLES))
            combo.setCurrentText(st["role"])
            combo.setToolTip(f"automatic: {st['auto_role']}")
            combo.currentTextChanged.connect(lambda _=None: self._roles_edited())
            t.setCellWidget(i, 1, combo)
            for j, v in enumerate((str(st["channels"]),
                                   f"{st['srate']:g}" if st["srate"] else "irreg."), start=2):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                t.setItem(i, j, it)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self._roles_edited()

    def stream_roles(self) -> dict[str, str]:
        t = self.stream_table
        return {t.item(i, 0).data(QtCore.Qt.UserRole): t.cellWidget(i, 1).currentText()
                for i in range(t.rowCount())}

    def _roles_edited(self):
        if self.rec is None:
            return
        cur = {s["key"]: s["role"] for s in self.rec.streams}
        new = self.stream_roles()
        applied = {s["key"]: (s.get("options") or {}) for s in self.rec.streams}
        opts_changed = {k for k, o in getattr(self, "_stream_opts", {}).items()
                        if o != applied.get(k, {})}
        changed = new != cur or bool(opts_changed)
        self.btn_roles.setEnabled(changed)
        bold = QtGui.QFont()
        for i in range(self.stream_table.rowCount()):
            k = self.stream_table.item(i, 0).data(QtCore.Qt.UserRole)
            f = QtGui.QFont(bold)
            f.setBold(new.get(k) != cur.get(k) or k in opts_changed)
            self.stream_table.item(i, 0).setFont(f)

    def _show_stream(self, row):
        if self.rec is None or not (0 <= row < len(self.rec.streams)):
            return
        st = self.rec.streams[row]
        self.statusBar().showMessage(
            f"{st['name']}: {st['type']}, {st['channels']} ch "
            f"({', '.join(st['labels'][:8])}{' …' if len(st['labels']) > 8 else ''}), "
            f"{st['samples']} samples — role {st['role']} (automatic: {st['auto_role']})", 10000)

    def _stream_details(self, row: int):
        if self.rec is None or not (0 <= row < len(self.rec.streams)):
            return
        st = self.rec.streams[row]
        combo = self.stream_table.cellWidget(row, 1)
        role = combo.currentText()
        dlg = StreamDialog(st, role, self._stream_opts.get(st["key"]) if role == st["role"]
                           else None, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        new_role, opts = dlg.result_options()
        self._stream_opts[st["key"]] = opts
        combo.blockSignals(True)
        combo.setCurrentText(new_role)
        combo.blockSignals(False)
        self._roles_edited()
        if self.btn_roles.isEnabled():
            self.statusBar().showMessage("Stream settings changed — press “Reload” in the "
                                         "Streams box to apply them.", 8000)

    def _auto_stream_roles(self):
        if self.rec is None:
            return
        self._stream_opts = {s["key"]: {} for s in self.rec.streams}
        auto = {s["key"]: s["auto_role"] for s in self.rec.streams}
        for i in range(self.stream_table.rowCount()):
            k = self.stream_table.item(i, 0).data(QtCore.Qt.UserRole)
            self.stream_table.cellWidget(i, 1).setCurrentText(auto[k])
        self._roles_edited()
        if self.btn_roles.isEnabled():
            self._apply_stream_roles()

    def _apply_stream_roles(self):
        if self.rec is None or self.rec.source is None:
            return
        roles = self.stream_roles()
        for role in ("EEG", "gaze"):
            n = sum(r == role for r in roles.values())
            if n > 1:
                QtWidgets.QMessageBox.warning(self, "Stream roles",
                                              f"Only one stream can be {role}.")
                return
        # options only apply to the role they were made for
        opts = {k: o for k, o in self._stream_opts.items() if o}
        if len(self.rec.sources) > 1:  # several files: the same settings by stream name
            name = {s["key"]: s["name"] for s in self.rec.streams}
            self.open_files(self.rec.sources, {name[k]: r for k, r in roles.items()},
                            {name[k]: o for k, o in opts.items()} or None)
            return
        self.open_file(self.rec.source, roles=roles, options=opts or None)

    # ================================================================== pipeline
    def _on_pipeline(self, what):
        if what == "preprocessing":
            self._update_display()
        elif what == "events":
            self._fill_event_list()
        elif what == "gaze":
            self._after_gaze_change()
        elif what == "gaze_events":
            self._show_gaze_events()
        elif what == "channels":
            self._channels_changed()

    def _channels_changed(self):
        """Channels renamed or retyped (montage, trigger channel): the preprocessing
        views follow and the viewer's default preprocessing runs again."""
        self.pre_tab.refresh_channels()
        self.pre_tab.refresh_reference_choices()
        self.pre_tab._fill_ica_table()
        self.display_raw = self.rec.raw
        self._load_eeg_plot()
        self.topo.set_raw(self.display_raw)
        self.pre_tab.run()


    def _update_display(self):
        if self.rec is None or self.rec.raw is None:
            return
        clean = self.pipeline.clean
        self.display_raw = clean if (self.show_clean.isChecked() and clean is not None) \
            else self.rec.raw
        self._load_eeg_plot()
        self.topo.set_raw(self.display_raw)

    def _update_pipeline_titles(self):
        mark = {"pending": "○", "ok": "●", "stale": "▲"}
        st = self.pipeline.state
        names = ("① Preprocessing", "② Epoching", "③ Analysis")
        for i, (stage, name) in enumerate(zip(STAGES, names)):
            self.eeg_tabs.setTabText(i, f"{name} {mark[st[stage]]}")
        words = {"pending": "not run", "ok": "up to date", "stale": "out of date"}
        self.pipe_label.setText(
            "Pipeline:  " + "  →  ".join(f"{n} <i>{words[st[s]]}</i>"
                                         for s, n in zip(STAGES, names))
            + "   <span style='color:gray'>(● up to date · ▲ out of date · ○ not run)</span>")

    def run_pipeline(self):
        """Run the first stage that is not up to date and everything after it."""
        st = self.pipeline.state
        stages = [self.pre_tab.run, self.epoch_tab.run, self.eeg_analysis.run]
        first = next((i for i, s in enumerate(STAGES) if st[s] != "ok"), None)
        if first is None:
            self.statusBar().showMessage("Every stage is up to date.", 4000)
            return

        def chain(i):
            if i >= len(stages):
                return
            self.eeg_tabs.setCurrentIndex(i)
            stages[i](then=lambda: chain(i + 1))

        chain(first)

    def open_mne_browser(self):
        if self.display_raw is None:
            return
        raw = self.display_raw.copy()
        if self.gaze_events is not None:
            C.add_gaze_annotations(raw, self.gaze_events, ("blinks", "saccades"))
        t0, t1 = self.eeg_plot.window()
        raw.plot(start=max(t0, 0), duration=t1 - t0, block=False, show=True,
                 scalings=dict(eeg=self.eeg_scale.value() * 1e-6,
                               eog=self.eeg_scale.value() * 4e-6))

    def _gaze_params(self, gaze=None):
        """The detection settings: the preset's, chosen from ``gaze`` when Automatic,
        else the values in the Controls panel."""
        key = self.p_preset.currentData()
        if key == "auto":
            return C.PRESETS[S.suggest_preset(gaze if gaze is not None else
                                              (self.rec.gaze if self.rec else None))][1]
        if key in C.PRESETS:
            return C.PRESETS[key][1]
        return C.DetectionParams(
            blink_min_dur=self.p_blink_min.value(), blink_max_dur=self.p_blink_max.value(),
            fix_dispersion=self.p_fix_disp.value(), fix_min_dur=self.p_fix_dur.value(),
            sac_threshold_factor=self.p_sac_lambda.value(), sac_min_dur=self.p_sac_dur.value(),
            velocity_method=self.p_velocity.currentData())

    def _show_params(self, p):
        """Show ``p`` in the controls without making the preset Custom."""
        self._showing_params = True
        try:
            self.p_fix_disp.setValue(p.fix_dispersion)
            self.p_fix_dur.setValue(p.fix_min_dur)
            self.p_sac_lambda.setValue(p.sac_threshold_factor)
            self.p_sac_dur.setValue(p.sac_min_dur)
            self.p_blink_min.setValue(p.blink_min_dur)
            self.p_blink_max.setValue(p.blink_max_dur)
            self.p_velocity.setCurrentIndex(max(self.p_velocity.findData(p.velocity_method), 0))
        finally:
            self._showing_params = False
        key = C.preset_name(p)
        auto = self.p_preset.currentData() == "auto"
        self.p_preset_note.setText(f"→ {C.PRESETS[key][0]}" if auto and key else "")
        self.p_preset_note.setToolTip(C.PRESETS[key][2] if key else "")
        self.p_preset.setToolTip(C.PRESETS[key][2] if key else
                                 "Detection settings for the kind of gaze data.")

    def _preset_chosen(self):
        key = self.p_preset.currentData()
        if key == "custom":
            self.p_preset_note.setText("")
            return
        self._show_params(self._gaze_params())

    def _params_edited(self):
        if getattr(self, "_showing_params", False):
            return
        self.p_preset.setCurrentIndex(self.p_preset.findData("custom"))
        self.p_preset_note.setText("")

    def detect_gaze_events(self, rebuild: bool = True):
        """Detect fixations, saccades and blinks (pymovements) in the background."""
        if self.rec is not None and self.rec.has_gaze:
            self.ctl.detect_gaze(self._gaze_params(), rebuild)


    def _show_gaze_events(self, rebuilt: bool = True):
        ev = self.gaze_events
        self._show_params(self.session.gaze_params)
        self.pupil_clean = (S.clean_pupil(self.rec.gaze, ev.blinks)
                            if self.cb_pupil_clean.isChecked() else None)
        self.gaze_plot.set_gaze(self.rec.gaze, self.pupil_clean)
        self.eeg_plot.gaze_events = ev
        self.gaze_plot.gaze_events = ev
        self._overlay_toggled()
        self._heat_toggled()
        if rebuilt:
            self.gaze_tab.update_gaze_stats()
            self.pre_tab.gaze_events_changed()  # ICA scores and blink check use the blinks
        self._update_screen()
        self._update_ribbon()


    # ================================================================== events / montage

    def _fill_event_list(self):
        """Dataset panel: the markers with every column they carry."""
        df = self.event_df
        m = df[df["kind"] == "marker"] if len(df) else df
        cols = ["onset", "name", "duration", "phase"] + [
            c for c in m.columns if c not in C.BASE_COLUMNS + ["phase", "x", "y",
                                                                "amplitude", "peak_velocity"]
            and m[c].notna().any() and (m[c].astype(str) != "").any()]
        cols = [c for c in cols if c in m.columns]
        view = m[cols].copy()
        for c in view.columns:  # empty cells instead of nan/None
            if view[c].dtype == object:
                view[c] = view[c].fillna("")
        fill_table(self.event_table, view.iloc[:5000], fmt="{:.3f}")

    def _aois_changed(self):
        self.gaze_tab.update_aoi_stats()
        self.ctl.set_aois(self.screen.aoi_specs())


    def action_montage(self):
        if self.rec is None or self.rec.raw is None:
            QtWidgets.QMessageBox.information(self, "Montage", "Load a recording with EEG first.")
            return
        if self.worker.busy:
            QtWidgets.QMessageBox.information(self, "Montage", "Wait for the running "
                                              "computation to finish.")
            return
        dlg = MontageDialog(self.rec.raw, self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        try:
            with self._busy("Applying montage…"):
                rep = self.ctl.set_montage(dlg.montage_source, dlg.names_out,
                                           self.pre_tab.bads())
        except Exception as e:
            self._error("Montage", e)
            return
        self.info.setPlainText(self.rec.summary() + "\n" + self.rec.description)
        self.statusBar().showMessage(
            f"Montage applied: {rep['n_renamed']} channels renamed, "
            f"{rep['n_positioned']} EEG channels with positions", 8000)


    # ================================================================== theme
    def _apply_theme(self):
        apply_theme(QtWidgets.QApplication.instance(), self)
        for w in (self.eeg_plot, self.gaze_plot, self.overview, self.screen, self.topo):
            w.restyle()
        for w in (self.pre_tab, self.epoch_tab, self.eeg_analysis, self.gaze_tab):
            w.restyle()
        self._heat_toggled()

    def changeEvent(self, ev):
        if ev.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange) \
                and hasattr(self, "gaze_tab"):
            dark = QtWidgets.QApplication.instance().palette().color(
                QtGui.QPalette.Base).lightness() < 128
            if dark != THEME.dark:
                QtCore.QTimer.singleShot(0, self._apply_theme)
        super().changeEvent(ev)

    def _update_gaze_visibility(self):
        """Eye-data views (gaze/pupil plots, screen view, Gaze tab, gaze controls) are
        shown only when the recording has gaze data."""
        has = self.rec is not None and self.rec.has_gaze
        self.gaze_plot.setVisible(has)
        self.screen.setVisible(has)
        self.tabs.setTabVisible(self.tabs.indexOf(self.gaze_tab), has)
        self.gaze_box.setVisible(has)
        for w in (*self.ov_checks.values(), self.cb_scan, self.cb_heat, self.trail):
            self.view_form.setRowVisible(w, has)

    def _after_gaze_change(self):
        g = self.rec.gaze
        self._update_gaze_visibility()
        self.gaze_plot.set_gaze(g)
        self.screen.set_screen(g.screen_size)
        self.detect_gaze_events()
        self.overview.set_recording(self.rec.duration, self.rec.all_events(), g)
        self.info.setPlainText(self.rec.summary())

    def _offset_changed(self):
        self.ctl.set_gaze_offset(self.offset.value() / 1000.0)


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
        self._show_segments()

    def _show_segments(self):
        """Where each subject's (and run's) recording starts, and the subject jump list."""
        segs = self.rec.segments if self.rec is not None else []
        self.eeg_plot._set_segment_lines(self.eeg_plot.plotItem, segs)
        self.gaze_plot._set_segment_lines(self.gaze_plot.pos_plot, segs, labels=False)
        spans = subject_spans(segs)
        self._subject_spans = spans
        self.overview.set_subjects(spans)
        self.subj_box.blockSignals(True)
        self.subj_box.clear()
        for sp in spans:
            self.subj_box.addItem(sp["label"])
            self.subj_box.setItemData(self.subj_box.count() - 1,
                                      f"{sp['start']:.1f}–{sp['end']:.1f} s",
                                      QtCore.Qt.ToolTipRole)
        self.subj_box.blockSignals(False)
        many = len(spans) >= 2
        self.subj_label.setVisible(many)
        self.subj_row.setVisible(many)
        self._sync_subject()

    def _subject_at(self, t: float) -> int:
        for k, sp in enumerate(self._subject_spans):
            if sp["start"] <= t < sp["end"]:
                return k
        return len(self._subject_spans) - 1 if self._subject_spans else -1

    def _jump_subject(self, k: int):
        if not 0 <= k < len(self._subject_spans):
            return
        sp = self._subject_spans[k]
        t0, t1 = self.eeg_plot.window()
        w = t1 - t0
        self.eeg_plot.set_window(sp["start"], sp["start"] + w)
        self.set_cursor(sp["start"])
        self.statusBar().showMessage(f"{sp['label']}: {sp['start']:.1f}–{sp['end']:.1f} s", 4000)

    def _step_subject(self, step: int):
        if not self._subject_spans:
            return
        t0, _ = self.eeg_plot.window()
        k = self._subject_at(t0 + 1e-6)
        self._jump_subject(int(np.clip(k + step, 0, len(self._subject_spans) - 1)))

    def _sync_subject(self):
        if not self._subject_spans:
            return
        t0, t1 = self.eeg_plot.window()
        k = self._subject_at((t0 + t1) / 2)
        if k != self.subj_box.currentIndex():
            self.subj_box.blockSignals(True)
            self.subj_box.setCurrentIndex(k)
            self.subj_box.blockSignals(False)
        self.subj_prev.setEnabled(k > 0 or t0 > self._subject_spans[0]["start"] + 1e-6)
        self.subj_next.setEnabled(k < len(self._subject_spans) - 1)

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
            hm = C.heatmap(self.gaze_events.fixations, self.screen.screen)
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
        self._sync_subject()
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
        if self.tabs.widget(i) is self.gaze_tab:
            self.gaze_tab.update_aoi_stats()


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

        def __exit__(self, *exc):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.win.statusBar().clearMessage()
            return False

    def _busy(self, msg):
        return self._BusyCtx(self, msg)

    def _error(self, title, e: Exception):
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(self, title, f"{type(e).__name__}: {e}")
