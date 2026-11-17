"""Pipeline stage 3: ERP and band-power analyses of the epoch groups, plus
cognitive-state indices, all compared across groups."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import bandpower as BP
from .. import eeg as E
from .. import stats as ST
from .theme import THEME
from .widgets import (NO_POS, StageHeader, _spin, _themed_plot, fill_table, has_positions,
                      hint, parse_channels)


def _sem(x: np.ndarray, axis=0) -> np.ndarray:
    n = np.sum(np.isfinite(x), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanstd(x, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


def _bars(plot: pg.PlotItem, xs, means, sems, color, width):
    plot.addItem(pg.BarGraphItem(x=xs, height=np.nan_to_num(means), width=width, brush=color,
                                 pen=None))
    ok = np.isfinite(means) & np.isfinite(sems)
    if ok.any():
        plot.addItem(pg.ErrorBarItem(x=np.asarray(xs)[ok], y=np.asarray(means)[ok],
                                     height=2 * np.asarray(sems)[ok], beam=width * 0.4,
                                     pen=pg.mkPen(THEME.fg, width=1.2)))


class AnalysisTab(QtWidgets.QWidget):
    def __init__(self, pipeline, worker, parent=None):
        super().__init__(parent)
        self.pl, self.worker = pipeline, worker
        self.erp = None
        self.bp = None
        self.tc = None
        self.idx = None
        outer = QtWidgets.QVBoxLayout(self)
        self.header = StageHeader("③ Analysis", "Run all analyses ▶")
        self.header.run.clicked.connect(lambda: self.run())
        outer.addWidget(self.header)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Groups to compare:"))
        self.group_list = QtWidgets.QListWidget()
        self.group_list.setFlow(QtWidgets.QListView.LeftToRight)
        self.group_list.setMaximumHeight(34)
        self.group_list.itemChanged.connect(lambda *_: self.pl.mark_stale("analysis"))
        row.addWidget(self.group_list, 1)
        row.addWidget(hint("Two groups: Welch t / Mann-Whitney, Hedges g (first − second), "
                           "cluster test. More: ANOVA / Kruskal. Epoch-level statistics."))
        outer.addLayout(row)
        self.tabs = QtWidgets.QTabWidget()
        outer.addWidget(self.tabs, 1)
        self._build_erp()
        self._build_bp()
        self._build_idx()
        self.pl.changed.connect(self._on_pipeline)
        self.pl.stateChanged.connect(self._on_state)

    # ================================================================== common
    def _on_pipeline(self, what):
        if what in ("epoching", "recording"):
            prev = {self.group_list.item(i).text(): self.group_list.item(i).checkState()
                    for i in range(self.group_list.count())}
            self.group_list.blockSignals(True)
            self.group_list.clear()
            for i, (g, ep) in enumerate(self.pl.epochs.items()):
                it = QtWidgets.QListWidgetItem(g)
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(prev.get(g, QtCore.Qt.Checked))
                it.setForeground(pg.mkColor(THEME.group(i)))
                it.setToolTip(f"{len(ep)} epochs — {self.pl.group_desc.get(g, '')}")
                self.group_list.addItem(it)
            self.group_list.blockSignals(False)
            self._fix_baseline_range()
        if what in ("events", "recording"):
            self._fill_baseline_events()
        self._on_state()

    def _on_state(self):
        if not self.pl.epochs:
            self.header.set_input("Input: no epochs yet (run ② Epoching)")
        else:
            gs = "  ·  ".join(f"{g}: {len(ep)}" for g, ep in self.pl.epochs.items())
            self.header.set_input(f"Input: epochs {self.pl.epoch_settings.describe()} — {gs}")
        self.header.set_state(self.pl.state["analysis"])

    def groups(self) -> dict[str, mne.Epochs]:
        out = {}
        for i in range(self.group_list.count()):
            it = self.group_list.item(i)
            if it.checkState() == QtCore.Qt.Checked and it.text() in self.pl.epochs:
                out[it.text()] = self.pl.epochs[it.text()]
        return out

    def _color(self, g: str, alpha=255):
        keys = list(self.pl.epochs)
        return THEME.s(keys.index(g) if g in keys else 0, alpha)

    def _need_groups(self):
        g = self.groups()
        if not g:
            QtWidgets.QMessageBox.information(self, "Analysis", "Build epochs (② Epoching) and "
                                              "tick at least one group.")
            return None
        return g

    def run(self, then=None):
        """Run ERP, band power and indices in sequence (pipeline 'run all')."""
        def after_idx():
            self.pl.set_analysis_done()
            if then:
                then()
        return self.compute_erp(then=lambda: self.compute_bp(
            then=lambda: self.compute_idx(then=after_idx)))

    def _roi(self, edit: QtWidgets.QLineEdit, ch_names: list[str]) -> list[str]:
        found, missing = parse_channels(edit.text(), ch_names)
        if edit.text().strip() and not found:
            raise ValueError(f"None of the ROI channels are available: {', '.join(missing)} "
                             "(bad channels are excluded from epochs)")
        return found or list(ch_names)

    def restyle(self):
        for p in (self.erp_plot, self.bp_plot, self.tc_plot):
            p.setBackground(THEME.bg)
        self.idx_plots.setBackground(THEME.bg)
        for i in range(self.group_list.count()):
            self.group_list.item(i).setForeground(pg.mkColor(THEME.group(i)))
        self._plot_erp()
        self._plot_bp()
        self._plot_tc()
        self._plot_idx()

    # ================================================================== ERP
    def _build_erp(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGridLayout()
        self.erp_bl_on = QtWidgets.QCheckBox("Baseline")
        self.erp_bl_on.setChecked(True)
        self.erp_bl0 = _spin(-60, 0, -0.2, 0.05, 3, " s")
        self.erp_bl1 = _spin(-60, 1, 0.0, 0.05, 3, " s")
        for x in (self.erp_bl0, self.erp_bl1):
            x.setToolTip("Pre-event baseline window, subtracted from every epoch")
        self.erp_roi = QtWidgets.QLineEdit()
        self.erp_roi.setPlaceholderText("ROI, e.g. Pz, P3, P4  (empty = all good channels)")
        self.erp_m0 = _spin(-60, 120, 0.3, 0.05, 3, " s")
        self.erp_m1 = _spin(-60, 120, 0.5, 0.05, 3, " s")
        self.erp_perm = QtWidgets.QSpinBox()
        self.erp_perm.setRange(0, 100000)
        self.erp_perm.setValue(1000)
        self.erp_perm.setSpecialValueText("no test")
        self.erp_spatial = QtWidgets.QCheckBox("spatio-temporal (all channels)")
        self.erp_post = QtWidgets.QCheckBox("test from 0 s")
        self.erp_post.setChecked(True)
        g.addWidget(self.erp_bl_on, 0, 0)
        g.addWidget(self.erp_bl0, 0, 1)
        g.addWidget(QtWidgets.QLabel("to"), 0, 2)
        g.addWidget(self.erp_bl1, 0, 3)
        g.addWidget(QtWidgets.QLabel("ROI"), 0, 4)
        g.addWidget(self.erp_roi, 0, 5, 1, 3)
        g.addWidget(QtWidgets.QLabel("measure window"), 1, 0)
        g.addWidget(self.erp_m0, 1, 1)
        g.addWidget(QtWidgets.QLabel("to"), 1, 2)
        g.addWidget(self.erp_m1, 1, 3)
        g.addWidget(QtWidgets.QLabel("cluster test"), 1, 4)
        g.addWidget(self.erp_perm, 1, 5)
        g.addWidget(self.erp_spatial, 1, 6)
        g.addWidget(self.erp_post, 1, 7)
        g.setColumnStretch(5, 1)
        lay.addLayout(g)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Compute ERP ▶")
        b.clicked.connect(lambda: self.compute_erp())
        row.addWidget(b)
        for label, fn in (("Compare evokeds (MNE)…", self.erp_compare),
                          ("Joint plot (MNE)…", self.erp_joint),
                          ("Topomaps in window (MNE)…", self.erp_topo),
                          ("Export…", self.erp_export)):
            bb = QtWidgets.QPushButton(label)
            bb.clicked.connect(fn)
            row.addWidget(bb)
        row.addStretch()
        lay.addLayout(row)
        self.erp_info = hint("")
        lay.addWidget(self.erp_info)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.erp_plot = _themed_plot(bottom=("Time", "s"), left="Amplitude (µV)")
        self.erp_plot.addLegend()
        split.addWidget(self.erp_plot)
        tabs = QtWidgets.QTabWidget()
        self.erp_table = QtWidgets.QTableWidget()
        self.erp_clusters = QtWidgets.QTableWidget()
        tabs.addTab(self.erp_table, "Measures in window")
        tabs.addTab(self.erp_clusters, "Clusters")
        split.addWidget(tabs)
        split.setSizes([520, 200])
        lay.addWidget(split, 1)
        self.tabs.addTab(w, "ERP")

    def _fix_baseline_range(self):
        st = self.pl.epoch_settings
        for x in (self.erp_bl0, self.erp_bl1, self.erp_m0, self.erp_m1):
            x.setRange(st.tmin, st.tmax)
        if self.erp_bl0.value() < st.tmin:
            self.erp_bl0.setValue(st.tmin)

    def compute_erp(self, then=None):
        groups = self._need_groups()
        if groups is None:
            return False
        first = next(iter(groups.values()))
        try:
            roi = self._roi(self.erp_roi, first.ch_names)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "ROI", str(e))
            return False
        bl = (self.erp_bl0.value(), self.erp_bl1.value()) if self.erp_bl_on.isChecked() else None
        if bl and bl[1] <= bl[0]:
            QtWidgets.QMessageBox.warning(self, "Baseline", "The baseline end must be after "
                                          "its start.")
            return False
        win = (self.erp_m0.value(), self.erp_m1.value())
        n_perm = self.erp_perm.value()
        spatial = self.erp_spatial.isChecked() and has_positions(first)
        post = self.erp_post.isChecked()

        def job():
            notes = []
            eps = {g: (ep.copy().apply_baseline(bl, verbose="error") if bl else ep)
                   for g, ep in groups.items()}
            evokeds, sem, per = {}, {}, []
            for g, ep in eps.items():
                ev = ep.average()
                ev.comment = g
                evokeds[g] = ev
                sem[g] = _sem(ep.get_data(picks=roi).mean(1)) * 1e6
                m = E.erp_measures(ep, roi, win)
                m["group"] = g
                per.append(m)
            per = pd.concat(per, ignore_index=True)
            table = ST.comparison_table(per, [c for c in per.columns if c != "group"],
                                        list(eps))
            stats, diff = None, None
            if len(eps) == 2:
                (la, a), (lb, b) = eps.items()
                diff = mne.combine_evoked([evokeds[la], evokeds[lb]], weights=[1, -1])
                diff.comment = f"{la} − {lb}"
                ov = len(set(a.events[:, 0]) & set(b.events[:, 0]))
                if ov:
                    notes.append(f"⚠ {ov} events are in both groups (tests assume independence)")
                if n_perm:
                    try:
                        stats = E.cluster_contrast(a, b, roi, n_perm,
                                                   tmin=0.0 if post else None, spatial=spatial)
                        if stats.get("excluded"):
                            notes.append("flat channels left out of the test: "
                                         + ", ".join(stats["excluded"]))
                    except Exception as e:  # report, keep the ERPs
                        notes.append(f"cluster test failed: {e}")
            elif n_perm:
                notes.append("cluster test needs exactly two groups")
            return dict(evokeds=evokeds, sem=sem, per_epoch=per, table=table, stats=stats,
                        diff=diff, roi=roi, baseline=bl, window=win, notes=notes)

        def done(res):
            self.erp = res
            self._plot_erp()
            if then:
                then()

        return self.worker.run("Computing ERPs…", job, done)

    def _plot_erp(self):
        p = self.erp_plot
        p.clear()
        self.erp_clusters.clear()
        self.erp_clusters.setRowCount(0)
        r = self.erp
        if r is None:
            self.erp_table.clear()
            return
        roi = r["roi"]
        for g, ev in r["evokeds"].items():
            t = ev.times
            y = ev.copy().pick(roi).data.mean(0) * 1e6
            lo = p.plot(t, y - r["sem"][g], pen=None)
            hi = p.plot(t, y + r["sem"][g], pen=None)
            p.addItem(pg.FillBetweenItem(lo, hi, brush=self._color(g, 50)))
            p.plot(t, y, pen=pg.mkPen(self._color(g), width=2.2), name=f"{g} (n={ev.nave})")
        if r["diff"] is not None:
            d = r["diff"].copy().pick(roi).data.mean(0) * 1e6
            p.plot(r["diff"].times, d, pen=pg.mkPen(THEME.fg, width=1.5,
                                                     style=QtCore.Qt.DashLine),
                   name=r["diff"].comment)
        p.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(THEME.muted)))
        if r["baseline"]:
            reg = pg.LinearRegionItem(r["baseline"], movable=False,
                                      brush=THEME.rgba("muted", 35), pen=pg.mkPen(None))
            reg.setZValue(-30)
            p.addItem(reg)
        reg = pg.LinearRegionItem(r["window"], movable=False, brush=THEME.s(3, 30),
                                  pen=pg.mkPen(THEME.s(3, 120)))
        reg.setZValue(-25)
        p.addItem(reg)
        rows = []
        st = r["stats"]
        if st:
            for k, (mask, pv) in enumerate(zip(st["clusters"], st["p_values"])):
                tm = mask.any(axis=1) if mask.ndim == 2 else mask
                if not tm.any():
                    continue
                t0, t1 = float(st["times"][tm][0]), float(st["times"][tm][-1])
                rows.append({"cluster": k + 1, "tmin (s)": t0, "tmax (s)": t1,
                             "channels": int(mask.any(axis=0).sum()) if mask.ndim == 2
                             else len(roi), "p": float(pv)})
                if pv < 0.05:
                    reg = pg.LinearRegionItem((t0, t1), movable=False,
                                              brush=THEME.rgba("accent", 55), pen=pg.mkPen(None))
                    reg.setZValue(-20)
                    p.addItem(reg)
            if rows:
                fill_table(self.erp_clusters, pd.DataFrame(rows).sort_values("p"), fmt="{:.4g}")
        fill_table(self.erp_table, r["table"], fmt="{:.4g}")
        sig = sum(1 for x in rows if x["p"] < 0.05)
        roi_txt = "all good channels" if len(roi) == len(next(iter(r["evokeds"].values())).ch_names) \
            else ", ".join(roi)
        test = ""
        if st:
            test = (f"  ·  {'spatio-temporal' if st['spatial'] else 'ROI'} cluster test: "
                    f"{len(st['p_values'])} clusters, {sig} with p < 0.05")
        bl = f"{r['baseline'][0]:g}…{r['baseline'][1]:g} s" if r["baseline"] else "none"
        self.erp_info.setText(f"ROI: {roi_txt}  ·  baseline {bl}  ·  measures in "
                              f"{r['window'][0]:g}…{r['window'][1]:g} s{test}"
                              + ("\n" + "; ".join(r["notes"]) if r["notes"] else ""))

    def erp_compare(self):
        r = self.erp
        if r is None:
            return
        cols = {g: THEME.group(list(self.pl.epochs).index(g)) for g in r["evokeds"]}
        try:
            mne.viz.plot_compare_evokeds(r["evokeds"], picks=r["roi"],
                                         combine="mean" if len(r["roi"]) > 1 else None,
                                         colors=cols, show=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Compare evokeds", str(e))

    def erp_joint(self):
        r = self.erp
        if r is None:
            return
        ev = r["diff"] if r["diff"] is not None else next(iter(r["evokeds"].values()))
        try:
            if has_positions(ev):
                ev.plot_joint(title=ev.comment, show=True)
            else:
                ev.plot(gfp=True, show=True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Joint plot", str(e))

    def erp_topo(self):
        r = self.erp
        if r is None:
            return
        evs = dict(r["evokeds"])
        if r["diff"] is not None:
            evs[r["diff"].comment] = r["diff"]
        if not has_positions(next(iter(evs.values()))):
            QtWidgets.QMessageBox.information(self, "Topomaps", NO_POS)
            return
        import matplotlib.pyplot as plt
        t0, t1 = r["window"]
        fig, axes = plt.subplots(1, len(evs), figsize=(3 * len(evs), 3.2), squeeze=False)
        data = {g: ev.copy().crop(t0, t1).data.mean(1) * 1e6 for g, ev in evs.items()}
        lim = max(np.nanmax(np.abs(v)) for v in data.values())
        for ax, (g, ev) in zip(axes[0], evs.items()):
            mne.viz.plot_topomap(data[g], ev.info, axes=ax, show=False, vlim=(-lim, lim),
                                 cmap="RdBu_r", contours=4)
            ax.set_title(g, fontsize=10)
        fig.suptitle(f"Mean amplitude {t0:g}…{t1:g} s (±{lim:.1f} µV)")
        plt.show()

    def erp_export(self):
        r = self.erp
        if r is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export ERP results to folder")
        if not d:
            return
        out = Path(d)
        evs = list(r["evokeds"].values()) + ([r["diff"]] if r["diff"] is not None else [])
        mne.write_evokeds(out / "erp-ave.fif", evs, overwrite=True, verbose="error")
        r["table"].to_csv(out / "erp_measures_comparison.csv", index=False)
        r["per_epoch"].to_csv(out / "erp_measures_per_epoch.csv", index=False)
        if r["stats"]:
            st = r["stats"]
            rows = []
            for k, (mask, pv) in enumerate(zip(st["clusters"], st["p_values"])):
                tm = mask.any(axis=1) if mask.ndim == 2 else mask
                if tm.any():
                    rows.append({"cluster": k + 1, "tmin": st["times"][tm][0],
                                 "tmax": st["times"][tm][-1], "p": pv})
            pd.DataFrame(rows).to_csv(out / "erp_clusters.csv", index=False)
        self._write_provenance(out / "erp.txt", {"ROI": ", ".join(r["roi"]),
                                                 "baseline": r["baseline"],
                                                 "measure window": r["window"]})

    def _write_provenance(self, path: Path, extra: dict):
        lines = [f"preprocessing: {self.pl.params.describe()}",
                 f"epochs: {self.pl.epoch_settings.describe()}"]
        lines += [f"group {g}: {d}" for g, d in self.pl.group_desc.items()]
        lines += [f"{k}: {v}" for k, v in extra.items()]
        path.write_text("\n".join(lines) + "\n")
        self.window().statusBar().showMessage(f"Exported to {path.parent}", 6000)

    # ================================================================== band power
    def _build_bp(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(430)
        scroll.setMinimumWidth(360)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        scroll.setWidget(left)
        h.addWidget(scroll)

        box = QtWidgets.QGroupBox("Bands")
        bv = QtWidgets.QVBoxLayout(box)
        self.band_table = QtWidgets.QTableWidget(0, 3)
        self.band_table.setHorizontalHeaderLabels(["band", "from Hz", "to Hz"])
        self.band_table.setMinimumHeight(170)
        bv.addWidget(self.band_table)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("+", lambda: self._add_band("band", 1, 2)),
                          ("−", lambda: self.band_table.removeRow(self.band_table.currentRow())),
                          ("Defaults", self._default_bands)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch()
        bv.addLayout(row)
        lv.addWidget(box)
        self._default_bands()

        box = QtWidgets.QGroupBox("Window and estimate")
        f = QtWidgets.QFormLayout(box)
        self.bp_t0 = _spin(-60, 120, 0.0, 0.1, 3, " s")
        self.bp_t1 = _spin(-60, 120, 0.8, 0.1, 3, " s")
        wr = QtWidgets.QHBoxLayout()
        wr.addWidget(self.bp_t0)
        wr.addWidget(QtWidgets.QLabel("to"))
        wr.addWidget(self.bp_t1)
        f.addRow("Window in epoch", wr)
        self.bp_method = QtWidgets.QComboBox()
        self.bp_method.addItems(["welch", "multitaper"])
        f.addRow("PSD (MNE)", self.bp_method)
        self.bp_roi = QtWidgets.QLineEdit()
        self.bp_roi.setPlaceholderText("ROI (empty = all good channels)")
        f.addRow("ROI", self.bp_roi)
        lv.addWidget(box)

        box = QtWidgets.QGroupBox("Baseline (after a baseline event)")
        f = QtWidgets.QFormLayout(box)
        self.bl_event = QtWidgets.QComboBox()
        self._bl_user = False
        self.bl_event.activated.connect(lambda *_: setattr(self, "_bl_user", True))
        self.bl_event.setToolTip("Event marking the baseline period, e.g. BASELINE_START or "
                                 "a resting/fixation-cross marker")
        f.addRow("Event", self.bl_event)
        self.bl_query = QtWidgets.QLineEdit()
        self.bl_query.setPlaceholderText('optional filter, e.g. phase == "BASELINE"')
        f.addRow("Filter", self.bl_query)
        self.bl_t0 = _spin(-600, 3600, 5.0, 1, 2, " s")
        self.bl_t1 = _spin(-600, 3600, 60.0, 1, 2, " s")
        br = QtWidgets.QHBoxLayout()
        br.addWidget(self.bl_t0)
        br.addWidget(QtWidgets.QLabel("to"))
        br.addWidget(self.bl_t1)
        f.addRow("After event", br)
        self.bl_mode = QtWidgets.QComboBox()
        self.bl_mode.addItem("mean of all baseline events", "mean")
        self.bl_mode.addItem("nearest preceding baseline event", "nearest")
        f.addRow("Use", self.bl_mode)
        self.bl_norm = QtWidgets.QComboBox()
        self.bl_norm.addItems(list(BP.NORMALISATIONS))
        self.bl_norm.setCurrentText("dB")
        f.addRow("Express as", self.bl_norm)
        f.addRow(hint("The baseline window is cut into segments as long as the analysis "
                      "window; their mean (and SD for z) is the reference. ‘absolute’ ignores "
                      "the baseline."))
        lv.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Compute band power ▶")
        b.clicked.connect(lambda: self.compute_bp())
        row.addWidget(b)
        lv.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        for label, fn in (("Topomaps (MNE)…", self.bp_topo), ("Export…", self.bp_export)):
            bb = QtWidgets.QPushButton(label)
            bb.clicked.connect(fn)
            row.addWidget(bb)
        lv.addLayout(row)
        lv.addStretch()

        right = QtWidgets.QVBoxLayout()
        self.bp_info = hint("")
        right.addWidget(self.bp_info)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.bp_plot = _themed_plot(bottom="band", left="power")
        self.bp_plot.addLegend()
        split.addWidget(self.bp_plot)
        tcw = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tcw)
        tl.setContentsMargins(0, 0, 0, 0)
        tr = QtWidgets.QHBoxLayout()
        tr.addWidget(QtWidgets.QLabel("Time course of"))
        self.tc_band = QtWidgets.QComboBox()
        tr.addWidget(self.tc_band)
        b = QtWidgets.QPushButton("Compute (Morlet) ▶")
        b.clicked.connect(lambda: self.compute_tc())
        tr.addWidget(b)
        tr.addWidget(hint("ERD/ERS over epoch time on the ROI, relative to the same "
                          "baseline"))
        tr.addStretch()
        tl.addLayout(tr)
        self.tc_plot = _themed_plot(bottom=("Time", "s"), left="power")
        self.tc_plot.addLegend()
        tl.addWidget(self.tc_plot, 1)
        split.addWidget(tcw)
        self.bp_table = QtWidgets.QTableWidget()
        split.addWidget(self.bp_table)
        split.setSizes([330, 260, 170])
        right.addWidget(split, 1)
        h.addLayout(right, 1)
        self.tabs.addTab(w, "Band power")

    def _add_band(self, name, lo, hi):
        r = self.band_table.rowCount()
        self.band_table.insertRow(r)
        for j, v in enumerate((name, f"{lo:g}", f"{hi:g}")):
            self.band_table.setItem(r, j, QtWidgets.QTableWidgetItem(v))

    def _default_bands(self):
        self.band_table.setRowCount(0)
        for n, (lo, hi) in BP.DEFAULT_BANDS.items():
            self._add_band(n, lo, hi)

    def bands(self) -> dict:
        out = {}
        for r in range(self.band_table.rowCount()):
            try:
                n = self.band_table.item(r, 0).text().strip()
                lo, hi = float(self.band_table.item(r, 1).text()), \
                    float(self.band_table.item(r, 2).text())
            except (AttributeError, ValueError):
                continue
            if n and hi > lo >= 0:
                out[n] = (lo, hi)
        if not out:
            raise ValueError("Define at least one band (name, from < to)")
        return out

    def _fill_baseline_events(self):
        cur = self.bl_event.currentText()
        t = self.pl.event_table
        self.bl_event.blockSignals(True)
        self.bl_event.clear()
        self.bl_event.addItem("— none (absolute power) —")
        if t is not None and len(t):
            m = t[t["kind"] == "marker"]["name"].astype(str).value_counts()
            names = sorted(m.index, key=lambda n: ("BASELINE" not in n.upper()
                                                   and "REST" not in n.upper(), n))
            for n in names:
                self.bl_event.addItem(f"{n}", n)
        i = self.bl_event.findText(cur) if self._bl_user else -1
        if i < 0:  # prefer an obvious baseline marker until the user picks one
            i = next((k for k in range(1, self.bl_event.count())
                      if self.bl_event.itemText(k).upper().endswith("START") and
                      ("BASELINE" in self.bl_event.itemText(k).upper() or
                       "REST" in self.bl_event.itemText(k).upper())), 0)
        self.bl_event.setCurrentIndex(i)
        self.bl_event.blockSignals(False)

    def baseline(self) -> BP.BandBaseline | None:
        name = self.bl_event.currentData()
        if not name:
            return None
        t = self.pl.event_table
        sel = t[t["name"].astype(str) == name]
        if self.bl_query.text().strip():
            from .. import events as EV
            sel = EV.filter_events(sel, query=self.bl_query.text())
        if not len(sel):
            raise ValueError(f"No “{name}” events match the baseline filter")
        if self.bl_t1.value() <= self.bl_t0.value():
            raise ValueError("The baseline end must be after its start")
        return BP.BandBaseline(np.sort(sel["onset"].to_numpy(float)), self.bl_t0.value(),
                               self.bl_t1.value(), self.bl_mode.currentData(), name)

    def compute_bp(self, then=None):
        groups = self._need_groups()
        if groups is None:
            return False
        first = next(iter(groups.values()))
        try:
            bands = self.bands()
            base = self.baseline()
            roi = self._roi(self.bp_roi, first.ch_names)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Band power", str(e))
            return False
        s = BP.BandSettings(bands, self.bp_t0.value(), self.bp_t1.value(),
                            self.bp_method.currentText())
        mode = self.bl_norm.currentText() if base is not None else "absolute"
        raw = self.pl.eeg_input()
        names = list(bands)

        def job():
            notes = []
            out = {"bp": {}, "ref": {}, "sd": {}, "norm": {}, "per": [], "bands": bands,
                   "mode": mode, "roi": roi, "base": base, "settings": s}
            ch = list(first.ch_names)
            seg = (min(s.tmax, first.tmax) - max(s.tmin, first.tmin))
            base_bp = owner = None
            if base is not None:
                base_bp, owner = BP.baseline_band_power(raw, base, s, ch, seg)
                notes.append(f"baseline: {len(base_bp)} segments of {seg:.2f} s from "
                             f"{len(np.unique(owner))} “{base.label}” event(s), "
                             f"{base.t0:g}…{base.t1:g} s after")
            ridx = BP.resolve_roi(roi, ch)
            for g, ep in groups.items():
                bp = BP.band_power(ep, s)
                out["bp"][g] = bp
                if base_bp is not None:
                    ref, sd = BP.reference_for(ep.metadata["onset"].to_numpy(float), base,
                                               base_bp, owner)
                else:
                    ref = sd = None
                out["ref"][g], out["sd"][g] = ref, sd
                if mode == "dB" and ref is not None:
                    # dB of ROI-mean power (averaging dB across channels is biased)
                    val = 10 * np.log10(BP.roi_mean(bp, ridx) / BP.roi_mean(ref, ridx))
                else:
                    val = BP.roi_mean(BP.normalise(bp, ref, sd, mode), ridx)
                out["norm"][g] = BP.normalise(bp, ref, sd, mode)
                df = pd.DataFrame(val, columns=names)
                df.insert(0, "group", g)
                df.insert(1, "onset", ep.metadata["onset"].to_numpy(float))
                out["per"].append(df)
            per = pd.concat(out["per"], ignore_index=True)
            out["per_epoch"] = per
            out["table"] = ST.comparison_table(per, names, list(groups))
            out["notes"] = notes
            return out

        def done(res):
            self.bp = res
            self.tc = None
            self.tc_band.clear()
            self.tc_band.addItems(list(res["bands"]))
            if "alpha" in res["bands"]:
                self.tc_band.setCurrentText("alpha")
            self._plot_bp()
            self._plot_tc()
            if then:
                then()

        return self.worker.run("Computing band power (MNE PSD)…", job, done)

    def _plot_bp(self):
        p = self.bp_plot
        p.clear()
        r = self.bp
        if r is None:
            self.bp_table.clear()
            return
        names = list(r["bands"])
        per = r["per_epoch"]
        gs = list(r["bp"])
        width = 0.8 / max(len(gs), 1)
        for k, g in enumerate(gs):
            sub = per[per["group"] == g]
            means = sub[names].mean().to_numpy(float)
            sems = _sem(sub[names].to_numpy(float))
            xs = np.arange(len(names)) - 0.4 + width * (k + 0.5)
            _bars(p.getPlotItem(), xs, means, sems, self._color(g, 200), width * 0.9)
            p.plot([], [], pen=pg.mkPen(self._color(g), width=6), name=f"{g} (n={len(sub)})")
        p.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(names)]])
        p.setLabel("left", f"band power ({BP.unit(r['mode'])})")
        if r["mode"] != "absolute":
            p.addItem(pg.InfiniteLine(0 if r["mode"] in ("dB", "% change", "z") else 1,
                                      angle=0, pen=pg.mkPen(THEME.muted)))
        fill_table(self.bp_table, r["table"], fmt="{:.4g}")
        roi = r["roi"]
        roi_txt = ", ".join(roi) if len(roi) <= 8 else f"{len(roi)} channels"
        self.bp_info.setText(f"ROI: {roi_txt}  ·  window {r['settings'].tmin:g}…"
                             f"{r['settings'].tmax:g} s  ·  {r['settings'].method}  ·  "
                             f"{r['mode']}" + ("\n" + "; ".join(r["notes"])
                                               if r["notes"] else ""))

    def compute_tc(self, then=None):
        r = self.bp
        if r is None:
            QtWidgets.QMessageBox.information(self, "Time course", "Compute band power first.")
            return False
        band = self.tc_band.currentText()
        rng = r["bands"][band]
        groups = {g: self.pl.epochs[g] for g in r["bp"] if g in self.pl.epochs}
        raw = self.pl.eeg_input()
        base, mode, roi = r["base"], r["mode"], r["roi"]

        def job():
            out = {"band": band, "mode": mode, "curves": {}}
            lvl = sd = None
            if base is not None and mode != "absolute":
                lvl, sd = BP.baseline_timecourse_level(raw, base, rng, roi)
            for g, ep in groups.items():
                t, pw = BP.band_timecourse(ep, rng, roi)
                if lvl is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        pw = {"dB": 10 * np.log10(pw / lvl), "% change": 100 * (pw - lvl) / lvl,
                              "ratio": pw / lvl, "z": (pw - lvl) / sd}[mode]
                out["curves"][g] = (t, np.nanmean(pw, 0), _sem(pw))
            return out

        def done(res):
            self.tc = res
            self._plot_tc()
            if then:
                then()

        return self.worker.run(f"Computing {band} time course…", job, done)

    def _plot_tc(self):
        p = self.tc_plot
        p.clear()
        r = self.tc
        if r is None:
            return
        for g, (t, m, s) in r["curves"].items():
            lo = p.plot(t, m - s, pen=None)
            hi = p.plot(t, m + s, pen=None)
            p.addItem(pg.FillBetweenItem(lo, hi, brush=self._color(g, 50)))
            p.plot(t, m, pen=pg.mkPen(self._color(g), width=2), name=g)
        p.addItem(pg.InfiniteLine(0, angle=90, pen=pg.mkPen(THEME.muted)))
        p.setLabel("left", f"{r['band']} power ({BP.unit(r['mode'])})")

    def bp_topo(self):
        r = self.bp
        if r is None:
            return
        g0 = next(iter(r["bp"]))
        ep = self.pl.epochs.get(g0)
        if ep is None or not has_positions(ep):
            QtWidgets.QMessageBox.information(self, "Topomaps", NO_POS)
            return
        import matplotlib.pyplot as plt
        names = list(r["bands"])
        cols = {g: np.nanmean(v, 0) for g, v in r["norm"].items()}  # (ch, bands)
        if r["mode"] == "dB" and all(r["ref"][g] is not None for g in cols):
            cols = {g: 10 * np.log10(np.nanmean(r["bp"][g], 0) / np.nanmean(r["ref"][g], 0))
                    for g in cols}
        if len(cols) == 2:
            a, b = cols
            cols[f"{a} − {b}"] = cols[a] - cols[b]
        fig, axes = plt.subplots(len(cols), len(names),
                                 figsize=(2.3 * len(names), 2.3 * len(cols)), squeeze=False)
        for i, (g, v) in enumerate(cols.items()):
            for j, n in enumerate(names):
                x = v[:, j]
                if r["mode"] == "absolute" and "−" not in g:
                    x = np.log10(x)
                ok = np.isfinite(x)
                lim = np.nanmax(np.abs(x[ok])) if ok.any() else 1
                sym = r["mode"] != "absolute" or "−" in g
                mne.viz.plot_topomap(np.nan_to_num(x, nan=np.nanmean(x[ok]) if ok.any() else 0),
                                     ep.info, axes=axes[i, j], show=False, contours=0,
                                     cmap="RdBu_r" if sym else "viridis",
                                     vlim=(-lim, lim) if sym else (None, None))
                axes[i, j].set_title(f"{g}\n{n}", fontsize=8)
        unit = "log10 µV²" if r["mode"] == "absolute" else BP.unit(r["mode"])
        fig.suptitle(f"Band power ({unit})")
        plt.show()

    def bp_export(self):
        r = self.bp
        if r is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export band power to folder")
        if not d:
            return
        out = Path(d)
        r["table"].to_csv(out / "bandpower_comparison.csv", index=False)
        r["per_epoch"].to_csv(out / "bandpower_per_epoch.csv", index=False)
        ch = list(next(iter(self.pl.epochs.values())).ch_names)
        rows = []
        for g, v in r["norm"].items():
            m = np.nanmean(v, 0)
            for i, c in enumerate(ch):
                rows.append({"group": g, "channel": c,
                             **{n: m[i, j] for j, n in enumerate(r["bands"])}})
        pd.DataFrame(rows).to_csv(out / "bandpower_per_channel.csv", index=False)
        b = r["base"]
        self._write_provenance(out / "bandpower.txt", {
            "bands": r["bands"], "ROI": ", ".join(r["roi"]), "unit": r["mode"],
            "window": (r["settings"].tmin, r["settings"].tmax),
            "baseline": f"{b.label} +{b.t0:g}…{b.t1:g} s ({b.mode})" if b else "none"})

    # ================================================================== indices
    def _build_idx(self):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        left.setMaximumWidth(430)
        left.setMinimumWidth(340)
        h.addWidget(left)
        box = QtWidgets.QGroupBox("Regions used by the indices")
        bv = QtWidgets.QVBoxLayout(box)
        self.roi_table = QtWidgets.QTableWidget(0, 2)
        self.roi_table.setHorizontalHeaderLabels(["region", "channels"])
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.verticalHeader().setVisible(False)
        for k, v in BP.DEFAULT_ROIS.items():
            r = self.roi_table.rowCount()
            self.roi_table.insertRow(r)
            it = QtWidgets.QTableWidgetItem(k)
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            self.roi_table.setItem(r, 0, it)
            self.roi_table.setItem(r, 1, QtWidgets.QTableWidgetItem(", ".join(v)))
        self.roi_table.setMaximumHeight(180)
        bv.addWidget(self.roi_table)
        bv.addWidget(hint("Edit the channel lists to match your montage. Bad channels are "
                          "not in the epochs: interpolate them in ① to keep these regions."))
        lv.addWidget(box)
        box = QtWidgets.QGroupBox("Indices")
        bv = QtWidgets.QVBoxLayout(box)
        self.idx_list = QtWidgets.QListWidget()
        for k, desc in list(BP.INDEX_INFO.items()) + [
                ("pupil dilation", "task-evoked pupil response (mean in window − 200 ms "
                                   "pre-event); larger = more load (Beatty, Kahneman)"),
                ("blink rate /min", "blinks per minute inside the window (eye tracker)"),
                ("mean fixation ms", "mean duration of fixations starting in the window")]:
            it = QtWidgets.QListWidgetItem(k)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked)
            it.setToolTip(desc)
            self.idx_list.addItem(it)
        bv.addWidget(self.idx_list)
        bv.addWidget(hint("Computed per epoch from the band-power window and bands (Band "
                          "power tab). ERD/ERS need a baseline event. Hover for references."))
        lv.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Compute indices ▶")
        b.clicked.connect(lambda: self.compute_idx())
        row.addWidget(b)
        bb = QtWidgets.QPushButton("Export…")
        bb.clicked.connect(self.idx_export)
        row.addWidget(bb)
        lv.addLayout(row)
        lv.addStretch()
        right = QtWidgets.QVBoxLayout()
        self.idx_info = hint("")
        right.addWidget(self.idx_info)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.idx_plots = pg.GraphicsLayoutWidget()
        self.idx_plots.setBackground(THEME.bg)
        split.addWidget(self.idx_plots)
        self.idx_table = QtWidgets.QTableWidget()
        split.addWidget(self.idx_table)
        split.setSizes([480, 220])
        right.addWidget(split, 1)
        h.addLayout(right, 1)
        self.tabs.addTab(w, "Cognitive indices")

    def rois(self) -> dict:
        out = {}
        for r in range(self.roi_table.rowCount()):
            k = self.roi_table.item(r, 0).text()
            v = self.roi_table.item(r, 1).text() if self.roi_table.item(r, 1) else ""
            out[k] = [c for c in v.replace(",", " ").split() if c]
        return out

    def compute_idx(self, then=None):
        if self.bp is None or set(self.bp["bp"]) != set(self.groups()):
            return self.compute_bp(then=lambda: self.compute_idx(then))
        r = self.bp
        rois = self.rois()
        chosen = [self.idx_list.item(i).text() for i in range(self.idx_list.count())
                  if self.idx_list.item(i).checkState() == QtCore.Qt.Checked]
        rec = self.pl.rec
        gev = self.pl.gaze_events
        s = r["settings"]

        def job():
            per, notes = [], []
            for g, bp in r["bp"].items():
                ep = self.pl.epochs[g]
                df, n = BP.eeg_indices(bp, list(ep.ch_names), r["bands"], rois, r["ref"][g])
                notes += [x for x in n if x not in notes]
                on = ep.metadata["onset"].to_numpy(float)
                gm = BP.gaze_metrics(on, max(s.tmin, ep.tmin), min(s.tmax, ep.tmax),
                                     rec.gaze if rec is not None else None, gev)
                df = pd.concat([df, gm], axis=1)
                df.insert(0, "group", g)
                df.insert(1, "onset", on)
                per.append(df)
            per = pd.concat(per, ignore_index=True)
            cols = [c for c in chosen if c in per.columns and per[c].notna().any()]
            missing = [c for c in chosen if c not in cols]
            if missing:
                notes.append("not available: " + ", ".join(missing))
            table = ST.comparison_table(per, cols, list(r["bp"]))
            return dict(per_epoch=per, table=table, measures=cols, notes=notes)

        def done(res):
            self.idx = res
            self._plot_idx()
            if then:
                then()

        return self.worker.run("Computing indices…", job, done)

    def _plot_idx(self):
        self.idx_plots.clear()
        r = self.idx
        if r is None:
            self.idx_table.clear()
            return
        per = r["per_epoch"]
        gs = list(dict.fromkeys(per["group"]))
        ncol = 3
        rng = np.random.default_rng(0)
        q = dict(zip(r["table"]["measure"], r["table"]["q (FDR)"])) if len(r["table"]) else {}
        for k, m in enumerate(r["measures"]):
            p = self.idx_plots.addPlot(row=k // ncol, col=k % ncol)
            qq = q.get(m, np.nan)
            star = "" if not np.isfinite(qq) else ("  ***" if qq < 0.001 else "  **" if qq < 0.01
                                                    else "  *" if qq < 0.05 else "  n.s.")
            p.setTitle(f"{m}{star}", size="9pt", color=THEME.fg)
            p.showGrid(y=True, alpha=0.25)
            for i, g in enumerate(gs):
                v = per.loc[per["group"] == g, m].to_numpy(float)
                v = v[np.isfinite(v)]
                if not len(v):
                    continue
                lo, hi = np.percentile(v, [2, 98]) if len(v) > 10 else (v.min(), v.max())
                vv = v[(v >= lo) & (v <= hi)]
                p.plot(i + rng.uniform(-0.25, 0.25, len(vv)), vv, pen=None, symbol="o",
                       symbolSize=3, symbolPen=None, symbolBrush=self._color(g, 60))
                _bars(p, [i], np.array([v.mean()]), np.array([_sem(v)]), self._color(g, 150),
                      0.6)
            p.getAxis("bottom").setTicks([[(i, g) for i, g in enumerate(gs)]])
        fill_table(self.idx_table, r["table"], fmt="{:.4g}")
        self.idx_info.setText("Bars: mean ± SEM per group, dots: epochs (2–98th percentile). "
                              "Stars: FDR-corrected q across indices."
                              + ("\n" + "; ".join(r["notes"]) if r["notes"] else ""))

    def idx_export(self):
        r = self.idx
        if r is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Export indices to folder")
        if not d:
            return
        out = Path(d)
        r["table"].to_csv(out / "indices_comparison.csv", index=False)
        r["per_epoch"].to_csv(out / "indices_per_epoch.csv", index=False)
        self._write_provenance(out / "indices.txt", {"regions": self.rois(),
                                                     "bands": self.bp["bands"]})
