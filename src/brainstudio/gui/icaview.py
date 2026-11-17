"""Embedded view of one ICA component: scalp map, spectrum, blink-locked average
and a stretch of its time course (with eye-tracker blinks marked)."""

from __future__ import annotations

import mne
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtWidgets

from .theme import THEME
from .widgets import has_positions


class ICAComponentView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self._args = None
        self._message("Fit ICA, then click a component in the table.\n"
                      "Double-click opens MNE's full properties window.")

    def _message(self, text):
        self.fig.clear()
        self.fig.set_facecolor(THEME.bg)
        ax = self.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", color=THEME.muted,
                transform=ax.transAxes)
        self.canvas.draw_idle()

    def restyle(self):
        if self._args is not None:
            self.show_component(*self._args)
        else:
            self._message("Fit ICA, then click a component in the table.")

    def show_component(self, ica: mne.preprocessing.ICA, fit_raw: mne.io.BaseRaw,
                       sources: np.ndarray, k: int, blinks=None, scores=None,
                       excluded: bool = False):
        self._args = (ica, fit_raw, sources, k, blinks, scores, excluded)
        fig = self.fig
        fig.clear()
        fig.set_facecolor(THEME.bg)
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], width_ratios=[1.3, 1, 1])
        ax_topo = fig.add_subplot(gs[0, 0])
        ax_psd = fig.add_subplot(gs[0, 1])
        ax_blink = fig.add_subplot(gs[0, 2])
        ax_ts = fig.add_subplot(gs[1, :])
        for ax in (ax_topo, ax_psd, ax_blink, ax_ts):
            ax.set_facecolor(THEME.bg)
        sf = fit_raw.info["sfreq"]
        src = sources[k]
        col = THEME.event if excluded else THEME.s(0)

        # scalp map of the mixing pattern
        if has_positions(fit_raw):
            info = mne.pick_info(fit_raw.info, mne.pick_channels(fit_raw.ch_names,
                                                                 ica.ch_names, ordered=True))
            pattern = ica.get_components()[:, k]
            mne.viz.plot_topomap(pattern, info, axes=ax_topo, show=False, contours=4,
                                 cmap="RdBu_r", sensors=True)
            from matplotlib.colors import to_hex
            for ln in ax_topo.lines:
                if to_hex(ln.get_color()) == "#000000":
                    ln.set_color(THEME.fg)
        else:
            ax_topo.set_axis_off()
            ax_topo.text(0.5, 0.5, "no electrode\npositions", ha="center", va="center",
                         color=THEME.muted, transform=ax_topo.transAxes)
        title = f"ICA{k:03d}" + ("  — excluded" if excluded else "")
        fig.suptitle(title, color=col, fontsize=11, fontweight="bold")
        ax_topo.set_title("scalp map", fontsize=9)

        # spectrum
        psd, freqs = mne.time_frequency.psd_array_welch(
            src, sf, fmin=0.5, fmax=min(60.0, sf / 2), n_fft=int(min(len(src), 4 * sf)),
            verbose="error")
        ax_psd.plot(freqs, 10 * np.log10(np.maximum(psd, 1e-30)), color=col, lw=1.4)
        ax_psd.set_xlabel("Hz", fontsize=8)
        ax_psd.set_title("spectrum (dB)", fontsize=9)
        ax_psd.tick_params(labelsize=7)
        ax_psd.grid(alpha=0.3)

        # blink-locked average (eye tracker)
        t = fit_raw.times
        if blinks is not None and len(blinks):
            on = np.searchsorted(t, blinks["onset"].to_numpy(float))
            i0, i1 = int(-0.3 * sf), int(0.6 * sf)
            on = on[(on + i0 >= 0) & (on + i1 < len(src))][:3000]
            if len(on) >= 3:
                z = (src - src.mean()) / (src.std() + 1e-20)
                seg = z[on[:, None] + np.arange(i0, i1)[None, :]]
                seg -= seg[:, : -i0].mean(1, keepdims=True)
                m, se = seg.mean(0), seg.std(0) / np.sqrt(len(on))
                tt = np.arange(i0, i1) / sf
                ax_blink.fill_between(tt, m - se, m + se, color=col, alpha=0.25, lw=0)
                ax_blink.plot(tt, m, color=col, lw=1.4)
                ax_blink.axvline(0, color=THEME.muted, lw=0.8)
                ax_blink.set_title(f"blink-locked (n={len(on)})", fontsize=9)
            else:
                ax_blink.text(0.5, 0.5, "too few blinks", ha="center", va="center",
                              color=THEME.muted, transform=ax_blink.transAxes)
        else:
            ax_blink.text(0.5, 0.5, "no eye-tracker\nblinks", ha="center", va="center",
                          color=THEME.muted, transform=ax_blink.transAxes)
        ax_blink.set_xlabel("s from blink", fontsize=8)
        ax_blink.tick_params(labelsize=7)
        ax_blink.grid(alpha=0.3)

        # time course: 20 s around the strongest blink-related stretch (or the start)
        win = int(20 * sf)
        start = 0
        if blinks is not None and len(blinks):
            mid = np.searchsorted(t, float(np.median(blinks["onset"])))
            start = int(np.clip(mid - win // 2, 0, max(len(src) - win, 0)))
        sl = slice(start, start + win)
        ax_ts.plot(t[sl], src[sl], color=col, lw=0.7)
        if blinks is not None and len(blinks):
            b = blinks[(blinks["onset"] >= t[sl][0]) & (blinks["onset"] <= t[sl][-1])]
            for on, d in zip(b["onset"], b["duration"]):
                ax_ts.axvspan(on, on + d, color=THEME.accent, alpha=0.25, lw=0)
        ax_ts.set_xlabel("time (s)  —  shaded: eye-tracker blinks", fontsize=8)
        ax_ts.set_yticks([])
        ax_ts.tick_params(labelsize=7)
        if scores is not None:
            r = scores.iloc[k]
            parts = []
            if np.isfinite(r.get("eog_score", np.nan)):
                parts.append(f"EOG/frontal |r| = {r['eog_score']:.2f}")
            if np.isfinite(r.get("blink_t", np.nan)):
                parts.append(f"blink-locked t = {r['blink_t']:.1f}")
            if r.get("suggested", False):
                parts.append("suggested as blink component")
            ax_ts.set_title("   ·   ".join(parts), fontsize=9)
        self.canvas.draw_idle()
