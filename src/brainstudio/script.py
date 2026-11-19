"""Stand-alone Python scripts that repeat an analysis of the app, from the files on.

A script has four sections (``# %%`` cells): Loading, Preprocessing, Epoching and
Analysis. Each section explains what it does with the settings used, then holds the
code it needs, then the steps. It is written from the settings each stage actually
ran with (the *recipes* the app keeps with its results).

Scripts do not need brAInStudio: standard steps are plain MNE / AutoReject / scipy
code, and the brAInStudio functions a script uses (reading XDF, gaze events, the event
table, the app's measures) are copied into it unchanged by :mod:`brainstudio.bundle`.
So a script computes exactly what the app computes.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt

import numpy as np

#: no module prefixes: the classes are copied into the script
_PREFIX: dict = {}

TITLES = {"erp": "ERP", "bandpower": "Band power", "indices": "Cognitive indices"}


WIDTH = 92


def _flat(x) -> str:
    """Python source for a value on one line (numbers exact, dataclasses as calls)."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        name = type(x).__name__
        args = ", ".join(f"{f.name}={_flat(getattr(x, f.name))}"
                         for f in dataclasses.fields(x))
        return f"{_PREFIX.get(name, '')}{'.' if name in _PREFIX else ''}{name}({args})"
    if isinstance(x, (bool, np.bool_)):
        return repr(bool(x))
    if isinstance(x, (int, np.integer)):
        return repr(int(x))
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return repr(v) if np.isfinite(v) else f"float('{v}')"
    if isinstance(x, str) or x is None:
        return repr(x)
    if isinstance(x, tuple):
        return "(" + ", ".join(_flat(v) for v in x) + ("," if len(x) == 1 else "") + ")"
    if isinstance(x, (list, np.ndarray)):
        return "[" + ", ".join(_flat(v) for v in list(x)) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{_flat(k)}: {_flat(v)}" for k, v in x.items()) + "}"
    return repr(x)


def lit(x, col: int = 0, base: int = 0) -> str:
    """Like :func:`_flat`, wrapped over several lines (black style) when it would pass
    ``WIDTH`` starting at column ``col`` on a line indented by ``base``."""
    flat = _flat(x)
    if col + len(flat) <= WIDTH:
        return flat
    pad = " " * (base + 4)
    end = " " * base
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        name = flat[:flat.index("(")]
        fields = [f"{pad}{f.name}="
                  f"{lit(getattr(x, f.name), len(pad) + len(f.name) + 1, len(pad))},"
                  for f in dataclasses.fields(x)]
        return name + "(\n" + "\n".join(fields) + "\n" + end + ")"
    if isinstance(x, dict):
        items = [f"{pad}{_flat(k)}: {lit(v, len(pad) + len(_flat(k)) + 2, len(pad))},"
                 for k, v in x.items()]
        return "{\n" + "\n".join(items) + "\n" + end + "}"
    if isinstance(x, (list, tuple, np.ndarray)):
        o, c = ("(", ")") if isinstance(x, tuple) else ("[", "]")
        parts = [lit(v, len(pad), len(pad)) for v in list(x)]
        if all("\n" not in p for p in parts):  # fill lines with short items
            lines, cur = [], pad
            for p in parts:
                if len(cur) + len(p) + 2 > WIDTH and cur.strip():
                    lines.append(cur.rstrip())
                    cur = pad
                cur += p + ", "
            lines.append(cur.rstrip())
            return o + "\n" + "\n".join(lines) + "\n" + end + c
        return o + "\n" + "\n".join(f"{pad}{p}," for p in parts) + "\n" + end + c
    return flat


def assign(prefix: str, x, suffix: str = "") -> str:
    """``prefix`` followed by the value, wrapped to fit."""
    last = prefix.split("\n")[-1]
    return prefix + lit(x, len(last), len(last) - len(last.lstrip())) + suffix


def check_recipe(recipe: dict | None) -> str | None:
    """Why a script cannot reproduce these results (None when it can)."""
    if recipe is None:
        return "These results come from epochs built before scripts could be exported: " \
               "build the epochs again."
    if not recipe["load"] or not recipe["load"]["steps"]:
        return "The recording was not loaded through a file, file list or BIDS run."
    pre = recipe.get("pre")
    if pre is not None:
        eeg_steps = [s for s in recipe["load"]["steps"] if s["kind"] != "gaze_offset"]
        pre_steps = [s for s in pre["load"]["steps"] if s["kind"] != "gaze_offset"]
        if eeg_steps != pre_steps:
            return "The recording changed (montage or trigger channel) after the " \
                   "preprocessing these epochs use: run preprocessing and epoching again."
        if pre["params"].ica_exclude and pre["ica"] is None:
            return "The ICA these results use is no longer known: fit ICA again."
    return None


# =========================================================================== helpers
BAR = "# " + "─" * 88

#: pip name of each library the copied code imports
PIP = {"mne": "mne", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
       "pyxdf": "pyxdf", "mne_bids": "mne-bids", "autoreject": "autoreject",
       "pymovements": "pymovements", "polars": "polars", "picard": "python-picard"}


def _comment(text: str, width: int = WIDTH) -> list[str]:
    """``text`` as wrapped ``#`` comment lines (paragraphs: blank lines; bullets: •)."""
    import textwrap
    out = []
    for para in text.strip().split("\n"):
        para = para.rstrip()
        if not para:
            out.append("#")
            continue
        ind = len(para) - len(para.lstrip())
        sub = "  " if para.lstrip().startswith("•") else ""
        out += textwrap.wrap(para.strip(), width - 2 - ind, initial_indent="# " + " " * ind,
                             subsequent_indent="# " + " " * ind + sub)
    return out


def _section(number: int, title: str, explanation: str) -> list[str]:
    return [f"# %% {number} {title}", BAR, f"# {number}  {title.upper()}", BAR,
            *_comment(explanation), BAR, ""]


class _Code:
    """Copies the brAInStudio code each section needs, once."""

    def __init__(self, sections: list[list[str]]):
        from .bundle import Bundle
        self.b = Bundle()
        allkeys = self.b.closure([r for sec in sections for r in sec], set())
        self.final = self.b.names(allkeys)
        self.externals = self.b.externals(allkeys)
        self.done: set = set()

    def name(self, root: str) -> str:
        mod, _, n = root.rpartition(".")
        return self.final[self.b.resolve(mod, n)]

    def helpers(self, roots: list[str], version: str) -> list[str]:
        keys = self.b.closure(roots, self.done)
        self.done.update(keys)
        if not keys:
            return []
        names = ", ".join(self.final[k] for k in keys if not self.final[k].startswith("_"))
        n_lines = sum(len(self.b.source(k, self.final).splitlines()) for k in keys)
        out = [f"# region brAInStudio {version} code used by this stage ({n_lines} lines; "
               "fold it in your editor)",
               *_comment("Copied unchanged from brAInStudio (only the module prefixes "
                         "are dropped), so this script needs no brAInStudio and computes "
                         "exactly what the app computes. Defined here: " + names + "."),
               ""]
        for k in keys:
            out += [self.b.source(k, self.final), "", ""]
        out += ["# endregion", "", "# ── steps " + "─" * 80]
        return out


def _requirements(externals: list[str]) -> str:
    mods = {"mne", "numpy", "pandas", "scipy"}
    for line in externals:
        mods.add(line.split()[1].split(".")[0])
    return " ".join(sorted({PIP.get(m, m) for m in mods if m not in
                            ("re", "json", "dataclasses", "pathlib", "math", "warnings",
                             "itertools", "collections", "typing", "copy", "os", "sys",
                             "datetime", "functools", "io", "textwrap")}))


# =========================================================================== explanations
def _describe_loading(recipe: dict) -> str:
    parts = ["Reads the recording and builds the event table that the epochs are later "
             "selected from.", ""]
    for st in recipe["load"]["steps"]:
        k = st["kind"]
        if k == "load":
            name = st["path"].rsplit("/", 1)[-1]
            parts.append(f"• {name}: " + (
                "an XDF file read with pyxdf. Each stream gets a role (EEG, gaze, markers "
                "or ignored); the streams are put on the EEG clock using the LSL clock "
                "offsets, samples stamped on another clock are removed, and timestamp "
                "jitter is smoothed while following any drift; the EEG becomes an MNE Raw "
                "on a regular time grid, the gaze is converted to screen pixels, and "
                "marker streams become events (start/stop rows paired into spans)."
                if name.lower().endswith((".xdf", ".xdfz")) else
                "read with MNE (mne.io.read_raw); channel types are set from the "
                "channel names."))
        elif k == "load_many":
            parts.append(f"• {len(st['paths'])} files read one by one (as above) and "
                         "joined end to end into one recording; each part keeps its "
                         "subject/session/run/file, and a boundary annotation marks each "
                         "join so that no epoch or filter crosses it.")
        elif k == "load_bids":
            subs = sorted({r[0] for r in st["runs"] if r[0]})
            parts.append(f"• BIDS dataset {st['root']}: {len(st['runs'])} run(s)"
                         + (f" of subject(s) {', '.join(subs)}" if subs else "")
                         + ", read with MNE-BIDS (EEG, events.tsv with all its columns, "
                         "channels.tsv bad channels, electrode positions) plus the eye-"
                         "tracking files, then joined in this order with a boundary "
                         "annotation at each join; each part keeps its subject, session, "
                         "task and run.")
        elif k == "attach_gaze":
            parts.append(f"• Eye-tracking data replaced with {st['path'].rsplit('/', 1)[-1]}.")
        elif k == "gaze_offset":
            parts.append(f"• Gaze shifted by {st['seconds'] * 1000:+g} ms relative to the "
                         "EEG (a manual synchronisation correction).")
        elif k == "montage":
            parts.append("• Channels renamed in recording order, typed by name (EOG, ECG, "
                         "…) and given positions"
                         + (f" from the {st['source']} montage." if st["source"] else
                            " (no montage)."))
        elif k == "trigger_events":
            codes = ", ".join(f"{c} → {n}" for c, n in st["names"].items())
            parts.append(f"• Trigger channel {st['channel']} decoded "
                         f"({'integer codes' if st['mode'] == 'codes' else 'pulses'}"
                         + (f", mask {st['mask']}" if st["mask"] else "")
                         + f") into events: {codes}"
                         + ("; the channel is then treated as a stim channel."
                            if st["as_stim"] else "."))
        elif k == "remove_trigger_events":
            parts.append("• Trigger events removed again.")
    gz = recipe["load"]["gaze"] or {}
    p = gz.get("params")
    if p is not None:
        parts.append(f"• Gaze events detected with pymovements on the screen-pixel gaze: "
                     f"fixations with I-DT (dispersion ≤ {p.fix_dispersion:g} px for at least "
                     f"{p.fix_min_dur * 1000:g} ms); saccades with the Engbert & Kliegl "
                     f"velocity threshold (λ = {p.sac_threshold_factor:g}, at least "
                     f"{p.sac_min_dur * 1000:g} ms; velocity from Gaze.pos2vel, "
                     f"'{p.velocity_method}'); blinks from the pupil signal (lost or abruptly "
                     f"changing, {p.blink_min_dur * 1000:g}–{p.blink_max_dur * 1000:g} ms).")
    if gz.get("aois"):
        parts.append(f"• Areas of interest ({', '.join(gz['aois'])}, rectangles): each "
                     "fixation gets the name of the AOI it falls in (column 'aoi', "
                     "pymovements Events.map_to_aois).")
    parts.append("• The event table: one row per marker and gaze event, with its onset "
                 "(s), duration, name, every column the marker carries, and the subject/"
                 "session/run/file of the part it falls in.")
    return "\n".join(parts)


def _describe_preprocessing(recipe: dict) -> str:
    pre = recipe.get("pre")
    if pre is None:
        return ("No preprocessing: the epochs are cut from the recording as loaded.")
    p, ica = pre["params"], pre["ica"]
    hp = (f"high-pass {p.l_freq:g} Hz" + (" (zero-phase 4th-order Butterworth, applied "
          "separately between boundaries)" if p.l_freq and p.l_freq < 0.5 else
          " (MNE's zero-phase FIR)")) if p.l_freq else "no high-pass"
    lp = f"low-pass {p.h_freq:g} Hz (MNE's zero-phase FIR)" if p.h_freq else "no low-pass"
    out = [f"Cleans the continuous EEG before epoching: {hp}; {lp}; "
           + (f"notch at {p.notch:g} Hz and its harmonics" if p.notch else "no notch") + "; "
           + (f"bad channels {', '.join(p.bads)} "
              + ("interpolated (spherical splines)" if p.interpolate_bads else
                 "left out of the analyses") if p.bads else "no bad channels") + "."]
    if p.ica_exclude and ica is not None:
        s = ica["steps"]
        out += ["",
                "Eye blinks are removed with ICA, as in the ERP guide "
                "(neuraldatascience.io/eeg/erp-artifacts):",
                f"• The ICA is fitted on a separate copy of {len(s['chans'])} good EEG "
                f"channels, band-passed {s['l_freq']:g}–{s['h_freq']:g} Hz (ICA works "
                "best without slow drifts), average-referenced"
                + (f", from {s['tmin']:g} s (the first event: set-up and calibration "
                   "before it are left out)" if s["tmin"] else "")
                + (f", resampled to {s['resample']:g} Hz" if s["resample"] else "") + ".",
                "• It is cut into 1 s segments; "
                + ({"autoreject": "AutoReject finds the bad ones (fitted on at most "
                                  f"{s['autoreject_max']} segments), which are left out "
                                  "of the fit.",
                    "threshold": f"segments above {s['reject_uv']:g} µV peak-to-peak are "
                                 "left out of the fit." if s["reject_uv"] else
                                 "all of them are used.",
                    "all": "all of them are used."}[s["segments"]]),
                f"• ICA ({s['method']}"
                + (", extended" if s["fit_params"] and s["fit_params"].get("extended")
                   else "") + "): "
                + (f"enough components for {s['n_components']:.0%} of the variance"
                   if s["n_components"] < 1 else f"{s['n_components']} components")
                + f", every {s['decim']}rd sample, random seed {s['seed']}.",
                f"• Components {', '.join(map(str, p.ica_exclude))} (chosen in the app) "
                "are removed from the band-passed data."]
    return "\n".join(out)


def _describe_epoching(recipe: dict) -> str:
    st = recipe["settings"]
    out = ["Cuts the data into epochs around the events of each group. A group is a set "
           "of events from the event table: the event names ticked, the filters and the "
           "recordings (subjects, sessions, runs, files) kept:"]
    for g in recipe["groups"]:
        bits = [", ".join(g.names) or "no names"]
        if g.conditions:
            bits.append(" and ".join(" ".join(map(str, c)) for c in g.conditions))
        if g.text:
            bits.append(f"text ~ /{g.text}/")
        if g.query:
            bits.append(g.query)
        if g.recordings:
            bits.append("; ".join(f"{k}: {', '.join(map(str, v))}"
                                  for k, v in g.recordings.items() if k != "file"))
        out.append(f"• {g.label}: " + " | ".join(bits))
    out += ["",
            f"Each epoch runs from {st.tmin:g} to {st.tmax:g} s around its event, on the "
            "good EEG channels, without baseline correction (each analysis applies its "
            "own). Events whose window does not fit in the recording, or that share a "
            "sample with another event of the group, are skipped"
            + (f"; groups above {st.max_epochs} events are randomly subsampled (seed 0)"
               if st.max_epochs else "")
            + (f"; events are moved by {st.delay_ms:+g} ms (EEG delay)" if st.delay_ms
               else "")
            + (f"; epochs above {st.reject_uv:g} µV peak-to-peak are dropped"
               if st.reject_uv else "")
            + (f"; epochs flatter than {st.flat_uv:g} µV are dropped" if st.flat_uv else "")
            + ("; epochs containing an eye-tracker blink are dropped" if st.drop_blinks
               else "")
            + "; epochs overlapping a BAD_ span or a file boundary are dropped."
            + (" Then AutoReject (n_interpolate 1, 2, 4; seed 42), fitted once on the "
               "epochs of every group, interpolates bad channels within an epoch and "
               "drops the epochs it cannot repair." if st.autoreject else ""),
            "",
            "'summary' counts, per group, the events selected and why each one did or "
            "did not become an epoch."]
    return "\n".join(out)


def _describe_erp(spec, label: str, n_subjects: int) -> str:
    c = spec.component
    bl = f"{spec.baseline[0]:g} … {spec.baseline[1]:g} s" if spec.baseline else "none"
    return "\n".join([
        f"ERP component {c.name}: the {'positive' if c.polarity > 0 else 'negative'} "
        f"deflection {c.tmin * 1000:.0f}–{c.tmax * 1000:.0f} ms after the event, on the "
        f"mean of {', '.join(c.channels)}. Groups: {', '.join(spec.groups)}.",
        f"• Reference: {label}. Baseline: {bl} (subtracted from every epoch).",
        (f"• {n_subjects} subjects, as in neuraldatascience.io/eeg/erp-group-viz: each "
         "subject's epochs are averaged on their own (one Evoked per subject and group), "
         "then those Evokeds are averaged with mne.grand_average, so every subject counts "
         "once whatever its number of epochs. Each subject's mean amplitude in the window "
         "is compared between the groups with a paired t-test across subjects (the "
         "subjects are the observations); 'summary_table' adds 95 % CIs, the grand "
         "average's peaks, Wilcoxon and Cohen's d_z, and 'per_subject' has every "
         "subject's values."
         if n_subjects >= 2 else
         "• One subject: trials are the observations; the epochs' mean amplitudes in the "
         "window are compared with a Welch t-test; 'summary_table' adds the peaks, "
         "Mann-Whitney and Hedges' g. (With several subjects, the code below averages "
         "each subject first, then the subjects.)"),
        "",
        "Valid if the window was chosen before looking at the data."])


def _describe_bp(spec, tc_band=None) -> str:
    b = spec.baseline
    bands = ", ".join(f"{n} {lo:g}–{hi:g} Hz" for n, (lo, hi) in spec.bands.items())
    return "\n".join([
        f"Band power of {', '.join(spec.groups)} in {spec.tmin:g} … {spec.tmax:g} s of each "
        f"epoch: the power spectrum of every epoch (Epochs.compute_psd, {spec.method}) "
        f"averaged in each band ({bands}), then over the ROI "
        f"({', '.join(spec.roi) if len(spec.roi) <= 10 else f'{len(spec.roi)} channels'}).",
        f"• Reference: {spec.reference}.",
        (f"• Baseline: {b.t0:g} … {b.t1:g} s after each '{b.event}' event"
         + (f" matching {b.query}" if b.query else "")
         + f", cut into segments as long as the analysis window; every epoch is expressed "
         f"in {spec.norm} relative to its own subject's baseline (never one pooled across "
         "subjects)." if b else "• No baseline: absolute power (V²/Hz)."),
        "• The groups are compared per band (Welch t-test and Mann-Whitney for two groups, "
        "ANOVA and Kruskal-Wallis for more; Hedges' g; FDR across bands); trials are the "
        "observations.",
        (f"• Time course: power of the {tc_band} band over the epoch (Morlet wavelets) on "
         "the same ROI, reference and baseline (ERD/ERS)." if tc_band else "")])


def _describe_idx(bp_spec, spec) -> str:
    rois = "; ".join(f"{k}: {', '.join(v)}" for k, v in spec.rois.items())
    return "\n".join([
        _describe_bp(bp_spec),
        "",
        f"Cognitive indices per epoch from that band power ({', '.join(spec.measures)}), "
        f"with these channel regions: {rois}. Eye metrics (pupil dilation, blink rate, "
        "fixation duration) come from the eye tracker in the same window. The groups are "
        "compared per index as for band power."])


# =========================================================================== sections
def _loading_roots(recipe: dict) -> list[str]:
    roots = []
    for st in recipe["load"]["steps"]:
        roots += {"load": ["io.load"], "load_many": ["io.load_many"],
                  "load_bids": ["workflow.find_bids_runs", "io.load_bids_runs"],
                  "attach_gaze": ["io.load_eyetracking_file"],
                  "montage": ["eeg.apply_montage"],
                  "trigger_events": ["workflow.add_trigger_events"],
                  "remove_trigger_events": ["workflow.replace_trigger_events"],
                  }.get(st["kind"], [])
    gz = recipe["load"]["gaze"] or {}
    if gz.get("params") is not None:
        roots += ["gaze.DetectionParams", "gaze.detect_events"]
    if gz.get("aois"):
        roots += ["gaze.make_aois"]
    return roots + ["events.build_event_table"]


def _montage_code(source) -> list[str]:
    """The montage, built with the same standard MNE calls as the app (so the electrode
    positions are bit-identical)."""
    from . import eeg as E
    if not source:
        return ["montage = None"]
    layouts = E.layout_montages()
    if source in layouts:
        names, template = E.read_channel_layout(layouts[source])
        if template == "standard_1005" and "colin27_1005" in __import__(
                "mne").channels.get_builtin_montages():
            template = "colin27_1005"
        return [f"# the {source} layout: these electrodes (amplifier input order) at the "
                f"{template} template's positions",
                assign("layout = ", names),
                f"template = mne.channels.make_standard_montage({lit(template)})"
                ".get_positions()",
                'lookup = {k.upper(): v for k, v in template["ch_pos"].items()}',
                "montage = mne.channels.make_dig_montage(",
                "    ch_pos={n: lookup[n.upper()] for n in layout}, nasion=template['nasion'],",
                "    lpa=template['lpa'], rpa=template['rpa'], "
                "coord_frame=template['coord_frame'])"]
    from pathlib import Path
    p = Path(source)
    if not p.exists():
        return [f"montage = mne.channels.make_standard_montage({lit(source)})"]
    if p.suffix.lower() == ".fif":
        return [f"montage = mne.channels.read_dig_fif({lit(str(p))})"]
    if p.name.endswith("_electrodes.tsv"):
        return [f"# BIDS electrodes file (positions in m, mm or cm)",
                f"el = pd.read_csv({lit(str(p))}, sep='\\t', na_values=['n/a'])"
                ".dropna(subset=['x', 'y', 'z'])",
                "pos = el[['x', 'y', 'z']].to_numpy(float)",
                "if np.nanmax(np.abs(pos)) > 1.0:  # mm / cm → m",
                "    pos = pos / (1000.0 if np.nanmax(np.abs(pos)) > 30 else 100.0)",
                "montage = mne.channels.make_dig_montage(dict(zip(el['name'].astype(str), "
                "pos)), coord_frame='head')"]
    return [f"montage = mne.channels.read_custom_montage({lit(str(p))})"]


def _loading(recipe: dict, code: _Code, version: str) -> list[str]:
    L = _section(1, "Loading", _describe_loading(recipe))
    std = sorted({x for x in code.externals if x.split()[1].split(".")[0] in
                  ("re", "json", "dataclasses", "pathlib", "math", "warnings", "itertools",
                   "collections", "typing", "copy", "os", "sys", "datetime", "functools")})
    third = sorted({"import mne", "import numpy as np", "import pandas as pd",
                    "from scipy import stats"} | (set(code.externals) - set(std)))
    order = (lambda x: (x.startswith("from"), x))  # noqa: E731  imports, then froms
    L += sorted(std, key=order) + [""] + sorted(third, key=order)
    L += ["", 'mne.set_log_level("error")', ""]
    L += code.helpers(_loading_roots(recipe), version)
    n = code.name
    for st in recipe["load"]["steps"]:
        k = st["kind"]
        if k == "load":
            L += [f"rec = {n('io.load')}({lit(st['path'])}, screen_size={lit(st['screen_size'])},",
                  f"           roles={lit(st['roles'])}, options={lit(st['options'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "load_many":
            L += [assign(f"rec = {n('io.load_many')}(", st["paths"], ","),
                  f"    screen_size={lit(st['screen_size'])},",
                  f"    roles_by_name={lit(st['roles_by_name'])},",
                  f"    options_by_name={lit(st['options_by_name'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "load_bids":
            L += [f"bids_root = {lit(st['root'])}",
                  assign(f"runs = {n('workflow.find_bids_runs')}(bids_root, ", st["runs"], ")"),
                  f"rec = {n('io.load_bids_runs')}(bids_root, runs, "
                  f"screen_size={lit(st['screen_size'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "attach_gaze":
            L += [f"rec.gaze = {n('io.load_eyetracking_file')}({lit(st['path'])}, "
                  f"{lit(st['screen_size'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "gaze_offset":
            L += [f"rec.gaze = raw_gaze.shifted({lit(st['seconds'])})  # gaze offset (s)"]
        elif k == "montage":
            L += _montage_code(st["source"])
            L += [assign("channel_names = ", st["names"]),
                  f"{n('eeg.apply_montage')}(rec.raw, montage, channel_names)"]
        elif k == "trigger_events":
            L += [f"{n('workflow.add_trigger_events')}(rec, {lit(st['channel'])}, "
                  f"mask={lit(st['mask'])}, mode={lit(st['mode'])},",
                  f"    names={lit(st['names'])}, as_stim={lit(st['as_stim'])})"]
        elif k == "remove_trigger_events":
            L += [f"{n('workflow.replace_trigger_events')}(rec, None)"]
    gz = recipe["load"]["gaze"] or {}
    L += ["raw = rec.raw  # the EEG (mne.io.Raw)", ""]
    if gz.get("params") is not None:
        L += [assign("detection = ", gz["params"]),
              f"gaze_events = {n('gaze.detect_events')}(rec.gaze, detection) "
              "if rec.has_gaze else None"]
    else:
        L += ["gaze_events = None"]
    L += [assign(f"aois = {n('gaze.make_aois')}(", gz["aois"], ")") if gz.get("aois")
          else "aois = None",
          f"table = {n('events.build_event_table')}(rec, gaze_events, aois)",
          "print(f'{len(table)} events:', table['name'].value_counts().head(10).to_dict())",
          ""]
    return L


def _preprocessing(recipe: dict, code: _Code, version: str) -> list[str]:
    pre = recipe.get("pre")
    L = _section(2, "Preprocessing", _describe_preprocessing(recipe))
    if pre is None:
        return L + ["data = raw", ""]
    L += code.helpers(["preprocessing.filtered_copy"], version)
    fc = code.name("preprocessing.filtered_copy")
    p, ica = pre["params"], pre["ica"]
    use_ica = bool(p.ica_exclude) and ica is not None
    if use_ica:
        s = ica["steps"]
        L += ["# ICA data: band-passed, average-referenced copy of the good EEG channels",
              f"ica_raw = {fc}(raw, {lit(s['l_freq'])}, {lit(s['h_freq'])}, {lit(s['notch'])},",
              assign("    picks=", s["chans"], ","),
              f"    tmin={lit(s['tmin'])})",
              'ica_raw.info["bads"] = []',
              'ica_raw.set_eeg_reference("average", projection=False)']
        if s["resample"]:
            L += [f"ica_raw.resample({lit(s['resample'])})"]
        L += ["# 1 s segments",
              "events = mne.make_fixed_length_events(ica_raw, duration=1.0)",
              "segments = mne.Epochs(ica_raw, events, tmin=0.0, tmax=1.0, baseline=None,",
              "                      preload=True, reject_by_annotation=True)"]
        if s["segments"] == "autoreject":
            L += ["# bad segments, found by AutoReject",
                  "from autoreject import AutoReject",
                  f"ar = AutoReject(n_interpolate=[1, 2, 4], random_state={s['seed']}, "
                  "n_jobs=1, verbose=False,",
                  "                picks=mne.pick_types(segments.info, eeg=True, eog=False))",
                  f"if len(segments) > {s['autoreject_max']}:  # fit on a random subset",
                  f"    ar.fit(segments[np.sort(np.random.default_rng({s['seed']}).choice(",
                  f"        len(segments), {s['autoreject_max']}, replace=False))])",
                  "else:",
                  "    ar.fit(segments)",
                  "good = ~np.asarray(ar.get_reject_log(segments).bad_epochs, bool)"]
        elif s["segments"] == "threshold" and s["reject_uv"]:
            L += [f"# bad segments: above {s['reject_uv']:g} µV peak-to-peak",
                  "kept = segments.copy().drop_bad(reject=dict(eeg="
                  f"{lit(s['reject_uv'])} * 1e-6))",
                  "good = np.isin(segments.selection, kept.selection)"]
        else:
            L += ["good = np.ones(len(segments), bool)  # every segment"]
        L += ["print(f'ICA fitted on {good.sum()} of {len(good)} segments')",
              f"ica = mne.preprocessing.ICA(n_components={lit(s['n_components'])}, "
              f"method={lit(s['method'])},",
              f"                            fit_params={lit(s['fit_params'])},",
              f"                            max_iter={lit(s['max_iter'])}, rng={lit(s['seed'])})",
              f"ica.fit(segments[np.flatnonzero(good)], decim={lit(s['decim'])})",
              assign("ica.exclude = ", list(p.ica_exclude), "  # removed in the app"),
              ""]
    L += ["# the data for epoching",
          f"clean = {fc}(raw, {lit(p.l_freq)}, {lit(p.h_freq)}, {lit(p.notch)})",
          assign("bads = ", list(p.bads)),
          'clean.info["bads"] = [b for b in bads if b in clean.ch_names]']
    if use_ica:
        L += ["ica.apply(clean)"]
    if p.interpolate_bads:
        L += ['if clean.info["bads"]:', "    clean.interpolate_bads(reset_bads=True)"]
    if p.reference:
        ref = "average" if p.reference in ("average", "avg", "car") else p.reference
        L += [f"clean.set_eeg_reference({lit(ref)}, projection=False)"]
    return L + ["data = clean", ""]


def _epoching(recipe: dict, code: _Code, version: str) -> list[str]:
    L = _section(3, "Epoching", _describe_epoching(recipe))
    L += code.helpers(["workflow.GroupSpec", "workflow.select_events", "epoching.EpochSettings",
                       "epoching.build_group_epochs"], version)
    n = code.name
    L += ["group_specs = ["]
    for g in recipe["groups"]:
        L += [assign("    ", g, ",")]
    L += ["]",
          f"groups = {{g.label: {n('workflow.select_events')}(table, g) for g in group_specs}}",
          assign("settings = ", recipe["settings"]),
          "blinks = gaze_events.blinks if gaze_events is not None else None",
          f"epochs, summary = {n('epoching.build_group_epochs')}(data, groups, settings, blinks)",
          "print(summary.to_string(index=False))",
          ""]
    return L


def _erp(spec, extra: dict, code: _Code, version: str) -> list[str]:
    from . import eeg as E
    ref, label = E.resolve_erp_reference(spec.reference, extra["ch_names"])
    L = _section(4, "Analysis: ERP", _describe_erp(spec, label, extra["n_subjects"]))
    L += code.helpers(["eeg.Component", "eeg.subjects_in", "eeg.epoch_subjects",
                       "eeg.group_component_analysis", "eeg.component_analysis"], version)
    n = code.name
    c = spec.component
    L += [assign("component = ", c),
          assign("analysed = ", list(spec.groups)),
          f"# reference: {label}",
          f"reference = {lit(ref)}",
          f"baseline = {lit(spec.baseline)}",
          "erp_epochs = {}",
          "for g in analysed:",
          "    ep = epochs[g]",
          "    if reference is not None:",
          "        ep = ep.copy().set_eeg_reference(reference, projection=False)",
          "    if baseline:",
          "        ep = ep.copy().apply_baseline(baseline)",
          "    erp_epochs[g] = ep",
          "roi = [ch for ch in component.channels if ch in erp_epochs[analysed[0]].ch_names]",
          f"subjects = {n('eeg.subjects_in')}(erp_epochs)",
          "",
          "if len(subjects) >= 2:",
          "    # one Evoked per subject and group, then the grand average of those",
          "    evokeds = {}",
          "    for g, ep in erp_epochs.items():",
          f"        who = {n('eeg.epoch_subjects')}(ep)",
          "        evokeds[g] = {s: ep[np.flatnonzero(who == s)].average()",
          "                      for s in subjects if (who == s).any()}",
          "    grand = {g: mne.grand_average(list(v.values()), interpolate_bads=False,",
          "                                  drop_bads=False) if len(v) > 1",
          "             else next(iter(v.values())).copy() for g, v in evokeds.items()}",
          "",
          "    # each subject's mean amplitude in the window, on the ROI",
          "    def window_mean(ev):",
          "        m = (ev.times >= component.tmin) & (ev.times <= component.tmax)",
          "        return ev.get_data(picks=roi).mean(0)[m].mean() * 1e6",
          "    means = pd.DataFrame({g: {s: window_mean(ev) for s, ev in v.items()}",
          "                          for g, v in evokeds.items()})",
          "    print(means)",
          "    if len(analysed) == 2:  # subjects are the observations",
          "        both = means.dropna()",
          "        print(stats.ttest_rel(both[analysed[0]], both[analysed[1]]))",
          "",
          "    # the app's tables",
          f"    result = {n('eeg.group_component_analysis')}(erp_epochs, component)",
          "    summary_table = result['summary']",
          "    per_subject = result['subjects']",
          "else:",
          "    grand = {g: ep.average() for g, ep in erp_epochs.items()}",
          "    if len(analysed) == 2:  # trials are the observations",
          "        m = (erp_epochs[analysed[0]].times >= component.tmin) & \\",
          "            (erp_epochs[analysed[0]].times <= component.tmax)",
          "        a, b = (erp_epochs[g].get_data(picks=roi).mean(1)[:, m].mean(1) * 1e6",
          "                for g in analysed)",
          "        print(stats.ttest_ind(a, b, equal_var=False))",
          f"    summary_table, per_epoch = {n('eeg.component_analysis')}(erp_epochs, component)",
          "print(summary_table.T.to_string())",
          "",
          "if __name__ == '__main__':",
          "    mne.viz.plot_compare_evokeds(",
          "        {g: list(v.values()) for g, v in evokeds.items()} if len(subjects) >= 2",
          "        else grand, picks=roi, combine='mean' if len(roi) > 1 else None)",
          ""]
    return L


def _bandpower(spec, tc_band, code: _Code, version: str, number=True) -> list[str]:
    roots = ["workflow.BandPowerSpec", "workflow.BaselineSpec", "workflow.run_bandpower"]
    if tc_band:
        roots.append("workflow.run_timecourse")
    L = _section(4, "Analysis: band power", _describe_bp(spec, tc_band)) if number else []
    L += code.helpers(roots, version)
    n = code.name
    L += [assign("bp_spec = ", spec),
          f"bp = {n('workflow.run_bandpower')}(epochs, data, table, bp_spec)",
          "print(bp['table'].to_string(index=False))"]
    if tc_band:
        L += [f"tc = {n('workflow.run_timecourse')}(bp, epochs, data, {lit(tc_band)})"]
    return L + [""]


def _indices(bp_spec, spec, code: _Code, version: str) -> list[str]:
    L = _section(4, "Analysis: cognitive indices", _describe_idx(bp_spec, spec))
    L += _bandpower(bp_spec, None, code, version, number=False)
    L += code.helpers(["workflow.IndicesSpec", "workflow.run_indices"], version)
    n = code.name
    return L + [assign("idx_spec = ", spec),
                f"idx = {n('workflow.run_indices')}(bp, epochs, rec.gaze, gaze_events, idx_spec)",
                "print(idx['table'].to_string(index=False))", ""]


ANALYSIS_ROOTS = {
    "erp": ["eeg.Component", "eeg.subjects_in", "eeg.epoch_subjects",
            "eeg.group_component_analysis", "eeg.component_analysis"],
    "bandpower": ["workflow.BandPowerSpec", "workflow.BaselineSpec", "workflow.run_bandpower",
                  "workflow.run_timecourse"],
    "indices": ["workflow.BandPowerSpec", "workflow.BaselineSpec", "workflow.run_bandpower",
                "workflow.IndicesSpec", "workflow.run_indices"]}


def build_script(kind: str, recipe: dict, spec, extra=None, version: str = "") -> str:
    """The stand-alone script for an analysis ``kind`` ("erp", "bandpower", "indices").

    ``spec``: the analysis settings (ErpSpec / BandPowerSpec / (BandPowerSpec,
    IndicesSpec)); ``extra``: the epochs' channel names (ERP) or the time-course band."""
    why = check_recipe(recipe)
    if why:
        raise ValueError(why)
    if kind not in TITLES:
        raise ValueError(f"Unknown analysis {kind!r}")
    if not version:
        from . import __version__ as version
    sections = [_loading_roots(recipe), ["preprocessing.filtered_copy"],
                ["workflow.GroupSpec", "workflow.select_events", "epoching.EpochSettings",
                 "epoching.build_group_epochs"], ANALYSIS_ROOTS[kind]]
    code = _Code(sections)
    body = _loading(recipe, code, version) + _preprocessing(recipe, code, version) + \
        _epoching(recipe, code, version)
    if kind == "erp":
        body += _erp(spec, extra, code, version)
    elif kind == "bandpower":
        body += _bandpower(spec, extra, code, version)
    else:
        body += _indices(*spec, code, version)
    text = "\n".join(body)
    externals = code.externals + (["import autoreject"] if "AutoReject" in text else []) \
        + (["import picard"] if "method='picard'" in text else [])
    head = [f'"""{TITLES[kind]} — exported from brAInStudio {version} on '
            f'{_dt.date.today().isoformat()}.',
            "",
            "Repeats the app's analysis, with the settings it ran with, from the data files",
            "to the result, in four sections (cells marked '# %%'):",
            "",
            "  1 Loading        read the recording and build the event table",
            "  2 Preprocessing  filters, bad channels and ICA blink removal",
            "  3 Epoching       epochs around the events of each group",
            f"  4 Analysis       {TITLES[kind]}",
            "",
            "Each section explains what it does, then defines the brAInStudio code it",
            "uses (copied unchanged, so brAInStudio itself is not needed and the numbers",
            "are the app's), then runs its steps.",
            "",
            f"Needs Python ≥ 3.12 and: pip install {_requirements(externals)}",
            '"""',
            "from __future__ import annotations",
            ""]
    return "\n".join(head + body) + "\n"
