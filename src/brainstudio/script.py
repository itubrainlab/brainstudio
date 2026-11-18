"""Python scripts that repeat an analysis of the app, from loading to the result.

A script has four sections (``# %%`` cells): Loading, Preprocessing, Epoching and
Analysis. It is written from the settings each stage actually ran with (the
*recipes* the app keeps with its results), so it gives the same numbers as the app.

The standard steps are written as plain MNE / AutoReject / scipy code. brAInStudio
functions are used where no standard library does the job: reading XDF with stream
roles and clock alignment, gaze event detection, the event table and its groups,
and the app's own measures (component tables, band-power baselines per subject,
cognitive indices).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt

import numpy as np

#: module alias used in scripts for each dataclass
_PREFIX = {"GroupSpec": "W", "ErpSpec": "W", "BaselineSpec": "W", "BandPowerSpec": "W",
           "IndicesSpec": "W", "Component": "E", "ICAParams": "P", "PreprocParams": "P",
           "EpochSettings": "EP", "DetectionParams": "G"}

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


# =========================================================================== sections
def _loading(recipe: dict) -> list[str]:
    L = ["# %% 1 Loading",
         "import mne",
         "import numpy as np",
         "import pandas as pd",
         "from scipy import stats",
         "",
         "from brainstudio import eeg as E",
         "from brainstudio import epoching as EP",
         "from brainstudio import events as EV",
         "from brainstudio import gaze as G",
         "from brainstudio import io as bio",
         "from brainstudio import preprocessing as P",
         "from brainstudio import workflow as W",
         "",
         'mne.set_log_level("error")',
         ""]
    for st in recipe["load"]["steps"]:
        k = st["kind"]
        if k == "load":
            L += ["# one file (XDF: pyxdf, stream roles and clock alignment; others: MNE)",
                  f"rec = bio.load({lit(st['path'])}, screen_size={lit(st['screen_size'])},",
                  f"              roles={lit(st['roles'])}, options={lit(st['options'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "load_many":
            L += ["# several files joined into one recording (one segment per file)",
                  f"rec = bio.load_many({lit(st['paths'])},",
                  f"                    screen_size={lit(st['screen_size'])},",
                  f"                    roles_by_name={lit(st['roles_by_name'])},",
                  f"                    options_by_name={lit(st['options_by_name'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "load_bids":
            L += ["# BIDS runs (MNE-BIDS for the EEG), joined in this order",
                  f"bids_root = {lit(st['root'])}",
                  assign("runs = W.find_bids_runs(bids_root, ", st["runs"], ")"),
                  f"rec = bio.load_bids_runs(bids_root, runs, screen_size={lit(st['screen_size'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "attach_gaze":
            L += ["# eye-tracking data from another file",
                  f"rec.gaze = bio.load_eyetracking_file({lit(st['path'])}, "
                  f"{lit(st['screen_size'])})",
                  "raw_gaze = rec.gaze"]
        elif k == "gaze_offset":
            L += [f"rec.gaze = raw_gaze.shifted({lit(st['seconds'])})  # gaze offset "
                  "(sync correction, s)"]
        elif k == "montage":
            L += ["# channel names (in recording order), types and positions",
                  f"W.set_montage(rec, {lit(st['source'])},",
                  assign("              ", st["names"], ")")]
        elif k == "trigger_events":
            L += ["# trigger channel → events",
                  f"W.add_trigger_events(rec, {lit(st['channel'])}, mask={lit(st['mask'])}, "
                  f"mode={lit(st['mode'])},",
                  f"                     names={lit(st['names'])}, as_stim={lit(st['as_stim'])})"]
        elif k == "remove_trigger_events":
            L += ["W.replace_trigger_events(rec, None)  # trigger events removed"]
    gz = recipe["load"]["gaze"] or {}
    L += ["raw = rec.raw", ""]
    if gz.get("params") is not None:
        L += ["# fixations, saccades and blinks (PyGaze has no offline detectors)",
              assign("detection = ", gz["params"]),
              "gaze_events = G.detect_events(rec.gaze, detection) if rec.has_gaze else None"]
    else:
        L += ["gaze_events = None"]
    if gz.get("aois"):
        L += [assign("aois = W.make_aois(", gz["aois"], ")  # PyGaze AOIs")]
    else:
        L += ["aois = None"]
    L += ["# one row per marker and gaze event, with every marker column",
          "table = EV.build_event_table(rec, gaze_events, aois)", ""]
    return L


def _preprocessing(recipe: dict) -> list[str]:
    pre = recipe.get("pre")
    L = ["# %% 2 Preprocessing"]
    if pre is None:
        return L + ["# none: the epochs were cut from the recording as loaded",
                    "data = raw", ""]
    p, ica = pre["params"], pre["ica"]
    use_ica = bool(p.ica_exclude) and ica is not None
    if use_ica:
        s = ica["steps"]
        L += ["# ICA, as in the ERP guide (neuraldatascience.io/eeg/erp-artifacts): a "
              f"{s['l_freq']:g}–{s['h_freq']:g} Hz,",
              "# average-referenced copy of the good EEG channels"
              + (", from the first event" if s["tmin"] else ""),
              f"ica_raw = P.filtered_copy(raw, {lit(s['l_freq'])}, {lit(s['h_freq'])}, "
              f"{lit(s['notch'])},",
              assign("                          picks=", s["chans"], ","),
              f"                          tmin={lit(s['tmin'])})",
              'ica_raw.info["bads"] = []',
              'ica_raw.set_eeg_reference("average", projection=False)']
        if s["resample"]:
            L += [f"ica_raw.resample({lit(s['resample'])})"]
        L += ["# 1 s segments",
              "events = mne.make_fixed_length_events(ica_raw, duration=1.0)",
              "segments = mne.Epochs(ica_raw, events, tmin=0.0, tmax=1.0, baseline=None,",
              "                      preload=True, reject_by_annotation=True)"]
        if s["segments"] == "autoreject":
            L += ["# bad segments: AutoReject, fitted on at most "
                  f"{s['autoreject_max']} segments",
                  "from autoreject import AutoReject",
                  f"ar = AutoReject(n_interpolate=[1, 2, 4], random_state={s['seed']}, "
                  "n_jobs=1, verbose=False,",
                  "                picks=mne.pick_types(segments.info, eeg=True, eog=False))",
                  f"if len(segments) > {s['autoreject_max']}:",
                  f"    ar.fit(segments[np.sort(np.random.default_rng({s['seed']}).choice(",
                  f"        len(segments), {s['autoreject_max']}, replace=False))])",
                  "else:",
                  "    ar.fit(segments)",
                  "good = ~np.asarray(ar.get_reject_log(segments).bad_epochs, bool)"]
        elif s["segments"] == "threshold" and s["reject_uv"]:
            L += [f"# bad segments: peak-to-peak above {s['reject_uv']:g} µV",
                  f"kept = segments.copy().drop_bad(reject=dict(eeg={lit(s['reject_uv'])} * 1e-6))",
                  "good = np.isin(segments.selection, kept.selection)"]
        else:
            L += ["good = np.ones(len(segments), bool)  # every segment"]
        L += [f"ica = mne.preprocessing.ICA(n_components={lit(s['n_components'])}, "
              f"method={lit(s['method'])},",
              f"                            fit_params={lit(s['fit_params'])},",
              f"                            max_iter={lit(s['max_iter'])}, rng={lit(s['seed'])})",
              f"ica.fit(segments[np.flatnonzero(good)], decim={lit(s['decim'])})",
              assign("ica.exclude = ", list(p.ica_exclude), "  # the components removed in the app"),
              ""]
    L += ["# the data for epoching: filtered, bad channels marked"
          + (", ICA components removed" if use_ica else ""),
          f"clean = P.filtered_copy(raw, {lit(p.l_freq)}, {lit(p.h_freq)}, {lit(p.notch)})",
          assign("bads = ", list(p.bads)),
          'clean.info["bads"] = [b for b in bads if b in clean.ch_names]']
    if use_ica:
        L += ["ica.apply(clean)"]
    if p.interpolate_bads:
        L += ['if clean.info["bads"]:',
              "    clean.interpolate_bads(reset_bads=True)"]
    if p.reference:
        ref = "average" if p.reference in ("average", "avg", "car") else p.reference
        L += [f"clean.set_eeg_reference({lit(ref)}, projection=False)"]
    L += ["data = clean", ""]
    return L


def _epoching(recipe: dict) -> list[str]:
    L = ["# %% 3 Epoching",
         "# event groups: the event names ticked, filters and recordings kept"]
    L += ["group_specs = ["]
    for g in recipe["groups"]:
        L += [assign("    ", g, ",")]
    L += ["]",
          "groups = {g.label: W.select_events(table, g) for g in group_specs}",
          "",
          "# mne.Epochs locked to each group's events (event table as metadata), no",
          "# baseline correction; then the rejections of the app (AutoReject: interpolate",
          "# bad channels within an epoch, drop the epochs it cannot repair)",
          assign("settings = ", recipe["settings"]),
          "blinks = gaze_events.blinks if gaze_events is not None else None",
          "epochs, summary = EP.build_group_epochs(data, groups, settings, blinks)",
          "print(summary.to_string(index=False))",
          ""]
    return L


def _erp(spec, ch_names) -> list[str]:
    from . import eeg as E
    ref, label = E.resolve_erp_reference(spec.reference, ch_names)
    c = spec.component
    L = ["# %% 4 Analysis: ERP",
         assign("component = ", c),
         f"analysed = {lit(list(spec.groups))}",
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
         "subjects = E.subjects_in(erp_epochs)",
         "",
         "if len(subjects) >= 2:",
         "    # group ERP (neuraldatascience.io/eeg/erp-group-viz): one Evoked per subject and",
         "    # group, then the grand average of those, every subject weighing the same",
         "    evokeds = {g: {s: ep[np.flatnonzero(E.epoch_subjects(ep) == s)].average()",
         "                   for s in subjects if (E.epoch_subjects(ep) == s).any()}",
         "               for g, ep in erp_epochs.items()}",
         "    grand = {g: mne.grand_average(list(v.values()), interpolate_bads=False,",
         "                                  drop_bads=False) if len(v) > 1",
         "             else next(iter(v.values())).copy() for g, v in evokeds.items()}",
         "    # mean amplitude in the component window on the ROI, per subject",
         "    def window_mean(ev):",
         "        m = (ev.times >= component.tmin) & (ev.times <= component.tmax)",
         "        return ev.get_data(picks=roi).mean(0)[m].mean() * 1e6",
         "    means = pd.DataFrame({g: {s: window_mean(ev) for s, ev in v.items()}",
         "                          for g, v in evokeds.items()})",
         "    print(means)",
         "    if len(analysed) == 2:  # subjects are the observations: paired t-test",
         "        both = means.dropna()",
         "        print(stats.ttest_rel(both[analysed[0]], both[analysed[1]]))",
         "    # every table of the app (CIs, grand-average peaks, Wilcoxon, d_z,",
         "    # within-subject tests)",
         "    result = E.group_component_analysis(erp_epochs, component)",
         "    summary_table = result['summary']",
         "    per_subject = result['subjects']",
         "else:",
         "    # one subject: trials are the observations",
         "    grand = {g: ep.average() for g, ep in erp_epochs.items()}",
         "    if len(analysed) == 2:",
         "        m = (erp_epochs[analysed[0]].times >= component.tmin) & \\",
         "            (erp_epochs[analysed[0]].times <= component.tmax)",
         "        a, b = (erp_epochs[g].get_data(picks=roi).mean(1)[:, m].mean(1) * 1e6",
         "                for g in analysed)",
         "        print(stats.ttest_ind(a, b, equal_var=False))",
         "    summary_table, per_epoch = E.component_analysis(erp_epochs, component)",
         "print(summary_table.T.to_string())",
         "",
         "if __name__ == '__main__':",
         "    mne.viz.plot_compare_evokeds(",
         "        {g: list(v.values()) for g, v in evokeds.items()} if len(subjects) >= 2",
         "        else grand, picks=roi, combine='mean' if len(roi) > 1 else None)",
         ""]
    return L


def _bandpower(spec, tc_band=None) -> list[str]:
    L = ["# %% 4 Analysis: band power",
         "# Epochs.compute_psd per epoch, averaged in each band, on the ROI mean;",
         "# normalised per subject against the baseline after a baseline event",
         assign("bp_spec = ", spec),
         "bp = W.run_bandpower(epochs, data, table, bp_spec)",
         "print(bp['table'].to_string(index=False))"]
    if tc_band:
        L += [f"# time course of the {tc_band} band (Morlet wavelets), same baseline",
              f"tc = W.run_timecourse(bp, epochs, data, {lit(tc_band)})"]
    return L + [""]


def _indices(bp_spec, spec) -> list[str]:
    return _bandpower(bp_spec)[:-1] + [
        "",
        "# cognitive indices per epoch from the band power, plus eye metrics",
        assign("idx_spec = ", spec),
        "idx = W.run_indices(bp, epochs, rec.gaze, gaze_events, idx_spec)",
        "print(idx['table'].to_string(index=False))", ""]


def build_script(kind: str, recipe: dict, spec, extra=None, version: str = "") -> str:
    """The script for an analysis ``kind`` ("erp", "bandpower" or "indices").

    ``spec``: the analysis settings (ErpSpec / BandPowerSpec / (BandPowerSpec,
    IndicesSpec)); ``extra``: the epochs' channel names (ERP) or the time-course band."""
    why = check_recipe(recipe)
    if why:
        raise ValueError(why)
    if not version:
        from . import __version__ as version
    head = [f'"""{TITLES[kind]} — exported from brAInStudio {version} on '
            f'{_dt.date.today().isoformat()}.',
            "",
            "Repeats the app's analysis with the settings it ran with, in four sections:",
            "1 Loading, 2 Preprocessing, 3 Epoching, 4 Analysis (cells: '# %%').",
            f"Needs: pip install brainstudio=={version}",
            '"""', ""]
    body = _loading(recipe) + _preprocessing(recipe) + _epoching(recipe)
    if kind == "erp":
        body += _erp(spec, extra)
    elif kind == "bandpower":
        body += _bandpower(spec, extra)
    elif kind == "indices":
        body += _indices(*spec)
    else:
        raise ValueError(f"Unknown analysis {kind!r}")
    return "\n".join(head + body) + "\n"

