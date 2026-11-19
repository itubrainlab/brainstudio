"""Short, readable Python scripts that repeat an analysis of the app from the data files.

A script has four cells (``# %%``): Loading, Preprocessing, Epoching and Analysis. It
uses only standard libraries (MNE, MNE-BIDS, pyxdf, pymovements, AutoReject, scipy,
pandas) with the settings the results on screen were computed with (the *recipes* the
app keeps), written in as plain values. Only the steps that apply are written.

The scripts favour brevity over replicating every detail of the app (e.g. the app's
XDF clock handling is reduced to a few lines), so their numbers are close to the app's
but not identical.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import re
from pathlib import Path

import numpy as np

TITLES = {"erp": "ERP", "bandpower": "Band power", "indices": "Cognitive indices"}
_PREFIX: dict = {}

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



# =========================================================================== text helpers
def _c(text: str) -> list[str]:
    """Short ``#`` comment lines."""
    import textwrap
    return textwrap.wrap(text, WIDTH - 2, initial_indent="# ", subsequent_indent="# ")


def _cell(n: int, title: str, what: str) -> list[str]:
    return ["", "", f"# %% {n} {title}", *_c(what)]


# =========================================================================== loading
def _xdf_reader(streams: list[dict], screen) -> tuple[list[str], dict]:
    """``read_xdf(path)`` for files with these streams (names, roles, columns)."""
    by_role = {r: [s for s in streams if s.get("role") == r] for r in ("EEG", "gaze", "markers")}
    eeg = by_role["EEG"][0] if by_role["EEG"] else None
    gz = by_role["gaze"][0] if by_role["gaze"] else None
    L = ["def read_xdf(path):",
         '    """EEG (MNE Raw), events and gaze (pandas) of one XDF file, in seconds from',
         '    the first EEG sample."""',
         "    streams = {s['info']['name'][0]: s for s in "
         "pyxdf.load_xdf(path, dejitter_timestamps=False)[0]}"]
    used = {"pyxdf", "ndimage"}
    if eeg is not None:
        L += ["",
              f"    eeg = streams[{eeg['name']!r}]",
              "    # start and rate from a straight line through the (jittered) timestamps",
              "    step, t0 = np.polyfit(np.arange(len(eeg['time_stamps'])), "
              "eeg['time_stamps'], 1)",
              "    sfreq = 1 / step",
              "    info = mne.create_info(CHANNELS, sfreq, CHANNEL_TYPES)",
              "    raw = mne.io.RawArray(np.asarray(eeg['time_series'], float).T "
              "* 1e-6, info)  # µV → V"]
    else:
        L += ["    t0, raw = None, None"]
    L += ["",
          "    def lsl_time(s):",
          '        """Timestamps on the EEG clock. Some trackers (e.g. Gazepoint) stamp',
          "        samples with their own clock and only a few with the LSL clock: the",
          "        difference between the two gives the offset. Regularly sampled streams",
          "        get their jitter removed while following any drift: a straight line plus",
          '        its 5 s trend."""',
          "        ts = s['time_stamps'].copy()",
          "        own = np.abs(ts - np.median(ts)) < 1e4",
          "        if (~own).any() and own.any():  # runs of samples on the other clock",
          "            j = np.flatnonzero(~own)",
          "            runs = np.split(j, np.flatnonzero(np.diff(j) > 1) + 1)",
          "            ts = ts + np.median([(ts[r[0]] + ts[r[-1]] - ts[r[0] - 1] - "
          "ts[r[-1] + 1]) / 2 for r in runs])",
          "        if float(s['info']['nominal_srate'][0]) > 0 and own.sum() > 2:",
          "            t = ts[own]",
          "            k = np.arange(len(t))",
          "            slope, start = np.polyfit(k, t, 1)",
          "            line = start + slope * k",
          "            trend = uniform_filter1d(t - line, max(3, round(5 / slope)), "
          "mode='nearest')",
          "            ts[own] = np.maximum.accumulate(line + trend) "
          "if np.abs(trend).max() > max(0.01, 2 * slope) else line",
          "        return ts - t0, own",
          "",
          "    rows = []"]
    for m in by_role["markers"]:
        L += [f"    ts, own = lsl_time(streams[{m['name']!r}])",
              f"    for t, v in zip(ts[own], np.asarray(streams[{m['name']!r}]['time_series'],"
              " dtype=object)[own]):",
              "        if str(v[0]).strip():",
              "            name, *attrs = str(v[0]).split('|')  # 'NAME|key=value|…'",
              "            rows.append({'onset': t, 'name': name.strip(),",
              "                         **dict(a.split('=', 1) for a in attrs if '=' in a)})"]
    L += ["    events = pd.DataFrame(rows)", ""]
    gaze_cols = {}
    if gz is not None:
        cols = (gz.get("options") or {}).get("columns") or \
            (gz.get("auto_options") or {}).get("gaze", {}).get("columns") or []
        labels = list(gz.get("labels") or [])
        if isinstance(labels, str):
            labels = eval(labels)  # noqa: S307  (stored as a list repr)
        if not cols:
            from .io.xdf import gaze_columns
            m = gaze_columns([lab.lower() for lab in labels], len(labels))
            cols = ["extra"] * len(labels)
            for role in ("x", "y", "pupil", "valid"):
                for k in m.get(role, []):
                    cols[k] = role
        pick = {r: [labels[k] for k, c in enumerate(cols) if c == r]
                for r in ("x", "y", "pupil", "valid")}
        gaze_cols = pick
        L += [f"    g = streams[{gz['name']!r}]",
              "    ts, own = lsl_time(g)",
              "    s = pd.DataFrame(np.asarray(g['time_series'], float)[own], "
              f"columns={labels!r})",
              "    lost = (s[" + repr(pick["x"][0]) + "] == 0) & (s[" + repr(pick["y"][0]) + "] == 0)"]
        if pick["valid"]:
            L += [f"    lost |= s[{pick['valid'][0]!r}] != 1"]
        L += ["    gaze = pd.DataFrame({'t': ts[own],",
              f"                         'x': s[{pick['x'][0]!r}].where(~lost),",
              f"                         'y': s[{pick['y'][0]!r}].where(~lost)}})"]
        if pick["pupil"]:
            L += [f"    gaze['pupil'] = s[{pick['pupil'][0]!r}].where(~lost & "
                  f"(s[{pick['pupil'][0]!r}] > 0))"]
        w, h = screen
        L += ["    if gaze[['x', 'y']].abs().quantile(0.95).max() <= 1.5:  # 0–1 → px",
              f"        gaze[['x', 'y']] *= {list(screen)!r}",
              "    # samples far outside the screen are tracker glitches",
              f"    off = ~gaze['x'].between({-0.5 * w:g}, {1.5 * w:g}) | "
              f"~gaze['y'].between({-0.5 * h:g}, {1.5 * h:g})",
              "    gaze.loc[off, ['x', 'y']] = np.nan"]
    else:
        L += ["    gaze = None"]
    L += ["    return raw, events, gaze"]
    return L, gaze_cols


def _loading(recipe: dict) -> tuple[list[str], set]:
    """The Loading cell and the imports it needs."""
    from . import eeg as E
    steps = recipe["load"]["steps"]
    first = steps[0]
    k = first["kind"]
    imports = {"mne", "numpy", "pandas"}
    L = []
    screen = tuple(first.get("screen_size") or (1920, 1080))
    rec_info = first.get("recording", {})
    if k in ("load", "load_many") and str(first.get("path", first.get("paths", [""])[0])) \
            .lower().endswith((".xdf", ".xdfz")):
        imports |= {"pyxdf", "ndimage"}
        what = ("Reads the XDF file with pyxdf: the EEG stream becomes an MNE Raw, marker "
                "streams become events and the gaze stream a table in screen pixels, all in "
                "seconds from the first EEG sample." if k == "load" else
                "Reads each XDF file (EEG, events, gaze) with pyxdf and joins them end to "
                "end; each part keeps its subject/session/run/file.")
        L += _cell(1, "Loading", what)
        L += [assign("CHANNELS = ", rec_info.get("channels_in", [])),
              assign("CHANNEL_TYPES = ", rec_info.get("types_in", [])), ""]
        reader, _ = _xdf_reader(rec_info.get("streams", []), screen)
        L += reader + [""]
        if k == "load":
            L += [f"raw, events, gaze = read_xdf({lit(first['path'])})"]
        else:
            L += _join_parts(first["paths"], rec_info.get("entities", {}), "read_xdf")
    elif k == "load_bids":
        imports |= {"mne_bids", "json", "warnings"}
        L += _cell(1, "Loading", "Reads the BIDS runs with MNE-BIDS (EEG, events.tsv) "
                   "plus their eye-tracking files, and joins them end to end; each part "
                   "keeps its subject, session and run.")
        L += [f"ROOT = Path({lit(first['root'])})",
              "RUNS = ["]
        for r in rec_info.get("runs", []):
            L += [assign("    ", r, ",")]
        L += ["]", "",
              "def read_run(run):",
              '    """EEG, events and gaze of one BIDS run (seconds from its start)."""',
              "    path = ROOT / run['eeg']",
              "    with warnings.catch_warnings():  # MNE-BIDS notes about EOG positions etc.",
              "        warnings.simplefilter('ignore')",
              "        raw = read_raw_bids(get_bids_path_from_fname(path), verbose=False)",
              "    raw.load_data()",
              "    events = pd.read_csv(str(path).rsplit('_eeg', 1)[0] + '_events.tsv', "
              "sep='\\t')",
              "    events['name'] = events['trial_type']",
              "    gaze = None",
              "    if run.get('eye'):",
              "        eye = ROOT / run['eye']",
              "        meta = json.loads(eye.with_name(eye.name.replace('.tsv.gz', '.json'))"
              ".read_text(encoding='utf-8'))",
              "        g = pd.read_csv(eye, sep='\\t', header=None, names=meta['Columns'], "
              "na_values='n/a')",
              "        gaze = pd.DataFrame({'t': meta.get('StartTime', 0) + np.arange(len(g)) "
              "/ meta['SamplingFrequency'],",
              "                             'x': g[run['x']], 'y': g[run['y']]})",
              "        if run.get('pupil'):",
              "            gaze['pupil'] = g[run['pupil']].where(g[run['pupil']] > 0)",
              "    return raw, events, gaze", ""]
        L += _join_parts(None, None, "read_run")
    else:
        what = "Reads the file with MNE; its annotations become the events."
        L += _cell(1, "Loading", what)
        paths = [first["path"]] if k == "load" else first["paths"]
        L += ["def read_file(path):",
              "    raw = mne.io.read_raw(path, preload=True)",
              "    events = pd.DataFrame({'onset': raw.annotations.onset - raw.first_time,",
              "                           'name': raw.annotations.description})",
              "    return raw, events, None", ""]
        if k == "load":
            L += [f"raw, events, gaze = read_file({lit(paths[0])})"]
        else:
            L += _join_parts(paths, rec_info.get("entities", {}), "read_file")
    # later steps
    for st in steps[1:]:
        if st["kind"] == "gaze_offset":
            L += [f"gaze['t'] += {lit(st['seconds'])}  # gaze offset (sync correction, s)"]
        elif st["kind"] == "montage":
            L += [""] + _montage_code(st["source"])
            L += [assign("names = ", st["names"]),
                  "raw.rename_channels(dict(zip(raw.ch_names, names)))",
                  "raw.set_montage(montage, on_missing='ignore')"]
        elif st["kind"] == "trigger_events":
            L += ["", f"# trigger channel {st['channel']} → events",
                  f"found = mne.find_events(raw, stim_channel={lit(st['channel'])}, "
                  f"mask={lit(st['mask'])}, mask_type='and', shortest_event=1)",
                  assign("codes = ", st["names"]),
                  "events = pd.concat([events, pd.DataFrame(",
                  "    {'onset': (found[:, 0] - raw.first_samp) / raw.info['sfreq'],",
                  "     'name': [codes.get(c) for c in found[:, 2]]}).dropna()])"]
            if st["as_stim"]:
                L += [f"raw.set_channel_types({{{st['channel']!r}: 'stim'}})"]
        elif st["kind"] == "remove_trigger_events":
            L += ["# (trigger events were removed again)"]
        elif st["kind"] == "attach_gaze":
            raise ValueError("Scripts cannot yet repeat attaching a separate eye-tracking "
                             "file")
    L += ["events = events.sort_values('onset', ignore_index=True)",
          "print(raw)",
          "print(events['name'].value_counts().head(10))"]
    return L, imports


def _join_parts(paths, entities, reader) -> list[str]:
    """Read several parts and join them end to end."""
    if reader == "read_run":
        L = ["parts = [(r, *read_run(r)) for r in RUNS]"]
        tag = "**{k: r[k] for k in ('subject', 'session', 'run') if r.get(k)}"
    else:
        L = [assign("FILES = ", {p: entities.get(p, {}) for p in paths}),
             f"parts = [(entities, *{reader}(path)) for path, entities in FILES.items()]"]
        tag = "**r"
    return L + [
        "starts = np.cumsum([0] + [p[1].n_times / p[1].info['sfreq'] for p in parts[:-1]])",
        "events = pd.concat([ev.assign(onset=ev['onset'] + t, " + tag + ")",
        "                    for (r, _, ev, _), t in zip(parts, starts)], ignore_index=True)",
        "gaze = (pd.concat([g.assign(t=g['t'] + t) for (_, _, _, g), t in zip(parts, starts)",
        "                   if g is not None], ignore_index=True)",
        "        if any(p[3] is not None for p in parts) else None)",
        "raw = mne.concatenate_raws([p[1] for p in parts])  # boundaries are annotated"]


def _montage_code(source) -> list[str]:
    from . import eeg as E
    if not source:
        return ["montage = None"]
    layouts = E.layout_montages()
    if source in layouts:
        names, template = E.read_channel_layout(layouts[source])
        import mne
        if template == "standard_1005" and "colin27_1005" in mne.channels.get_builtin_montages():
            template = "colin27_1005"
        return [f"# {source}: these electrodes at the {template} template's positions",
                assign("layout = ", names),
                f"pos = mne.channels.make_standard_montage({lit(template)}).get_positions()",
                "lookup = {k.upper(): v for k, v in pos['ch_pos'].items()}",
                "montage = mne.channels.make_dig_montage(",
                "    ch_pos={n: lookup[n.upper()] for n in layout}, nasion=pos['nasion'],",
                "    lpa=pos['lpa'], rpa=pos['rpa'], coord_frame=pos['coord_frame'])"]
    if not Path(source).exists():
        return [f"montage = mne.channels.make_standard_montage({lit(source)})"]
    if source.lower().endswith(".fif"):
        return [f"montage = mne.channels.read_dig_fif({lit(source)})"]
    return [f"montage = mne.channels.read_custom_montage({lit(source)})"]


def _gaze_events(recipe: dict) -> tuple[list[str], set]:
    gz = recipe["load"]["gaze"] or {}
    p = gz.get("params")
    if p is None:
        return [], set()
    from . import gaze as G
    key = G.preset_name(p)
    L = ["", "# fixations (I-DT), saccades (Engbert & Kliegl) and blinks with pymovements",
         *(_c(f"Preset: {G.PRESETS[key][0]}.") if key else []),
         "sf = 1 / np.polyfit(np.arange(len(gaze)), gaze['t'], 1)[0]  # sampling rate",
         "n = lambda seconds, least=1: max(int(round(seconds * sf)), least)  # → samples",
         "xy = gaze[['x', 'y']].to_numpy().T",
         "pg = pm.gaze.from_numpy(time=np.arange(len(gaze)), time_unit='ms', pixel=xy, "
         "position=xy,",
         "                        experiment=pm.Experiment(sampling_rate=sf))",
         "pupil = gaze['pupil'] if 'pupil' in gaze else gaze['x'].notna().astype(float)"
         ".where(gaze['x'].notna())",
         "pg.samples = pg.samples.with_columns(pl.Series('pupil', pupil.to_numpy(float)))",
         f"pg.pos2vel(method={lit(p.velocity_method)})",
         f"pg.detect('idt', dispersion_threshold={lit(p.fix_dispersion)}, "
         f"minimum_duration=n({lit(p.fix_min_dur)}, 2))",
         f"pg.detect('microsaccades', threshold_factor={lit(p.sac_threshold_factor)}, "
         f"minimum_duration=n({lit(p.sac_min_dur)}), name='saccade')",
         f"pg.detect('blink', minimum_duration=n({lit(p.blink_min_dur)}), "
         f"maximum_duration=n({lit(p.blink_max_dur)}), minimum_gap=n(0.005))",
         "pg.compute_event_properties('location', name='fixation')",
         "gaze_events = pg.events.frame.to_pandas()",
         "t = gaze['t'].to_numpy()",
         "gaze_events['duration'] = t[gaze_events['offset']] - t[gaze_events['onset']]",
         "gaze_events['onset'] = t[gaze_events['onset']]",
         "loc = gaze_events['location'].dropna()",
         "gaze_events.loc[loc.index, ['x', 'y']] = np.vstack(loc.to_numpy())"]
    if gz.get("aois"):
        L += [assign("AOIS = ", gz["aois"]) + "  # left, top, width, height (px)",
              "gaze_events['aoi'] = ''",
              "for aoi, (left, top, w, h) in AOIS.items():",
              "    inside = gaze_events['x'].between(left, left + w) & "
              "gaze_events['y'].between(top, top + h)",
              "    gaze_events.loc[inside & (gaze_events['aoi'] == ''), 'aoi'] = aoi"]
    L += ["events = pd.concat([events, gaze_events.drop(columns=['offset', 'location'])],",
          "                   ignore_index=True).sort_values('onset', ignore_index=True)",
          "print(gaze_events['name'].value_counts())"]
    return L, {"pymovements", "polars"}


# =========================================================================== preprocessing
def _preprocessing(recipe: dict) -> tuple[list[str], set]:
    pre = recipe.get("pre")
    if pre is None:
        return _cell(2, "Preprocessing", "None: the epochs are cut from the data as "
                     "loaded.") + ["data = raw"], set()
    p, ica = pre["params"], pre["ica"]
    use_ica = bool(p.ica_exclude) and ica is not None
    hp = f"{p.l_freq:g} Hz high-pass" if p.l_freq else "no high-pass"
    what = (f"Filters ({hp}, {p.h_freq:g} Hz low-pass"
            + (f", {p.notch:g} Hz notch" if p.notch else "") + ")"
            + (f", bad channels {', '.join(p.bads)}" if p.bads else "")
            + ("; removes eye blinks with ICA, fitted as in the ERP guide "
               "(neuraldatascience.io/eeg/erp-artifacts) on a 1–30 Hz average-referenced "
               "copy cut into 1 s segments" + (", the bad ones found by AutoReject"
                                               if ica["steps"]["segments"] == "autoreject"
                                               else "") if use_ica else "") + ".")
    L = _cell(2, "Preprocessing", what)
    imports = set()
    if p.bads:
        L += [assign("raw.info['bads'] = ", list(p.bads))]

    def filters(var, lo, hi, notch):
        out = []
        if lo and lo < 0.5:
            out += [f"{var}.filter({lit(lo)}, None, method='iir', "
                    "iir_params=dict(order=4, ftype='butter', output='sos'))"]
            lo = None
        if lo or hi:
            out += [f"{var}.filter({lit(lo)}, {lit(hi)})"]
        if notch:
            out += [f"{var}.notch_filter(np.arange({lit(notch)}, {var}.info['sfreq'] / 2, "
                    f"{lit(notch)}))"]
        return out

    if use_ica:
        s = ica["steps"]
        imports.add("autoreject") if s["segments"] == "autoreject" else None
        L += ["", "# ICA on a separate copy"
              + (f", from {s['tmin']:g} s (the first event)" if s["tmin"] else "")]
        L += [f"ica_raw = raw.copy().pick({lit(s['chans'])})"
              if len(lit(s["chans"])) < 60 else assign("ica_raw = raw.copy().pick(",
                                                         s["chans"], ")")]
        if s["tmin"]:
            L += [f"ica_raw.crop(tmin={lit(s['tmin'])})"]
        L += filters("ica_raw", s["l_freq"], s["h_freq"], s["notch"])
        L += ["ica_raw.set_eeg_reference('average')"]
        if s["resample"]:
            L += [f"ica_raw.resample({lit(s['resample'])})"]
        L += ["events_1s = mne.make_fixed_length_events(ica_raw, duration=1.0)",
              "segments = mne.Epochs(ica_raw, events_1s, tmin=0, tmax=1.0, baseline=None, "
              "preload=True)"]
        if s["segments"] == "autoreject":
            n = s["autoreject_max"]
            L += [f"ar = AutoReject(n_interpolate=[1, 2, 4], random_state={s['seed']}, "
                  "verbose=False)",
                  f"fit_on = np.sort(np.random.default_rng({s['seed']}).choice(len(segments), "
                  f"{n}, replace=False)) if len(segments) > {n} else slice(None)",
                  "good = ~ar.fit(segments[fit_on]).get_reject_log(segments).bad_epochs"]
        elif s["segments"] == "threshold" and s["reject_uv"]:
            L += [f"segments.drop_bad(reject=dict(eeg={lit(s['reject_uv'])}e-6))",
                  "good = np.ones(len(segments), bool)"]
        else:
            L += ["good = np.ones(len(segments), bool)"]
        L += [f"ica = mne.preprocessing.ICA(n_components={lit(s['n_components'])}, "
              f"method={lit(s['method'])},",
              f"                            fit_params={lit(s['fit_params'])}, "
              f"max_iter={s['max_iter']}, rng={s['seed']})",
              f"ica.fit(segments[good], decim={s['decim']})",
              f"ica.exclude = {lit(list(p.ica_exclude))}  # chosen in the app (blinks)"]
    L += ["", "data = raw.copy()"] + filters("data", p.l_freq, p.h_freq, p.notch)
    if use_ica:
        L += ["ica.apply(data)"]
    if p.interpolate_bads and p.bads:
        L += ["data.interpolate_bads()"]
    return L, imports


# =========================================================================== epoching
def _epoching(recipe: dict) -> tuple[list[str], set]:
    st = recipe["settings"]
    groups = recipe["groups"]
    what = (f"Cuts {st.tmin:g} … {st.tmax:g} s around the events of each group, without "
            "baseline correction"
            + (f"; drops epochs above {st.reject_uv:g} µV peak-to-peak" if st.reject_uv else "")
            + ("; AutoReject interpolates bad channels per epoch and drops the epochs it "
               "cannot repair" if st.autoreject else "") + ".")
    L = _cell(3, "Epoching", what)
    L += ["GROUPS = {"]
    for g in groups:
        sel = {"names": list(g.names)}
        if g.query.strip():
            sel["query"] = g.query
        rec = {k: v for k, v in g.recordings.items() if k in ("subject", "session", "run")}
        if rec:
            sel["keep"] = rec
        L += [assign(f"    {g.label!r}: ", sel, ",")]
    L += ["}", "",
          "def select(group):",
          "    rows = events[events['name'].isin(group['names'])]",
          "    if group.get('query'):",
          "        rows = rows.query(group['query'])",
          "    for column, values in group.get('keep', {}).items():  # recordings kept",
          "        if column in rows:",
          "            rows = rows[rows[column].astype(str).isin(values)]",
          "    return rows.reset_index(drop=True)", ""]
    if st.delay_ms:
        L += [f"events['onset'] += {lit(st.delay_ms / 1000)}  # EEG delay"]
    limits = "".join([f", reject=dict(eeg={lit(st.reject_uv)}e-6)" if st.reject_uv else "",
                      f", flat=dict(eeg={lit(st.flat_uv)}e-6)" if st.flat_uv else ""])
    L += ["epochs = {}",
          "for label, group in GROUPS.items():",
          "    rows = select(group)",
          "    samples = np.round(rows['onset'] * data.info['sfreq']).astype(int) "
          "+ data.first_samp",
          "    rows, samples = rows[~samples.duplicated()], samples[~samples.duplicated()]",
          "    epochs[label] = mne.Epochs(data, np.c_[samples, np.zeros_like(samples), "
          "np.ones_like(samples)],",
          f"                               tmin={lit(st.tmin)}, tmax={lit(st.tmax)}, "
          "baseline=None, picks='eeg',",
          f"                               metadata=rows, preload=True{limits})"]
    imports = set()
    if st.autoreject:
        imports.add("autoreject")
        L += ["", "# AutoReject fitted on (at most 600 of) the epochs of every group",
              "every = mne.concatenate_epochs(list(epochs.values()))",
              "fit_on = np.sort(np.random.default_rng(42).choice(len(every), 600, "
              "replace=False)) if len(every) > 600 else slice(None)",
              "ar = AutoReject(n_interpolate=[1, 2, 4], random_state=42, verbose=False)",
              "ar.fit(every[fit_on])",
              "epochs = {label: ar.transform(ep) for label, ep in epochs.items()}"]
    L += ["for label, ep in epochs.items():",
          "    print(f'{label}: {len(ep)} epochs')"]
    return L, imports


# =========================================================================== analyses
def _erp(spec, extra) -> tuple[list[str], set]:
    from . import eeg as E
    ref, label = E.resolve_erp_reference(spec.reference, extra["ch_names"])
    c = spec.component
    groups = list(spec.groups)
    multi = extra.get("n_subjects", 0) >= 2
    bl = f"{spec.baseline[0]:g} … {spec.baseline[1]:g} s" if spec.baseline else "no"
    what = (f"ERP {c.name}: mean amplitude {c.tmin * 1000:.0f}–{c.tmax * 1000:.0f} ms on "
            f"{', '.join(c.channels)} ({label}, {bl} baseline)."
            + (" Each subject's epochs are averaged first, then the subjects (grand "
               "average); subjects are compared with a paired t-test "
               "(neuraldatascience.io/eeg/erp-group-stats)." if multi else
               " Trials are the observations (Welch t-test)."))
    L = _cell(4, "Analysis: ERP", what)
    L += [assign("ROI = ", list(c.channels)),
          f"ROI = [ch for ch in ROI if ch in epochs[{groups[0]!r}].ch_names]  # good channels",
          f"WINDOW = ({lit(c.tmin)}, {lit(c.tmax)})",
          "erp = {}",
          f"for label in {lit(groups)}:",
          "    ep = epochs[label].copy()"]
    if ref is not None:
        L += [f"    ep.set_eeg_reference({lit(ref)})"]
    if spec.baseline:
        L += [f"    ep.apply_baseline({lit(spec.baseline)})"]
    L += ["    erp[label] = ep", ""]
    a, b = (groups + [None])[:2]
    if multi:
        L += ["# one Evoked per subject and group, then the grand average",
              "evokeds = {label: {s: ep[ep.metadata['subject'] == s].average()",
              "                   for s in sorted(ep.metadata['subject'].unique())}",
              "           for label, ep in erp.items()}",
              "grand = {label: mne.grand_average(list(v.values())) for label, v in "
              "evokeds.items()}",
              "",
              "# each subject's mean amplitude in the window (µV)",
              "def window_mean(ev):",
              "    in_window = (ev.times >= WINDOW[0]) & (ev.times <= WINDOW[1])",
              "    return ev.get_data(picks=ROI)[:, in_window].mean() * 1e6",
              "means = pd.DataFrame({label: {s: window_mean(ev) for s, ev in v.items()}",
              "                      for label, v in evokeds.items()})",
              "print(means)"]
        if b:
            L += [f"print(stats.ttest_rel(means[{a!r}], means[{b!r}], nan_policy='omit'))"]
        L += ["", "mne.viz.plot_compare_evokeds({label: list(v.values()) for label, v in "
              "evokeds.items()},",
              "                             picks=ROI, combine='mean')"]
    else:
        L += ["grand = {label: ep.average() for label, ep in erp.items()}",
              "in_window = (erp[" + repr(groups[0]) + "].times >= WINDOW[0]) & "
              "(erp[" + repr(groups[0]) + "].times <= WINDOW[1])",
              "trials = {label: ep.get_data(picks=ROI)[:, :, in_window].mean(axis=(1, 2)) * 1e6",
              "          for label, ep in erp.items()}",
              "print({label: round(v.mean(), 2) for label, v in trials.items()})"]
        if b:
            L += [f"print(stats.ttest_ind(trials[{a!r}], trials[{b!r}], equal_var=False))"]
        L += ["", "mne.viz.plot_compare_evokeds(grand, picks=ROI, combine='mean')"]
    return L, {"scipy"}


def _bandpower_core(spec) -> list[str]:
    from . import bandpower as BP  # noqa: F401
    b = spec.baseline
    groups = list(spec.groups)
    L = [assign("BANDS = ", {k: list(v) for k, v in spec.bands.items()}),
         assign("ROI = ", list(spec.roi)) if spec.roi else "ROI = None  # every channel",
         f"WINDOW = ({lit(spec.tmin)}, {lit(spec.tmax)})",
         "",
         "def band_power(ep, window=WINDOW):",
         '    """Mean band power on the ROI per epoch (µV²): Welch PSD (segments of up to 2 s,',
         '    50 % overlap) summed over each band."""',
         "    seg = min(int(round((window[1] - window[0]) * ep.info['sfreq'])) + 1, "
         "int(2 * ep.info['sfreq']))",
         f"    psd = ep.compute_psd(method={lit(spec.method)}, fmin=min(lo for lo, _ in "
         "BANDS.values()),",
         "                         fmax=min(max(hi for _, hi in BANDS.values()), "
         "ep.info['sfreq'] / 2),",
         "                         tmin=window[0], tmax=window[1], picks='eeg', exclude=(),",
         "                         n_fft=seg, n_per_seg=seg, n_overlap=seg // 2)",
         "    p, f = psd.get_data(picks=ROI, return_freqs=True)",
         "    return pd.DataFrame({b: p[:, :, (f >= lo) & (f < hi)].sum(-1).mean(-1) "
         "* np.diff(f).mean() * 1e12",
         "                         for b, (lo, hi) in BANDS.items()})",
         ""]
    ref = spec.reference if spec.reference not in ("recorded", "as recorded") else None
    L += [f"bp = {{label: band_power(epochs[label]"
          + (f".copy().set_eeg_reference({lit(ref)})" if ref else "") + ")",
          f"      for label in {lit(groups)}}}"]
    if b is not None:
        L += ["",
              f"# baseline: {b.t0:g}–{b.t1:g} s after each '{b.event}', cut into "
              "windows as long as the analysis window",
              f"starts = events[events['name'] == {b.event!r}]"
              + (f".query({b.query!r})" if b.query else ""),
              "seg = WINDOW[1] - WINDOW[0]",
              f"rows = pd.DataFrame([dict(r, onset=r['onset'] + t) for r in "
              "starts.to_dict('records')",
              f"                     for t in np.arange({lit(b.t0)}, {lit(b.t1)} - seg + 1e-9, "
              "seg)])",
              "samples = np.round(rows['onset'] * data.info['sfreq']).astype(int) + "
              "data.first_samp",
              "base_ep = mne.Epochs(data, np.c_[samples, 0 * samples, 1 + 0 * samples], tmin=0,",
              "                     tmax=seg - 1 / data.info['sfreq'], baseline=None, "
              "picks='eeg', metadata=rows, preload=True)"
              + (f"\nbase_ep.set_eeg_reference({lit(ref)})" if ref else ""),
              "base = band_power(base_ep, (base_ep.tmin, base_ep.tmax))",
              "if 'subject' in rows:  # each subject against its own baseline",
              "    base = base.groupby(base_ep.metadata['subject'].to_numpy()).mean()",
              "    level = {label: base.loc[epochs[label].metadata['subject']].to_numpy() "
              "for label in bp}",
              "else:",
              "    level = {label: base.mean().to_numpy() for label in bp}"]
        norm = {"dB": "10 * np.log10(v / level[label])",
                "% change": "100 * (v - level[label]) / level[label]",
                "ratio": "v / level[label]"}.get(spec.norm, "10 * np.log10(v / level[label])")
        L += [f"bp = {{label: {norm} for label, v in bp.items()}}  # {spec.norm}"]
    return L


def _bandpower(spec, tc_band) -> tuple[list[str], set]:
    b = spec.baseline
    what = (f"Band power in {spec.tmin:g} … {spec.tmax:g} s of each epoch (Welch PSD), "
            + (f"in {spec.norm} relative to each subject's baseline after '{b.event}'"
               if b else "absolute") + "; groups compared per band (Welch t-test).")
    L = _cell(4, "Analysis: band power", what) + _bandpower_core(spec)
    groups = list(spec.groups)
    L += ["", "for label, v in bp.items():",
          "    print(label, v.mean().round(3).to_dict())"]
    if len(groups) == 2:
        L += ["for band in BANDS:",
              f"    print(band, stats.ttest_ind(bp[{groups[0]!r}][band], "
              f"bp[{groups[1]!r}][band], equal_var=False, nan_policy='omit'))"]
    if tc_band:
        L += ["", f"# {tc_band} power over the epoch (Morlet wavelets), dB vs the "
              "epoch's mean",
              f"lo, hi = BANDS[{tc_band!r}]",
              "freqs = np.arange(lo, hi + 1e-9, max((hi - lo) / 6, 0.5))",
              "for label, ep in epochs.items():",
              "    tfr = ep.compute_tfr('morlet', freqs, n_cycles=np.maximum(freqs / 2, 3), "
              "picks=ROI, average=True)",
              "    tfr.plot(combine='mean', baseline=(None, None), mode='logratio', "
              "title=label)"]
    return L, {"scipy"}


def _indices(bp_spec, spec) -> tuple[list[str], set]:
    from . import bandpower as BP
    rois = {k: list(v) for k, v in {**BP.DEFAULT_ROIS, **spec.rois}.items()}
    what = (f"Cognitive indices per epoch in {bp_spec.tmin:g} … {bp_spec.tmax:g} s, from "
            "absolute band power (Welch PSD) averaged over channel regions: engagement "
            "β/(α+θ), workload frontal-midline θ / parietal α, θ/α, frontal θ/β and frontal "
            "alpha asymmetry.")
    L = _cell(4, "Analysis: cognitive indices", what)
    groups = list(bp_spec.groups)
    ref = bp_spec.reference if bp_spec.reference not in ("recorded", "as recorded") else None
    L += [assign("BANDS = ", {k: list(v) for k, v in bp_spec.bands.items()}),
          f"WINDOW = ({lit(bp_spec.tmin)}, {lit(bp_spec.tmax)})",
          assign("REGIONS = ", rois),
          "",
          "def power(ep, channels=None):",
          '    """Band power per epoch (µV²), averaged over ``channels`` (all if None):',
          '    Welch PSD (segments of up to 2 s, 50 % overlap) summed over each band."""',
          "    seg = min(int(round((WINDOW[1] - WINDOW[0]) * ep.info['sfreq'])) + 1, "
          "int(2 * ep.info['sfreq']))",
          f"    psd = ep.compute_psd(method={lit(bp_spec.method)}, "
          "fmin=min(lo for lo, _ in BANDS.values()),",
          "                         fmax=min(max(hi for _, hi in BANDS.values()), "
          "ep.info['sfreq'] / 2),",
          "                         tmin=WINDOW[0], tmax=WINDOW[1], picks='eeg', exclude=(),",
          "                         n_fft=seg, n_per_seg=seg, n_overlap=seg // 2)",
          "    picks = [c for c in channels if c in ep.ch_names] if channels else None",
          "    p, f = psd.get_data(picks=picks, return_freqs=True)",
          "    return {b: p[:, :, (f >= lo) & (f < hi)].sum(-1).mean(-1) * np.diff(f).mean() "
          "* 1e12",
          "            for b, (lo, hi) in BANDS.items()}",
          "",
          "indices = {}",
          f"for label in {lit(groups)}:",
          "    ep = epochs[label]" + (f".copy().set_eeg_reference({lit(ref)})  # as for band power"
                                     if ref else ""),
          "    eng, fm, par = power(ep, REGIONS['engagement']), "
          "power(ep, REGIONS['frontal midline']), power(ep, REGIONS['parietal'])",
          "    left, right, every = power(ep, REGIONS['left frontal']), "
          "power(ep, REGIONS['right frontal']), power(ep)",
          "    frontal = power(ep, REGIONS['frontal midline'] + REGIONS['left frontal'] + "
          "REGIONS['right frontal'])",
          "    indices[label] = pd.DataFrame({",
          "        'engagement β/(α+θ)': eng['beta'] / (eng['alpha'] + eng['theta']),",
          "        'workload FMθ/Pα': fm['theta'] / par['alpha'],",
          "        'θ/α': every['theta'] / every['alpha'],",
          "        'θ/β frontal': frontal['theta'] / frontal['beta'],",
          "        'FAA ln α(F4)−ln α(F3)': np.log(right['alpha']) - np.log(left['alpha'])})",
          "    print(label, indices[label].mean().round(3).to_dict())"]
    if len(groups) == 2:
        L += ["for index in indices[" + repr(groups[0]) + "]:",
              f"    print(index, stats.ttest_ind(indices[{groups[0]!r}][index], "
              f"indices[{groups[1]!r}][index], equal_var=False, nan_policy='omit'))"]
    return L, {"scipy"}


# =========================================================================== script
_IMPORT = {"mne": "import mne", "numpy": "import numpy as np", "pandas": "import pandas as pd",
           "pyxdf": "import pyxdf", "json": "import json", "warnings": "import warnings",
           "mne_bids": "from mne_bids import get_bids_path_from_fname, read_raw_bids",
           "pymovements": "import pymovements as pm", "polars": "import polars as pl",
           "autoreject": "from autoreject import AutoReject", "scipy": "from scipy import stats",
           "ndimage": "from scipy.ndimage import uniform_filter1d"}
_PIP = {"mne": "mne", "numpy": "numpy", "pandas": "pandas", "pyxdf": "pyxdf",
        "mne_bids": "mne-bids", "pymovements": "pymovements", "polars": "polars",
        "autoreject": "autoreject", "scipy": "scipy", "ndimage": "scipy"}


def build_script(kind: str, recipe: dict, spec, extra=None, version: str = "") -> str:
    """The script for an analysis ``kind`` ("erp", "bandpower" or "indices").

    ``spec``: the analysis settings (ErpSpec / BandPowerSpec / (BandPowerSpec,
    IndicesSpec)); ``extra``: {"ch_names", "n_subjects"} (ERP) or the time-course band."""
    why = check_recipe(recipe)
    if why:
        raise ValueError(why)
    if kind not in TITLES:
        raise ValueError(f"Unknown analysis {kind!r}")
    if not version:
        from . import __version__ as version
    load, imp = _loading(recipe)
    gev, imp2 = _gaze_events(recipe)
    pre, imp3 = _preprocessing(recipe)
    epo, imp4 = _epoching(recipe)
    if kind == "erp":
        ana, imp5 = _erp(spec, extra)
    elif kind == "bandpower":
        ana, imp5 = _bandpower(spec, extra)
    else:
        ana, imp5 = _indices(*spec)
    imports = imp | imp2 | imp3 | imp4 | imp5
    if any("ica" in x and "picard" in x for x in pre):
        pip_extra = ["python-picard"]
    else:
        pip_extra = []
    order = ["json", "warnings", "mne", "numpy", "pandas", "polars", "pymovements", "pyxdf", "autoreject",
             "mne_bids", "scipy", "ndimage"]
    std = [_IMPORT[m] for m in order if m in imports and m in ("json", "warnings")]
    third = [_IMPORT[m] for m in order if m in imports and m not in ("json", "warnings")]
    pip = " ".join(sorted({_PIP[m] for m in imports if m in _PIP} | set(pip_extra)))
    head = [f'"""{TITLES[kind]} — exported from brAInStudio {version} on '
            f'{_dt.date.today().isoformat()}.',
            "",
            "The app's analysis with the same settings, from the data files to the result,",
            "in plain MNE code. Run it cell by cell (# %%) or as a whole. Results are close",
            "to the app's; the app's loading handles some more edge cases.",
            "",
            f"Needs: pip install {pip}",
            '"""']
    if any("Path(" in x for x in load):
        std.append("from pathlib import Path")
    body = head + [""] + std + ([""] if std else []) + third + ["", "mne.set_log_level('error')"]
    body += load + gev + pre + epo + ana
    return "\n".join(body) + "\n"
