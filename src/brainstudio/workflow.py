"""The steps of the app as plain functions, from loading to each analysis.

The app runs exactly these functions, and the scripts it exports
(:mod:`brainstudio.script`) call them with the same settings — so a script gives the
same results as the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
import pandas as pd

from . import bandpower as BP
from . import eeg as E
from . import events as EV
from . import gaze as G
from . import stats as ST

# =========================================================================== loading


def find_bids_runs(root, keys: list[tuple]) -> list:
    """The BIDS runs of ``root`` with these entity keys, in this order."""
    from . import io as bio
    runs = {r.key: r for r in bio.scan_bids(root)}
    missing = [k for k in map(tuple, keys) if k not in runs]
    if missing:
        raise ValueError(f"Runs not found in {root}: {missing}")
    return [runs[tuple(k)] for k in keys]


def recording_info(rec, step: dict) -> dict:
    """What an exported script needs to know about a recording as it was loaded:
    channel names and types, the XDF stream table, the entities of each file, and the
    files and gaze columns of BIDS runs."""
    from pathlib import Path

    from .io.multi import entities_from_path
    out = {"channels_in": list(rec.raw.ch_names) if rec.raw is not None else [],
           "types_in": list(rec.raw.get_channel_types()) if rec.raw is not None else [],
           "streams": [dict(s) for s in (rec.streams or [])]}
    if step["kind"] == "load_many":
        out["entities"] = {p: {k: v for k, v in entities_from_path(p).items() if v}
                           for p in step["paths"]}
    if step["kind"] == "load_bids":
        from .io.xdf import gaze_columns
        root = Path(step["root"])
        runs = []
        for r in find_bids_runs(root, step["runs"]):
            item = {k: v for k, v in (("subject", r.entities.get("subject")),
                                      ("session", r.entities.get("session")),
                                      ("run", r.entities.get("run"))) if v}
            item["eeg"] = str(Path(r.eeg).relative_to(root))
            if r.gaze:
                import json
                eye = Path(r.gaze[0])
                meta = json.loads(eye.with_name(eye.name.replace(".tsv.gz", ".json")
                                                .replace(".tsv", ".json"))
                                  .read_text(encoding="utf-8"))
                cols = meta.get("Columns", [])
                m = gaze_columns([c.lower() for c in cols], len(cols))
                item["eye"] = str(eye.relative_to(root))
                for role in ("x", "y", "pupil"):
                    if m.get(role):
                        item[role] = cols[m[role][0]]
            runs.append(item)
        out["runs"] = runs
    return out


def set_montage(rec, source: str | None, names: list[str] | None = None) -> dict:
    """Rename the channels (in recording order), set their types and apply a montage:
    a built-in layout (``gtec_64``), an MNE montage name or a file."""
    montage = E.montage_from_source(source) if source else None
    return E.apply_montage(rec.raw, montage, names)


def add_trigger_events(rec, channel: str, mask: int | None, mode: str,
                       names: dict[int, str], as_stim: bool = True) -> int:
    """Turn the codes of a trigger channel into events named ``names[code]`` (codes not
    in ``names`` are ignored), replacing earlier trigger events. ``as_stim`` marks the
    channel as a stim channel, so it is no longer treated as EEG."""
    found = E.trigger_codes(rec.raw, channel, mask, mode)
    sel = found[found["code"].isin(names)]
    ev = pd.DataFrame({"onset": sel["onset"].to_numpy(float), "duration": 0.0,
                       "description": [f"{names[c]}|trigger_code={c}|trigger_channel={channel}"
                                       for c in sel["code"]]})
    replace_trigger_events(rec, ev)
    if as_stim and rec.raw.get_channel_types([channel])[0] != "stim":
        rec.raw.set_channel_types({channel: "stim"}, verbose="error")
    return len(ev)


def replace_trigger_events(rec, ev: pd.DataFrame | None) -> None:
    old = rec.events
    keep = ~old["description"].astype(str).str.contains("|trigger_code=", regex=False)
    parts = [old[keep]] + ([ev] if ev is not None and len(ev) else [])
    rec.events = pd.concat(parts, ignore_index=True).sort_values("onset", ignore_index=True)



# =========================================================================== epoching
@dataclass
class GroupSpec:
    """An event group: the event names ticked, the filters and the recordings kept
    (``recordings``: column → values kept, e.g. ``{"subject": ["01", "02"]}``)."""
    label: str
    names: list[str]
    text: str = ""
    query: str = ""
    conditions: list = field(default_factory=list)
    recordings: dict = field(default_factory=dict)


def select_events(table: pd.DataFrame, spec: GroupSpec) -> pd.DataFrame:
    """Rows of the event table in a group (raises ValueError for invalid filters)."""
    df = EV.filter_events(table, spec.text, spec.query, set(spec.names),
                          [tuple(c) for c in spec.conditions])
    for col, keep in spec.recordings.items():
        if col in df:
            df = df[df[col].fillna("").astype(str).isin(keep)]
    return df


# =========================================================================== ERP
@dataclass
class ErpSpec:
    groups: list[str]
    component: E.Component
    baseline: tuple[float, float] | None = (-0.1, 0.0)
    reference: str = "recorded"   # average | mastoids | recorded | channel names


def run_erp(epochs: dict[str, mne.BaseEpochs], spec: ErpSpec) -> dict:
    """ERP component analysis. With several subjects: one Evoked per subject and group,
    grand average of those, subjects as observations (:func:`eeg.group_component_analysis`);
    otherwise trials are the observations (:func:`eeg.component_analysis`)."""
    groups = {g: epochs[g] for g in spec.groups}
    first = next(iter(groups.values()))
    ref, ref_label = E.resolve_erp_reference(spec.reference, list(first.ch_names))
    comp, bl = spec.component, spec.baseline
    notes = []
    eps = {g: E.reference_epochs(ep, ref) for g, ep in groups.items()}
    eps = {g: (ep.copy().apply_baseline(bl, verbose="error") if bl else ep)
           for g, ep in eps.items()}
    evokeds = {}
    for g, ep in eps.items():
        ev = ep.average()
        ev.comment = g
        evokeds[g] = ev
    summary, per = E.component_analysis(eps, comp)
    subj = {g: E.epoch_subjects(ep) for g, ep in eps.items()}
    if any(k is not None for k in subj.values()):
        per.insert(1, "subject", np.concatenate(
            [k if k is not None else np.full(len(ep), "", object)
             for k, ep in zip(subj.values(), eps.values())]))
    group = None
    if len(E.subjects_in(eps)) >= 2:
        group = E.group_component_analysis(eps, comp)
        unknown = sum(int((k == "").sum()) for k in subj.values() if k is not None)
        if unknown:
            notes.append(f"⚠ {unknown} epochs without a subject are left out of the "
                         "subject averages")
        return dict(eps=eps, evokeds=group["grand"], summary=group["summary"],
                    per_epoch=per, diff=group["grand_diff"], roi=list(comp.channels),
                    comp=comp, baseline=bl, notes=notes, reference=ref_label, group=group,
                    spec=spec)
    diff = None
    if len(eps) == 2:
        (la, a), (lb, b) = eps.items()
        diff = mne.combine_evoked([evokeds[la], evokeds[lb]], weights=[1, -1])
        diff.comment = f"{la} − {lb}"
        ov = len(set(a.events[:, 0]) & set(b.events[:, 0]))
        if ov:
            notes.append(f"⚠ {ov} events are in both groups (tests assume independence)")
    return dict(eps=eps, evokeds=evokeds, summary=summary, per_epoch=per, diff=diff,
                roi=list(comp.channels), comp=comp, baseline=bl, notes=notes,
                reference=ref_label, group=None, spec=spec)


# =========================================================================== band power
@dataclass
class BaselineSpec:
    """Baseline after a baseline event: ``t0``…``t1`` s after each ``event`` matching
    ``query``; ``mode`` "all" or "nearest" (see :class:`bandpower.BandBaseline`)."""
    event: str
    t0: float
    t1: float
    mode: str = "all"
    query: str = ""


def band_baseline(table: pd.DataFrame, spec: BaselineSpec | None) -> BP.BandBaseline | None:
    if spec is None:
        return None
    sel = table[table["name"].astype(str) == spec.event]
    if spec.query.strip():
        sel = EV.filter_events(sel, query=spec.query)
    if not len(sel):
        raise ValueError(f"No “{spec.event}” events match the baseline filter")
    if spec.t1 <= spec.t0:
        raise ValueError("The baseline end must be after its start")
    sel = sel.sort_values("onset")
    subjects = sel["subject"].to_numpy() if "subject" in sel else None
    return BP.BandBaseline(sel["onset"].to_numpy(float), spec.t0, spec.t1, spec.mode,
                           spec.event, subjects)


@dataclass
class BandPowerSpec:
    groups: list[str]
    bands: dict
    tmin: float
    tmax: float
    method: str = "welch"
    roi: list[str] = field(default_factory=list)   # empty: every channel
    reference: str = "average"
    baseline: BaselineSpec | None = None
    norm: str = "dB"                                # used only with a baseline


def run_bandpower(epochs: dict[str, mne.BaseEpochs], raw: mne.io.BaseRaw,
                  table: pd.DataFrame, spec: BandPowerSpec) -> dict:
    """Band power per epoch (``Epochs.compute_psd``), normalised per subject against the
    baseline after a baseline event, on the ROI mean; group comparisons per band."""
    groups = {g: epochs[g] for g in spec.groups}
    first = next(iter(groups.values()))
    bands = dict(spec.bands)
    base = band_baseline(table, spec.baseline)
    roi = list(spec.roi) or list(first.ch_names)
    s = BP.BandSettings(bands, spec.tmin, spec.tmax, spec.method)
    eeg_ref, ref_label = E.resolve_erp_reference(spec.reference, list(first.ch_names))
    mode = spec.norm if base is not None else "absolute"
    names = list(bands)
    notes = []
    out = {"bp": {}, "ref": {}, "sd": {}, "norm": {}, "per": [], "bands": bands,
           "mode": mode, "roi": roi, "base": base, "settings": s,
           "reference": eeg_ref, "reference_label": ref_label, "spec": spec}
    eps = {g: E.reference_epochs(ep, eeg_ref) for g, ep in groups.items()}
    ch = list(first.ch_names)
    seg = (min(s.tmax, first.tmax) - max(s.tmin, first.tmin))
    base_bp = owner = None
    if base is not None:
        base_bp, owner = BP.baseline_band_power(raw, base, s, ch, seg, ref=eeg_ref)
        notes.append(f"baseline: {len(base_bp)} segments of {seg:.2f} s from "
                     f"{len(np.unique(owner))} “{base.label}” event(s), "
                     f"{base.t0:g}…{base.t1:g} s after")
        subj_all = sorted({str(x) for ep in eps.values() if ep.metadata is not None
                           and "subject" in ep.metadata
                           for x in ep.metadata["subject"].fillna("")} - {""})
        if subj_all:
            notes.append(f"normalised per subject ({len(subj_all)}: "
                         f"{', '.join(subj_all[:8])}{' …' if len(subj_all) > 8 else ''})")
            missing = BP.subjects_without_baseline(base, subj_all)
            if missing:
                notes.append("⚠ no baseline event for subject(s) "
                             f"{', '.join(missing)}: their epochs are left out "
                             "of the normalised values")
    ridx = BP.resolve_roi(roi, ch)
    for g, ep in eps.items():
        bp = BP.band_power(ep, s)
        out["bp"][g] = bp
        if base_bp is not None:
            subj = (ep.metadata["subject"].to_numpy()
                    if "subject" in ep.metadata else None)
            ref, sd = BP.reference_for(ep.metadata["onset"].to_numpy(float), base,
                                       base_bp, owner, subj)
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
    out["table"] = ST.comparison_table(per, names, list(eps))
    out["notes"] = notes
    return out


def sem(x: np.ndarray, axis=0) -> np.ndarray:
    """``scipy.stats.sem`` ignoring NaNs; NaN where fewer than two values (no warnings)."""
    import warnings

    from scipy import stats
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        return np.asarray(stats.sem(np.asarray(x, float), axis=axis, nan_policy="omit"),
                          float)


def run_timecourse(bp: dict, epochs: dict[str, mne.BaseEpochs], raw: mne.io.BaseRaw,
                   band: str) -> dict:
    """Morlet power time course of one band (ERD/ERS) on the band-power result's ROI,
    reference and baseline (per subject)."""
    rng = bp["bands"][band]
    groups = {g: epochs[g] for g in bp["bp"] if g in epochs}
    base, mode, roi = bp["base"], bp["mode"], bp["roi"]
    out = {"band": band, "mode": mode, "curves": {}}
    levels = None
    if base is not None and mode != "absolute":
        levels = BP.baseline_timecourse_levels(   # per subject
            raw, base, rng, roi, ref=bp["reference"],
            ch_names=list(next(iter(groups.values())).ch_names))
    for g, ep in groups.items():
        t, pw = BP.band_timecourse(E.reference_epochs(ep, bp["reference"]), rng, roi)
        if levels is not None:
            subj = BP._subject_keys(ep.metadata["subject"].to_numpy()
                                    if ep.metadata is not None
                                    and "subject" in ep.metadata else None, len(pw))
            if set(levels) == {""}:
                subj[:] = ""
            lvl = np.array([levels.get(x, (np.nan, np.nan))[0] for x in subj])[:, None]
            sd = np.array([levels.get(x, (np.nan, np.nan))[1] for x in subj])[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                pw = {"dB": 10 * np.log10(pw / lvl), "% change": 100 * (pw - lvl) / lvl,
                      "ratio": pw / lvl, "z": (pw - lvl) / sd}[mode]
        out["curves"][g] = (t, np.nanmean(pw, 0), sem(pw))
    return out


# =========================================================================== indices
@dataclass
class IndicesSpec:
    rois: dict
    measures: list[str]


def run_indices(bp: dict, epochs: dict[str, mne.BaseEpochs], gaze, gaze_events,
                spec: IndicesSpec) -> dict:
    """Cognitive indices per epoch from the band-power result, plus eye metrics."""
    s = bp["settings"]
    per, notes = [], []
    for g, b in bp["bp"].items():
        ep = epochs[g]
        df, n = BP.eeg_indices(b, list(ep.ch_names), bp["bands"], spec.rois, bp["ref"][g])
        notes += [x for x in n if x not in notes]
        on = ep.metadata["onset"].to_numpy(float)
        gm = BP.gaze_metrics(on, max(s.tmin, ep.tmin), min(s.tmax, ep.tmax), gaze,
                             gaze_events)
        df = pd.concat([df, gm], axis=1)
        df.insert(0, "group", g)
        df.insert(1, "onset", on)
        per.append(df)
    per = pd.concat(per, ignore_index=True)
    cols = [c for c in spec.measures if c in per.columns and per[c].notna().any()]
    missing = [c for c in spec.measures if c not in cols]
    if missing:
        notes.append("not available: " + ", ".join(missing))
    table = ST.comparison_table(per, cols, list(bp["bp"]))
    return dict(per_epoch=per, table=table, measures=cols, notes=notes, spec=spec)
