"""The application state (the *model* of the GUI), without Qt.

A :class:`Session` holds the loaded recording, the gaze events and event table, and
the results of the EEG stages (preprocessing → epoching → analysis) with the settings
each was computed from, so an exported script repeats it. Every change goes through a
method that tells the subscribers what changed (``session.subscribe(fn)``; ``fn`` gets
one of ``recording``, ``gaze``, ``events``, ``preprocessing``, ``epoching``,
``analysis`` or ``state``). The computations themselves are the functions of
:mod:`brainstudio.core`; the GUI controllers run them and store the results here.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from . import core as C

STAGES = ("preprocessing", "epoching", "analysis")


# =========================================================================== loading
def find_bids_runs(root, keys: list[tuple]) -> list:
    """The BIDS runs of ``root`` with these entity keys, in this order."""
    runs = {r.key: r for r in C.scan_bids(root)}
    missing = [k for k in map(tuple, keys) if k not in runs]
    if missing:
        raise ValueError(f"Runs not found in {root}: {missing}")
    return [runs[tuple(k)] for k in keys]


def load_step(step: dict):
    """The Recording a loading step describes: ``load`` (one file), ``load_many``
    (files joined) or ``load_bids`` (BIDS runs joined)."""
    from . import io as bio
    kind = step["kind"]
    if kind == "load":
        return bio.load(step["path"], step["screen_size"], roles=step.get("roles"),
                        options=step.get("options"))
    if kind == "load_many":
        return bio.load_many(step["paths"], step["screen_size"], step.get("roles_by_name"),
                             step.get("options_by_name"))
    if kind == "load_bids":
        return bio.load_bids_runs(step["root"], find_bids_runs(step["root"], step["runs"]),
                                  step.get("screen_size"))
    raise ValueError(f"Unknown loading step {kind!r}")


def recording_info(rec, step: dict) -> dict:
    """What an exported script needs to know about a recording as it was loaded:
    channel names and types, the XDF stream table, the entities of each file, and the
    files and gaze columns of BIDS runs."""
    out = {"channels_in": list(rec.raw.ch_names) if rec.raw is not None else [],
           "types_in": list(rec.raw.get_channel_types()) if rec.raw is not None else [],
           "streams": [dict(s) for s in (rec.streams or [])]}
    if step["kind"] == "load_many":
        out["entities"] = {p: {k: v for k, v in C.entities_from_path(p).items() if v}
                           for p in step["paths"]}
    if step["kind"] == "load_bids":
        root = Path(step["root"])
        runs = []
        for r in find_bids_runs(root, step["runs"]):
            item = {k: v for k, v in (("subject", r.entities.get("subject")),
                                      ("session", r.entities.get("session")),
                                      ("run", r.entities.get("run"))) if v}
            item["eeg"] = str(Path(r.eeg).relative_to(root))
            if r.gaze:
                eye = Path(r.gaze[0])
                meta = json.loads(eye.with_name(eye.name.replace(".tsv.gz", ".json")
                                                .replace(".tsv", ".json"))
                                  .read_text(encoding="utf-8"))
                cols = meta.get("Columns", [])
                m = C.gaze_columns([c.lower() for c in cols], len(cols))
                item["eye"] = str(eye.relative_to(root))
                for role in ("x", "y", "pupil"):
                    if m.get(role):
                        item[role] = cols[m[role][0]]
            runs.append(item)
        out["runs"] = runs
    return out


def detection_params(gaze, preset: str | None, custom: C.DetectionParams | None = None):
    """Gaze detection settings: the preset's (``auto``: chosen from the data) or
    ``custom``."""
    if preset == "auto":
        from .io import gaze_table
        return C.PRESETS[C.suggest_preset(gaze_table(gaze))][1]
    if preset in C.PRESETS:
        return C.PRESETS[preset][1]
    return custom or C.DetectionParams()


def detect_gaze_events(gaze, params: C.DetectionParams):
    from .io import gaze_table
    return C.detect_gaze_events(gaze_table(gaze), params)


def trigger_events(raw, channel: str, mask: int | None, mode: str,
                   names: dict[int, str]) -> pd.DataFrame:
    """The codes of a trigger channel as events named ``names[code]`` (other codes are
    ignored)."""
    found = C.trigger_codes(raw, channel, mask, mode)
    sel = found[found["code"].isin(names)]
    return pd.DataFrame({"onset": sel["onset"].to_numpy(float), "duration": 0.0,
                         "description": [f"{names[c]}|trigger_code={c}|"
                                         f"trigger_channel={channel}" for c in sel["code"]]})


def with_trigger_events(events: pd.DataFrame, ev: pd.DataFrame | None) -> pd.DataFrame:
    """``events`` with earlier trigger events replaced by ``ev`` (None: removed)."""
    keep = ~events["description"].astype(str).str.contains("|trigger_code=", regex=False)
    parts = [events[keep]] + ([ev] if ev is not None and len(ev) else [])
    return pd.concat(parts, ignore_index=True).sort_values("onset", ignore_index=True)


# =========================================================================== the session
class Session:
    def __init__(self):
        self._subscribers = []
        self.rec = None
        self.aois: dict = {}  # areas of interest on the screen (kept across recordings)
        self.clear_recording()

    # ------------------------------------------------------------------ observers
    def subscribe(self, fn):
        """Call ``fn(what)`` after every change."""
        self._subscribers.append(fn)

    def notify(self, what: str):
        for fn in list(self._subscribers):
            fn(what)

    # ------------------------------------------------------------------ recording
    def clear_recording(self):
        #: how the recording was loaded and edited, in order (for exported scripts)
        self.load_steps: list[dict] = []
        self.raw_gaze = None        # the gaze as loaded (before the offset)
        self.gaze_offset = 0.0      # manual sync correction (s), applied to the gaze
        self.gaze_params = None     # detection settings of the current gaze events
        self.gaze_events = None
        self.event_table = C.event_table(None)
        self.reset_stages()

    def set_recording(self, rec, step: dict | None = None):
        """A newly loaded recording; ``step`` is how it was loaded. Gaze events detected
        while loading (``rec.gaze_events`` / ``rec.detected_with``) are kept."""
        self.rec = rec
        self.clear_recording()
        self.raw_gaze = rec.gaze
        if step is not None:
            try:  # what an exported script needs to know about the files
                step["recording"] = recording_info(rec, step)
            except Exception:  # noqa: BLE001  (export will say it cannot)
                step["recording"] = {}
            self.load_steps = [step]
        if getattr(rec, "detected_with", None) is not None:
            self.gaze_events, self.gaze_params = rec.gaze_events, rec.detected_with
        self.event_table = C.event_table(rec.all_events(), self.gaze_events,
                                         C.make_aois(self.aois), rec.segments)
        self.notify("recording")
        self.notify("state")

    def load_recipe(self) -> dict:
        return {"steps": list(self.load_steps),
                "gaze": {"params": self.gaze_params, "aois": dict(self.aois)}}

    def set_gaze(self, gaze, step: dict | None = None):
        """Gaze attached from another file."""
        self.rec.gaze = self.raw_gaze = gaze
        self.gaze_offset = 0.0
        if step is not None:
            self.load_steps.append(step)
        self.notify("gaze")

    def set_gaze_offset(self, seconds: float) -> bool:
        """Shift the gaze (as loaded) by ``seconds``; one offset step replaces an
        earlier one. False when nothing changed."""
        if self.rec is None or self.raw_gaze is None or seconds == self.gaze_offset:
            return False
        self.gaze_offset = seconds
        self.rec.gaze = self.raw_gaze.shifted(seconds)
        self.load_steps = [s for s in self.load_steps if s["kind"] != "gaze_offset"]
        if seconds:
            self.load_steps.append({"kind": "gaze_offset", "seconds": seconds})
        self.notify("gaze")
        return True

    def set_gaze_events(self, ev, params, rebuild: bool = True):
        self.gaze_events, self.gaze_params = ev, params
        self.rec.gaze_events, self.rec.detected_with = ev, None
        self.notify("gaze_events")
        if rebuild:
            self.rebuild_event_table()

    def set_aois(self, specs: dict):
        self.aois = dict(specs)
        if self.rec is not None:
            self.rebuild_event_table()

    def rebuild_event_table(self):
        rec = self.rec
        self.event_table = C.event_table(rec.all_events() if rec is not None else None,
                                         self.gaze_events, C.make_aois(self.aois),
                                         rec.segments if rec is not None else None)
        self.notify("events")
        self.mark_stale("epoching")

    def set_trigger_events(self, channel: str, mask, mode: str, names: dict[int, str],
                           as_stim: bool = True) -> bool:
        """Events from a trigger channel (replacing earlier ones). True when the channel
        became a stim channel, so the EEG channels changed."""
        raw = self.rec.raw
        self.rec.events = with_trigger_events(self.rec.events,
                                              trigger_events(raw, channel, mask, mode, names))
        self.load_steps.append({"kind": "trigger_events", "channel": channel, "mask": mask,
                                "mode": mode, "names": {int(k): v for k, v in names.items()},
                                "as_stim": as_stim})
        retyped = as_stim and raw.get_channel_types([channel])[0] != "stim"
        if retyped:
            raw.set_channel_types({channel: "stim"}, verbose="error")
            self.params.bads = [b for b in self.params.bads if b != channel]
            self.invalidate_ica()
        self.rebuild_event_table()
        if retyped:
            self.notify("channels")
            self.mark_stale("preprocessing")
        return retyped

    def remove_trigger_events(self):
        self.rec.events = with_trigger_events(self.rec.events, None)
        self.load_steps.append({"kind": "remove_trigger_events"})
        self.rebuild_event_table()

    def set_montage(self, source: str | None, names: list[str] | None = None,
                    bads=None) -> dict:
        """Rename the channels (in recording order), set their types and apply a montage:
        a built-in layout (``gtec_64``), an MNE montage name or a file. The bad channels
        (``bads``, default the current ones) keep their new names; ICA must be refitted."""
        old = list(self.rec.raw.ch_names)
        montage = C.montage_from_source(source) if source else None
        report = C.apply_montage(self.rec.raw, montage, names)
        self.load_steps.append({"kind": "montage", "source": source,
                                "names": list(names) if names else None})
        rename = dict(zip(old, self.rec.raw.ch_names))
        self.params.bads = [rename.get(b, b) for b in (self.params.bads if bads is None
                                                         else bads)]
        self.invalidate_ica()
        self.notify("channels")
        self.mark_stale("preprocessing")
        return report

    # ------------------------------------------------------------------ EEG stages
    @property
    def raw(self):
        return self.rec.raw if self.rec is not None else None

    def reset_stages(self):
        self.params = C.PreprocParams(reference=None)
        self.clean = None
        self.bad_stats: pd.DataFrame | None = None
        self.ica = self.ica_fit_raw = self.ica_scores = self.ica_key = None
        self.ica_sources = None
        self.psd_before = self.psd_after = None
        self.epoch_settings = C.EpochSettings()
        self.epochs: dict = {}
        self.epoch_summary: pd.DataFrame | None = None
        self.group_desc: dict[str, str] = {}
        self.state = {s: "pending" for s in STAGES}
        # what each output was computed from, so an exported script reproduces it
        self.ica_spec: dict | None = None      # fit_ica arguments of the current ICA
        self.pre_recipe: dict | None = None    # loading + ICA + parameters of ``clean``
        self.epoch_recipe: dict | None = None  # loading + preprocessing + groups of epochs

    def eeg_input(self):
        """Data for epoching: the preprocessed recording (raw if not run yet)."""
        return self.clean if self.clean is not None else self.raw

    def set_clean(self, clean, params, psd_before=None, psd_after=None, recipe=None):
        self.clean, self.params, self.pre_recipe = clean, params, recipe
        self.psd_before, self.psd_after = psd_before, psd_after
        self.state["preprocessing"] = "ok"
        self.mark_stale("epoching")
        self.notify("preprocessing")

    def set_epochs(self, epochs, summary, settings, group_desc, recipe=None):
        self.epochs, self.epoch_summary, self.epoch_recipe = epochs, summary, recipe
        self.epoch_settings, self.group_desc = settings, group_desc
        self.state["epoching"] = "ok"
        self.mark_stale("analysis")
        self.notify("epoching")

    def set_analysis_done(self):
        self.state["analysis"] = "ok"
        self.notify("state")

    def set_ica(self, ica, fit_raw, scores, key, spec):
        self.ica, self.ica_fit_raw, self.ica_scores = ica, fit_raw, scores
        self.ica_key, self.ica_spec, self.ica_sources = key, spec, None

    def invalidate_ica(self):
        self.ica = self.ica_fit_raw = self.ica_scores = self.ica_key = None
        self.ica_spec = self.ica_sources = None

    def mark_stale(self, stage: str):
        """Settings of ``stage`` changed: it and every later stage are out of date."""
        for s in STAGES[STAGES.index(stage):]:
            if self.state[s] == "ok":
                self.state[s] = "stale"
        self.notify("state")

    # ------------------------------------------------------------------ recipes
    def snapshot_pre_recipe(self, params, ica) -> dict:
        return {"load": copy.deepcopy(self.load_recipe()),
                "ica": copy.deepcopy(self.ica_spec) if ica is not None else None,
                "params": copy.deepcopy(params)}

    def snapshot_epoch_recipe(self, raw, specs, settings) -> dict:
        """Everything the epochs are built from. ``data`` / ``table`` keep the objects
        themselves, to tell later whether an analysis still used the same ones."""
        from_clean = self.clean is not None and raw is self.clean
        return {"load": copy.deepcopy(self.load_recipe()),
                "pre": copy.deepcopy(self.pre_recipe) if from_clean else None,
                "groups": copy.deepcopy(specs), "settings": copy.deepcopy(settings),
                "data": raw, "table": self.event_table, "gaze_events": self.gaze_events}
