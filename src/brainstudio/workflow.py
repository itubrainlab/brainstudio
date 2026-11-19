"""Steps on a loaded Recording (to become the session controller); the analyses are in
:mod:`brainstudio.core`."""

from __future__ import annotations

import pandas as pd

from . import core as C
from .core import (  # noqa: F401
    BandPowerSpec,
    BaselineSpec,
    ErpSpec,
    GroupSpec,
    IndicesSpec,
    band_baseline,
    run_bandpower,
    run_erp,
    run_indices,
    run_timecourse,
    select_events,
    sem)


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
    montage = C.montage_from_source(source) if source else None
    return C.apply_montage(rec.raw, montage, names)


def add_trigger_events(rec, channel: str, mask: int | None, mode: str,
                       names: dict[int, str], as_stim: bool = True) -> int:
    """Turn the codes of a trigger channel into events named ``names[code]`` (codes not
    in ``names`` are ignored), replacing earlier trigger events. ``as_stim`` marks the
    channel as a stim channel, so it is no longer treated as EEG."""
    found = C.trigger_codes(rec.raw, channel, mask, mode)
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
