"""Loading, for the app: :mod:`brainstudio.core` reads the files; this wraps the result in
the :class:`~brainstudio.model.Recording` the app works with."""

from __future__ import annotations

from pathlib import Path

from .. import core as C
from ..core import BidsRun, describe_streams, entities_from_path, scan_bids  # noqa: F401
from ..model import GazeData, Recording


def gaze_data(g) -> GazeData | None:
    """A core gaze table as GazeData (the tracker's own channels in ``extra``)."""
    if g is None:
        return None
    return GazeData(g["t"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy(),
                    g["pupil"].to_numpy() if "pupil" in g else None, g.attrs.get("sfreq"),
                    tuple(g.attrs.get("screen", (1920, 1080))), g.attrs.get("label", "gaze"),
                    {c[3:]: g[c].to_numpy() for c in g.columns if c.startswith("ch_")})


def recording(raw, events, gaze, info, source=None, annotate=False) -> Recording:
    rec = Recording(raw=raw, gaze=gaze_data(gaze), events=events.reset_index(drop=True),
                    source=Path(source) if source else None)
    rec.streams = info.get("streams", [])
    rec.metadata = info.get("metadata", {})
    rec.description = "\n".join(x for x in [info.get("description", "")] + info.get("notes", [])
                                if x)
    if annotate and raw is not None and len(events):  # XDF markers inside the EEG
        import mne
        inside = (events.onset >= 0) & (events.onset <= raw.times[-1])
        raw.set_annotations(raw.annotations + mne.Annotations(
            events.onset[inside].to_numpy(), events.duration[inside].to_numpy(),
            events.description[inside].to_numpy()))
        rec.events = events[~inside].reset_index(drop=True)
    return rec


def _read(path, screen, roles=None, options=None):
    if Path(path).name.lower().endswith((".xdf", ".xdfz")):
        return C.read_xdf(path, screen, roles, options)
    return C.read_file(path, screen)


def load(path, screen_size=(1920, 1080), roles=None, options=None) -> Recording:
    """One file: XDF, any file MNE reads, or an EyeLink .asc."""
    raw, events, gaze, info = _read(path, screen_size, roles, options)
    if raw is not None:
        changed = C.type_channels_by_name(raw)
        if changed:
            info.setdefault("notes", []).append("channel types from names: " + ", ".join(
                f"{c}→{t}" for c, t in changed.items()))
    rec = recording(raw, events, gaze, info, path,
                    annotate=Path(path).name.lower().endswith((".xdf", ".xdfz")))
    rec.sources = [Path(path)]
    rec.segments = [{"start": 0.0, "end": float(rec.duration),
                     **{k: v for k, v in entities_from_path(path).items() if v}}]
    return rec


def _joined(recs: list[Recording], first: Recording) -> Recording:
    """Recordings joined end to end (:func:`brainstudio.core.join`)."""
    if len(recs) == 1:
        return recs[0]
    parts = []
    for r in recs:
        events = r.all_events() if r.raw is not None else r.events
        if r.raw is not None:
            r.raw.set_annotations(r.raw.annotations[
                C.is_data_annotation(r.raw.annotations.to_data_frame()["description"])
                .to_numpy()] if len(r.raw.annotations) else r.raw.annotations)
            events = events[~C.is_data_annotation(events["description"])]
        parts.append((r.raw, events, gaze_table(r.gaze), {"segments": r.segments}))
    raw, events, gaze, segments, notes = C.join(parts)
    out = recording(raw, events, gaze, {"streams": first.streams}, first.source)
    out.segments, out.sources = segments, [r.source for r in recs]
    if any(r.metadata for r in recs):  # first file's info; bad channels from all files
        out.metadata = dict(recs[0].metadata)
        out.metadata["bad_channels"] = {c: why for r in recs
                                        for c, why in r.metadata.get("bad_channels", {}).items()}
    out.description = "\n".join(
        [f"{len(recs)} files combined" + (": " + "; ".join(notes) if notes else "")]
        + [f"[{s.get('file', r.source)} @ {s['start']:.1f} s] {r.description}"
           for s, r in zip(segments, recs)])
    return out


def gaze_table(g: GazeData | None):
    if g is None:
        return None
    import pandas as pd
    t = pd.DataFrame({"t": g.times, "x": g.x, "y": g.y})
    if g.pupil is not None:
        t["pupil"] = g.pupil
    for k, v in g.extra.items():
        t["ch_" + k] = v
    t.attrs.update(sfreq=g.sfreq, screen=tuple(g.screen_size), label=g.label)
    return t


def load_many(paths, screen_size=(1920, 1080), roles_by_name=None,
              options_by_name=None) -> Recording:
    """Several files joined; for XDF, roles and options are given per stream name."""
    recs = []
    for p in map(Path, paths):
        roles = options = None
        if p.name.lower().endswith((".xdf", ".xdfz")) and (roles_by_name or options_by_name):
            keys = {s["name"]: str(s["stream_id"]) for s in
                    describe_streams(p).to_dict("records")}
            roles = {keys[n]: r for n, r in (roles_by_name or {}).items() if n in keys} or None
            options = {keys[n]: o for n, o in (options_by_name or {}).items()
                       if n in keys} or None
        recs.append(load(p, screen_size, roles, options))
    return _joined(recs, recs[0])


def load_bids_run(root, run, screen_size=None) -> Recording:
    raw, events, gaze, info = C.read_bids_run(root, run, screen_size)
    return recording(raw, events, gaze, info, run.eeg or (run.gaze[0] if run.gaze else root))


def load_bids_runs(root, runs, screen_size=None) -> Recording:
    """BIDS runs joined, each keeping its subject/session/task/acq/run/file."""
    recs = []
    for run in runs:
        r = load_bids_run(root, run, screen_size)
        ent = {k: run.entities.get(e) for k, e in (("subject", "subject"), ("session", "session"),
                                                   ("bids_task", "task"), ("acq", "acquisition"),
                                                   ("run", "run"))}
        ent["file"] = Path(run.eeg or (run.gaze[0] if run.gaze else root)).name
        r.segments = [{"start": 0.0, "end": float(r.duration),
                       **{k: v for k, v in ent.items() if v not in (None, "")}}]
        recs.append(r)
    return _joined(recs, recs[0])


def load_eyetracking_file(path, screen_size=(1920, 1080)) -> GazeData:
    """Gaze from an EyeLink .asc, a BIDS eye-tracking TSV or an XDF file."""
    name = Path(path).name.lower()
    if name.endswith(".asc") or name.endswith((".xdf", ".xdfz")):
        g = _read(path, screen_size)[2]
    elif ".tsv" in name:
        g = C.read_bids_eyetrack(path, screen=screen_size)[0]
    else:
        raise ValueError(f"Unsupported eye-tracking file: {Path(path).name}")
    if g is None:
        raise ValueError("No gaze in this file")
    return gaze_data(g)


def load_xdf(path, screen_size=(1920, 1080), roles=None, options=None) -> Recording:
    """An XDF file (see :func:`brainstudio.core.read_xdf`)."""
    return load(path, screen_size, roles, options)


def concatenate(recs: list[Recording]) -> Recording:
    return _joined(recs, recs[0])
