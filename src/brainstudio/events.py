"""A unified, filterable table of events (markers + gaze events).

Every row has ``onset`` (s, recording time base), ``duration``, ``name``,
``kind`` (marker / fixation / saccade / blink) and ``description``. Extra
columns come from

* structured markers ``NAME|key=value|key=value`` → ``name`` = NAME plus
  one column per key (numeric where possible),
* gaze events → duration, x, y, amplitude, peak_velocity, ``aoi``,
* context: each event inherits ``phase`` (the innermost open
  ``<PHASE>_START`` … ``<PHASE>_END`` span) and that span's attributes,
  so e.g. fixations can be filtered by ``cond == "AI"``.

Filtering uses :meth:`pandas.DataFrame.query` expressions.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .model import GazeEvents, Recording

BASE_COLUMNS = ["onset", "duration", "name", "kind", "description"]
#: names of detected gaze events; markers with these names get a "_marker" suffix
GAZE_EVENT_NAMES = ("fixation", "saccade", "blink")
# "<PHASE>_START" … "<PHASE>_END"; a separator is required, so names that merely
# end in "on"/"off" (e.g. "fixation") are not phase markers
_START = re.compile(r"^(.+?)[_\- ](START|BEGIN|ON)$", re.I)
_END = re.compile(r"^(.+?)[_\- ](END|STOP|OFF)$", re.I)


def _safe_key(k: str) -> str:
    k = re.sub(r"\W+", "_", k.strip()).strip("_").lower() or "attr"
    if k[0].isdigit():
        k = "a_" + k
    if k in BASE_COLUMNS or k in ("phase", "aoi", "x", "y"):
        k = "attr_" + k
    return k


def parse_description(desc: str) -> tuple[str, dict]:
    """``"NAME|k=v|k2=v2"`` → ("NAME", {k: v, k2: v2}). Other strings → (desc, {})."""
    parts = str(desc).split("|")
    attrs = {}
    rest = []
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            attrs[_safe_key(k)] = v.strip()
        elif p.strip():
            rest.append(p.strip())
    if rest:
        attrs["extra"] = "|".join(rest)
    return parts[0].strip(), attrs


def _to_numeric_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            conv = pd.to_numeric(df[c], errors="coerce")
            # convert only if every non-empty value is numeric
            nonempty = df[c].notna() & (df[c].astype(str) != "")
            if nonempty.any() and conv[nonempty].notna().all():
                df[c] = conv
    return df


def _phase_spans(markers: pd.DataFrame) -> list[tuple[str, float, float, dict]]:
    """(phase, start, end, attrs) spans from *_START / *_END marker pairs and from
    markers that have a duration (e.g. paired start/stop rows of a marker stream)."""
    spans, open_ = [], {}
    for on, dur, desc in markers[["onset", "duration", "description"]].itertuples(index=False):
        name, attrs = parse_description(desc)
        if dur and float(dur) > 0:
            spans.append((name, float(on), float(on) + float(dur), attrs))
        if m := _START.match(name):
            open_[m.group(1).upper()] = (on, attrs)
        elif (m := _END.match(name)) and m.group(1).upper() in open_:
            key = m.group(1).upper()
            start, a = open_.pop(key)
            spans.append((key, start, on, a))
    end = float(markers["onset"].max()) if len(markers) else np.inf
    for key, (start, a) in open_.items():  # never closed: extends to the end
        spans.append((key, start, max(end, start) + 1e9, a))
    return spans


def build_event_table(rec: Recording, gaze_events: GazeEvents | None = None,
                      aois: dict | None = None) -> pd.DataFrame:
    markers = rec.all_events() if rec is not None else pd.DataFrame(
        columns=["onset", "duration", "description"])
    rows = []
    for on, dur, desc in markers[["onset", "duration", "description"]].itertuples(index=False):
        name, attrs = parse_description(desc)
        if name.lower() in GAZE_EVENT_NAMES:  # e.g. a "fixation" cross marker
            name = f"{name}_marker"
        rows.append({"onset": float(on), "duration": float(dur or 0.0), "name": name,
                     "kind": "marker", "description": str(desc), **attrs})
    tables = [pd.DataFrame(rows)] if rows else []
    attr_cols = sorted({k for r in rows for k in r} - set(BASE_COLUMNS))

    if gaze_events is not None:
        f = gaze_events.fixations.copy()
        if len(f):
            f["name"] = "fixation"
            f["kind"] = "fixation"
            f["description"] = "fixation"
            if aois:
                hit = np.full(len(f), "", dtype=object)
                xy = f[["x", "y"]].to_numpy()
                for aname, aoi in aois.items():
                    inside = np.array([bool(aoi.contains((x, y))) for x, y in xy])
                    hit[(hit == "") & inside] = aname
                f["aoi"] = hit
            tables.append(f.drop(columns=["offset"]))
        s = gaze_events.saccades.copy()
        if len(s):
            s["name"] = s["kind"] = s["description"] = "saccade"
            tables.append(s.drop(columns=["offset", "x0", "y0", "x1", "y1"]))
        b = gaze_events.blinks.copy()
        if len(b):
            b["name"] = b["kind"] = b["description"] = "blink"
            tables.append(b.drop(columns=["offset"]))

    if not tables:
        return pd.DataFrame(columns=BASE_COLUMNS + ["phase"])
    df = pd.concat(tables, ignore_index=True).sort_values("onset", ignore_index=True)

    # ---- context from phases: the innermost (latest-starting) span decides the
    # phase and fills attributes first; outer spans fill what is still empty.
    # Inherited attributes never overwrite an event's own value.
    spans = sorted(_phase_spans(markers), key=lambda x: -x[1])
    df["phase"] = ""
    if spans:
        onset = df["onset"].to_numpy()
        phase = np.full(len(df), "", dtype=object)
        cols: dict[str, np.ndarray] = {}
        for key, start, end, attrs in spans:
            lo, hi = np.searchsorted(onset, start, "left"), np.searchsorted(onset, end, "right")
            if hi <= lo:
                continue
            seg = phase[lo:hi]
            seg[seg == ""] = key
            for k, v in attrs.items():
                if k not in cols:
                    base = df[k].to_numpy(dtype=object, copy=True) if k in df else \
                        np.full(len(df), None, dtype=object)
                    cols[k] = base
                    attr_cols.append(k)
                c = cols[k][lo:hi]
                empty = np.array([x is None or (isinstance(x, float) and np.isnan(x)) or x == ""
                                  for x in c], dtype=bool)
                c[empty] = v
        df["phase"] = phase
        for k, arr in cols.items():
            df[k] = arr
    # ---- which file / subject / session / run each event comes from
    seg_cols: list[str] = []
    if rec is not None and getattr(rec, "segments", None):
        from .io.multi import segment_columns
        for k, v in segment_columns(df["onset"].to_numpy(float), rec.segments).items():
            if k in df:  # an event's own value wins
                own = df[k]
                empty = own.isna() | (own.astype(str) == "")
                df.loc[empty, k] = v[empty.to_numpy()]
            else:
                df[k] = v
            seg_cols.append(k)
    df = _to_numeric_columns(df, set(attr_cols) - set(seg_cols))  # entities stay text
    order = BASE_COLUMNS + ["phase"] + [c for c in df.columns
                                        if c not in BASE_COLUMNS and c != "phase"]
    return df[order]


def filter_events(df: pd.DataFrame, text: str = "", query: str = "",
                  names: set[str] | None = None) -> pd.DataFrame:
    """Apply a regex on ``description``, a pandas ``query`` and a name whitelist.

    Raises ``ValueError`` with a readable message for invalid filters.
    """
    out = df
    if text.strip():
        try:
            out = out[out["description"].astype(str).str.contains(text, case=False, regex=True)]
        except re.error as e:
            raise ValueError(f"Invalid text pattern: {e}") from e
    if query.strip():
        try:
            out = out.query(query, engine="python")
        except Exception as e:  # pandas raises many types here
            raise ValueError(f"Invalid filter expression: {e}") from e
    if names is not None:
        out = out[out["name"].isin(names)]
    return out


def first_event_time(table: pd.DataFrame) -> float | None:
    """Onset of the first marker (not gaze events, file boundaries or BAD_ spans)."""
    if table is None or not len(table) or "kind" not in table:
        return None
    m = table[(table["kind"] == "marker")]
    name = m["name"].astype(str)
    m = m[~name.str.upper().str.startswith(("BAD", "EDGE"))]
    return float(m["onset"].min()) if len(m) else None
