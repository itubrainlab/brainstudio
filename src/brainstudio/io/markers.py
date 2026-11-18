"""Turn multi-column marker streams into events.

LSL marker streams often carry several text columns per sample, e.g.
``['start', 'trials', 'conditions/Sad/JF15.mp4', '0', '2', 'sad', 'nd']``.
Each column gets a role:

* ``name``      – part of the event name (several are joined with ``_``)
* ``flag``      – start/stop flag: a ``start`` row and the next ``stop``/``end``
                  row with the same name become one event with a duration
* ``attribute`` – kept as a column of the event table (filterable)
* ``ignore``

Events are encoded as structured descriptions ``NAME|col=value|…`` that
:func:`brainstudio.events.parse_description` expands into columns.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

COLUMN_ROLES = ("name", "flag", "attribute", "ignore")
START_WORDS = {"start", "begin", "onset", "on", "open"}
STOP_WORDS = {"stop", "end", "offset", "off", "close"}


def _clean(v) -> str:
    return str(v).strip() if v is not None else ""


def column_table(rows: list[list[str]], labels: list[str]) -> pd.DataFrame:
    n = max((len(r) for r in rows), default=len(labels))
    n = max(n, len(labels))
    data = [[_clean(r[k]) if k < len(r) else "" for k in range(n)] for r in rows]
    cols = list(labels) + [f"ch{k + 1}" for k in range(len(labels), n)]
    return pd.DataFrame(data, columns=cols[:n])


def guess_columns(table: pd.DataFrame, labels: list[str], stream: str = ""
                  ) -> tuple[list[str], list[str]]:
    """Automatic (names, roles) for the columns of a marker table.

    Unlabelled columns are named ``<stream>_<k>`` so that columns of different
    marker streams do not collide in the event table."""
    n = table.shape[1]
    roles = ["attribute"] * n
    names = list(table.columns)
    generic = [bool(re.fullmatch(r"ch\d+|", str(lab).lower())) for lab in names]
    prefix = re.sub(r"\W+", "_", stream).strip("_").lower() or "col"
    names = [f"{prefix}_{k + 1}" if g else lab for k, (lab, g) in enumerate(zip(names, generic))]
    empty = [not (table.iloc[:, k] != "").any() for k in range(n)]
    flag = None
    for k in range(n):
        vals = set(table.iloc[:, k].str.lower()) - {""}
        if len(vals) >= 2 and vals <= START_WORDS | STOP_WORDS and vals & START_WORDS \
                and vals & STOP_WORDS:
            flag = k
            break
    if flag is not None:
        roles[flag] = "flag"
        if generic[flag]:
            names[flag] = "action"
    name_col = None
    for k in range(n):
        if k == flag or empty[k]:
            continue
        v = table.iloc[:, k]
        nonempty = v[v != ""]
        if 1 <= nonempty.nunique() <= 40 and nonempty.str.len().median() <= 32:
            name_col = k
            break
    if name_col is None:
        name_col = next((k for k in range(n) if k != flag and not empty[k]), 0)
    roles[name_col] = "name"
    if generic[name_col]:
        names[name_col] = "event"
    for k in range(n):
        if empty[k] and roles[k] == "attribute":
            roles[k] = "ignore"
    return names, roles


def _safe(v: str) -> str:
    return v.replace("|", "/").replace("=", ":")


def _desc(name: str, attrs: dict) -> str:
    return "|".join([_safe(name) or "marker"] + [f"{k}={_safe(v)}" for k, v in attrs.items()
                                                  if v != ""])


def events_from_markers(times: np.ndarray, table: pd.DataFrame, names: list[str],
                        roles: list[str], end_events: bool = True) -> pd.DataFrame:
    """Events (onset, duration, description) from a marker column table."""
    name_idx = [k for k, r in enumerate(roles) if r == "name"]
    flag_idx = next((k for k, r in enumerate(roles) if r == "flag"), None)
    attr_idx = [k for k, r in enumerate(roles) if r == "attribute"]
    vals = table.to_numpy(dtype=object)
    out = []
    open_: dict[str, list] = {}
    for t, row in zip(times, vals):
        name = "_".join(row[k] for k in name_idx if row[k]) or "marker"
        attrs = {names[k]: row[k] for k in attr_idx}
        flag = row[flag_idx].lower() if flag_idx is not None else ""
        if flag in START_WORDS:
            open_.setdefault(name, []).append((float(t), attrs))
        elif flag in STOP_WORDS:
            if open_.get(name):
                t0, a0 = open_[name].pop(0)
                merged = dict(a0)
                for k, v in attrs.items():
                    if v and not merged.get(k):
                        merged[k] = v          # e.g. the response, only known at the end
                    elif v and merged.get(k) != v:
                        merged[f"{k}_end"] = v
                out.append((t0, float(t) - t0, _desc(name, merged)))
                if end_events:
                    out.append((float(t), 0.0, _desc(f"{name}_end", merged)))
            else:
                out.append((float(t), 0.0, _desc(f"{name}_{flag}", attrs)))
        else:
            out.append((float(t), 0.0, _desc(name, attrs)))
    for name, items in open_.items():  # never closed
        for t0, a0 in items:
            out.append((t0, 0.0, _desc(name, a0)))
    df = pd.DataFrame(out, columns=["onset", "duration", "description"])
    return df.sort_values("onset", kind="stable").reset_index(drop=True)
