"""The event table (adapter over :mod:`brainstudio.core`)."""

from __future__ import annotations

from . import core as C
from .core import (BASE_COLUMNS, GAZE_EVENT_NAMES, OPERATORS, apply_conditions,  # noqa: F401
                   describe_conditions, filter_events, first_event_time, parse_description)


def build_event_table(rec, gaze_events=None, aois=None):
    return C.event_table(rec.all_events() if rec is not None else None, gaze_events, aois,
                         getattr(rec, "segments", None) if rec is not None else None)
