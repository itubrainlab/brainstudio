"""Gaze events on ``GazeData`` (adapter over :mod:`brainstudio.core`)."""

from __future__ import annotations

from . import core as C
from .core import (PRESETS, DetectionParams, aoi_statistics, fixation_aois,  # noqa: F401
                   heatmap, make_aois, preset_name)
from .io import gaze_table


def suggest_preset(g) -> str:
    return C.suggest_preset(gaze_table(g))


def sampling_rate(g) -> float:
    return C.gaze_rate(gaze_table(g))


def detect_events(g, p: DetectionParams = DetectionParams()):
    return C.detect_gaze_events(gaze_table(g), p)


def clean_pupil(g, blinks, pad: float = 0.05):
    return C.clean_pupil(gaze_table(g), blinks, pad)


summary_table = C.gaze_summary
