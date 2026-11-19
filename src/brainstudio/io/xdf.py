"""Moved to :mod:`brainstudio.core` (kept for imports)."""
from ..core import GAZE_COLUMN_ROLES, ROLES, describe_streams, gaze_columns, stream_key  # noqa: F401
from ..core import clean_timestamps as _clean_timestamps  # noqa: F401

EEG_CHANNEL_TYPES = ("eeg", "eog", "ecg", "emg", "misc", "stim")


def load_xdf(path, screen_size=(1920, 1080), roles=None, options=None):
    from . import load
    return load(path, screen_size, roles, options)
