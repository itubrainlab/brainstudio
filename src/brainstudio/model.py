"""Core data containers shared by loaders, analysis and GUI.

All times are in seconds on a single *recording time base*: t = 0 is the
first EEG sample when EEG is present, otherwise the first gaze sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd


@dataclass
class GazeData:
    """Continuous gaze samples in screen pixel coordinates."""

    times: np.ndarray  # (n,) seconds, recording time base
    x: np.ndarray  # (n,) px, NaN = missing
    y: np.ndarray  # (n,) px, NaN = missing
    pupil: np.ndarray | None = None  # (n,) arbitrary units
    sfreq: float | None = None  # nominal sampling rate
    screen_size: tuple[int, int] = (1920, 1080)  # px (width, height)
    label: str = "gaze"
    # Optional per-eye data, kept for reference
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.pupil is not None:
            self.pupil = np.asarray(self.pupil, dtype=float)
        n = len(self.times)
        if len(self.x) != n or len(self.y) != n:
            raise ValueError("times, x and y must have the same length")
        if self.sfreq is None and n > 1:
            self.sfreq = float(1.0 / np.median(np.diff(self.times)))

    @property
    def n_samples(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if self.n_samples else 0.0

    def window(self, tmin: float, tmax: float) -> slice:
        """Index slice of samples with tmin <= t < tmax."""
        i0 = int(np.searchsorted(self.times, tmin, "left"))
        i1 = int(np.searchsorted(self.times, tmax, "left"))
        return slice(i0, i1)

    def at(self, t: float) -> tuple[float, float]:
        """Gaze position closest to time t."""
        if not self.n_samples:
            return (np.nan, np.nan)
        i = int(np.clip(np.searchsorted(self.times, t), 0, self.n_samples - 1))
        return float(self.x[i]), float(self.y[i])

    def shifted(self, dt: float) -> GazeData:
        return GazeData(
            self.times + dt, self.x, self.y, self.pupil, self.sfreq,
            self.screen_size, self.label, dict(self.extra),
        )


@dataclass
class GazeEvents:
    """Result of oculomotor event detection (all times in seconds)."""

    fixations: pd.DataFrame  # onset, offset, duration, x, y
    saccades: pd.DataFrame  # onset, offset, duration, x0, y0, x1, y1, amplitude, peak_velocity
    blinks: pd.DataFrame  # onset, offset, duration

    @classmethod
    def empty(cls) -> GazeEvents:
        return cls(
            pd.DataFrame(columns=["onset", "offset", "duration", "x", "y"]),
            pd.DataFrame(columns=["onset", "offset", "duration", "x0", "y0",
                                  "x1", "y1", "amplitude", "peak_velocity"]),
            pd.DataFrame(columns=["onset", "offset", "duration"]),
        )


def is_data_annotation(descriptions: pd.Series) -> pd.Series:
    """True for annotations about the data rather than events (BAD_… spans, EDGE/BAD
    boundaries of joined files): these keep their time when events are shifted."""
    d = descriptions.astype(str).str.upper()
    return d.str.startswith("BAD") | d.str.startswith("EDGE")


@dataclass
class Recording:
    """One loaded session: EEG (MNE Raw), gaze, and discrete events."""

    raw: mne.io.BaseRaw | None = None
    gaze: GazeData | None = None
    # Discrete markers: columns onset (s), duration (s), description (str)
    events: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["onset", "duration", "description"])
    )
    source: Path | None = None
    description: str = ""
    gaze_events: GazeEvents | None = None
    # Streams of a multi-stream source (XDF): dicts with key, name, type, channels,
    # srate, format, samples, labels, auto_role and role (EEG/gaze/markers/ignore)
    streams: list = field(default_factory=list)
    # Files the recording is made of: dicts with start/end (s) and BIDS entities
    # (subject, session, task, acq, run) and file; see brainstudio.io.multi
    segments: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    # Dataset information (BIDS: dataset, participant, eeg.json fields, bad channels
    # with reasons, where the electrode positions come from); see io.bids.bids_metadata
    metadata: dict = field(default_factory=dict)

    @property
    def has_eeg(self) -> bool:
        return self.raw is not None

    @property
    def has_gaze(self) -> bool:
        return self.gaze is not None and self.gaze.n_samples > 0

    @property
    def duration(self) -> float:
        ends = []
        if self.raw is not None:
            ends.append(self.raw.times[-1])
        if self.has_gaze:
            ends.append(self.gaze.times[-1])
        return float(max(ends)) if ends else 0.0

    def all_events(self) -> pd.DataFrame:
        """Markers plus EEG annotations (deduplicated)."""
        frames = [self.events]
        if self.raw is not None and len(self.raw.annotations):
            a = self.raw.annotations
            # With orig_time set, onsets are relative to meas_date (sample 0),
            # otherwise to the first sample of the data (raw.times[0]).
            offset = self.raw.first_time if a.orig_time is not None else 0.0
            frames.append(pd.DataFrame({
                "onset": a.onset - offset,
                "duration": a.duration,
                "description": a.description,
            }))
        df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(
            len(f) for f in frames) else self.events.copy()
        # The same marker often arrives from several sources (EEG annotations,
        # eye-tracker messages) with ms-level jitter: merge within 10 ms.
        df = df.assign(description=df["description"].astype(str)).sort_values(
            ["description", "onset"], ignore_index=True)
        dup = (df["description"].eq(df["description"].shift())
               & (df["onset"].diff().abs() < 0.010))
        return df[~dup].sort_values("onset", ignore_index=True)

    def summary(self) -> str:
        lines = [f"Source: {self.source}" if self.source else "Source: (in memory)"]
        if self.raw is not None:
            r = self.raw
            types = mne.channel_indices_by_type(r.info, picks="all")
            ch = ", ".join(f"{k}:{len(v)}" for k, v in types.items() if len(v))
            lines.append(f"EEG: {len(r.ch_names)} ch ({ch}) @ {r.info['sfreq']:g} Hz, "
                         f"{r.times[-1]:.1f} s")
        else:
            lines.append("EEG: none")
        if self.has_gaze:
            g = self.gaze
            valid = np.isfinite(g.x) & np.isfinite(g.y)
            lines.append(f"Gaze: {g.n_samples} samples @ {g.sfreq:.1f} Hz, "
                         f"{g.duration:.1f} s, {100 * valid.mean():.1f}% valid, "
                         f"screen {g.screen_size[0]}x{g.screen_size[1]}"
                         f"{', pupil' if g.pupil is not None else ''}")
        else:
            lines.append("Gaze: none")
        lines.append(f"Events: {len(self.all_events())}")
        return "\n".join(lines)
