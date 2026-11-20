"""brAInStudio: EEG + eye-tracking analysis built on MNE and pymovements."""

from .model import GazeData, GazeEvents, Recording

__all__ = ["GazeData", "GazeEvents", "Recording"]
__version__ = "0.1.8"
