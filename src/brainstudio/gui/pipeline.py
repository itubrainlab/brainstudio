"""Qt bridge to the :class:`~brainstudio.session.Session`: its notifications become the
``changed`` / ``stateChanged`` signals the views connect to, and its state is read (and
written) through this object, e.g. ``pipeline.clean`` is ``session.clean``."""

from __future__ import annotations

from PySide6 import QtCore

from ..session import Session


class EEGPipeline(QtCore.QObject):
    #: what changed: recording, gaze, gaze_events, events, channels, preprocessing,
    #: epoching
    changed = QtCore.Signal(str)
    #: the stage states (pending / ok / stale) changed
    stateChanged = QtCore.Signal()

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.__dict__["session"] = session
        session.subscribe(self._notified)

    def _notified(self, what: str):
        if what == "state":
            self.stateChanged.emit()
        else:
            self.changed.emit(what)

    def __getattr__(self, name):
        session = self.__dict__.get("session")
        if session is None:
            raise AttributeError(name)
        return getattr(session, name)

    def __setattr__(self, name, value):
        session = self.__dict__.get("session")
        if session is not None and hasattr(session, name):
            setattr(session, name, value)
        else:
            super().__setattr__(name, value)
