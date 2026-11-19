"""Controllers: one method per user action. Each runs the :mod:`brainstudio.core`
computation (in the background worker when it takes a while) and stores the result in
the :class:`~brainstudio.session.Session`, whose notifications update the views."""

from __future__ import annotations

from .. import session as S


class Controller:
    def __init__(self, session: S.Session, worker):
        self.session, self.worker = session, worker

    # ------------------------------------------------------------------ loading
    def load(self, step: dict, label: str, gaze_preset: str, gaze_params, done,
             failed) -> bool:
        """Read the recording ``step`` describes and detect its gaze events, in the
        background; ``done(rec)`` runs on the GUI thread after the session has it."""
        def work():
            rec = S.load_step(step)
            if rec is not None and rec.has_gaze:
                params = S.detection_params(rec.gaze, gaze_preset, gaze_params)
                rec.gaze_events = S.detect_gaze_events(rec.gaze, params)
                rec.detected_with = params
            return rec

        def loaded(rec):
            self.session.set_recording(rec, step)
            done(rec)
        return self.worker.run(label, work, loaded, failed)

    def attach_gaze(self, path, screen_size):
        from .. import io as bio
        self.session.set_gaze(bio.load_eyetracking_file(path, screen_size),
                              {"kind": "attach_gaze", "path": str(path),
                               "screen_size": tuple(screen_size)})

    # ------------------------------------------------------------------ gaze
    def detect_gaze(self, params, rebuild: bool = True, done=None):
        """Detect fixations, saccades and blinks (pymovements) in the background (or
        here, when the worker is busy with something else)."""
        ses = self.session
        rec, gaze = ses.rec, ses.rec.gaze

        def store(ev):
            if ses.rec is rec and rec.gaze is gaze:  # still the same data
                ses.set_gaze_events(ev, params, rebuild)
                if done:
                    done()
        if self.worker.busy:
            store(S.detect_gaze_events(gaze, params))
            return
        self.worker.run("Detecting fixations, saccades and blinks (pymovements)…",
                        lambda: S.detect_gaze_events(gaze, params), store)

    def set_gaze_offset(self, seconds: float) -> bool:
        return self.session.set_gaze_offset(seconds)

    def set_aois(self, specs: dict):
        self.session.set_aois(specs)

    # ------------------------------------------------------------------ channels
    def set_trigger_events(self, channel, mask, mode, names, as_stim) -> bool:
        return self.session.set_trigger_events(channel, mask, mode, names, as_stim)

    def remove_trigger_events(self):
        self.session.remove_trigger_events()

    def set_montage(self, source, names, bads=None) -> dict:
        return self.session.set_montage(source, names, bads)
