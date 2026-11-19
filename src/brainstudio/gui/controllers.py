"""Controllers: one method per user action. Each runs the :mod:`brainstudio.core`
computation (in the background worker when it takes a while) and stores the result in
the :class:`~brainstudio.session.Session`, whose notifications update the views."""

from __future__ import annotations

from .. import core as C
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

    def set_stimuli(self, folder, column, mapping):
        """Which stimulus file each value of an event column shows."""
        self.session.set_stimuli(folder, column, mapping)

    def set_aois(self, specs: dict):
        self.session.set_aois(specs)

    # ------------------------------------------------------------------ channels
    def set_trigger_events(self, channel, mask, mode, names, as_stim) -> bool:
        return self.session.set_trigger_events(channel, mask, mode, names, as_stim)

    def remove_trigger_events(self):
        self.session.remove_trigger_events()

    def set_montage(self, source, names, bads=None) -> dict:
        return self.session.set_montage(source, names, bads)

    # ------------------------------------------------------------------ preprocessing
    def _blinks(self):
        ev = self.session.gaze_events
        return ev.blinks if ev is not None else None

    def preprocess(self, p: C.PreprocParams, ica, done=None, failed=None) -> bool:
        """Filter, remove ICA components, interpolate and re-reference; with the power
        spectra and the blink check before and after."""
        ses, blinks = self.session, self._blinks()
        raw, recipe = ses.raw, ses.snapshot_pre_recipe(p, ica)

        def job():
            clean = C.run_preprocessing(raw, p, ica)
            return (clean, *C.psd_check(raw, clean), C.blink_check(raw, clean, p, blinks))

        def store(res):
            clean, before, after, ses.blink_check = res
            ses.set_clean(clean, p, before, after, recipe=recipe)
            if done:
                done()
        return self.worker.run("Preprocessing…", job, store, failed)

    def detect_bad_channels(self, p: C.BadChannelParams, tmin, keep=(), done=None):
        """Peak-to-peak bad channels, added to ``keep`` (those the dataset marks bad)."""
        ses, raw = self.session, self.session.raw

        def store(df):
            ses.bad_stats = df
            keep_ = list(keep)
            ses.params.bads = keep_ + [b for b in df.loc[df["bad"], "channel"]
                                       if b not in keep_]
            if done:
                done()
            ses.mark_stale("preprocessing")
        self.worker.run("Measuring peak-to-peak per channel…",
                        lambda: C.detect_bad_channels(raw, p, tmin), store)

    def fit_ica(self, bads, notch, ip: C.ICAParams, tmin, done=None):
        """ICA on 1–30 Hz (ERP guide) with blink scores against the eye tracker."""
        import mne
        ses, raw, blinks, h = self.session, self.session.raw, self._blinks(), None
        eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True)]

        def job():
            ica, fit_raw = C.fit_ica(raw, bads, h, notch, ip, tmin=tmin)
            return (ica, fit_raw, C.score_blink_components(ica, fit_raw, blinks, eog),
                    ica.get_sources(fit_raw).get_data())

        def store(res):
            ica, fit_raw, scores, sources = res
            ses.set_ica(ica, fit_raw, scores, sources, (tuple(bads), h, notch, tmin),
                        dict(bads=list(bads), h_freq=h, notch=notch, params=ip, tmin=tmin,
                             steps=dict(fit_raw.fit_steps)))
            if done:
                done()
        what = "99 % variance" if ip.n_components < 1 else f"{ip.n_components} components"
        self.worker.run(f"Fitting ICA ({ip.method}, {what})…", job, store)

    def rescore_ica(self, done):
        """New eye-tracker blinks: rescore the ICA components and redo the blink check.
        ``done(scores_changed, check_changed)``."""
        import mne
        ses, blinks = self.session, self._blinks()
        raw, clean, p, ica, fit_raw = ses.raw, ses.clean, ses.params, ses.ica, ses.ica_fit_raw
        eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True)]

        def job():
            scores = (C.score_blink_components(ica, fit_raw, blinks, eog)
                      if ica is not None and fit_raw is not None else None)
            return scores, C.blink_check(raw, clean, p, blinks)

        def store(res):
            scores, check = res
            new_scores = ses.ica is ica and scores is not None
            if new_scores:
                ses.ica_scores = scores
            new_check = ses.clean is clean and clean is not None
            if new_check:
                ses.blink_check = check
            done(new_scores, new_check)
        self.worker.run("Rescoring ICA against the shifted blinks…", job, store)

    # ------------------------------------------------------------------ epoching
    def build_epochs(self, groups: dict, specs, desc: dict, settings: C.EpochSettings,
                     done=None, failed=None) -> bool:
        """Epochs of each group's events (from the preprocessed data, else the raw)."""
        ses, blinks = self.session, self._blinks()
        raw = ses.eeg_input()
        # what these epochs come from, for scripts that reproduce them
        recipe = ses.snapshot_epoch_recipe(raw, specs, settings)

        def store(res):
            epochs, summary = res
            ses.set_epochs(epochs, summary, settings, desc, recipe=recipe)
            if done:
                done(epochs)
        return self.worker.run("Building epochs…",
                               lambda: C.build_group_epochs(raw, groups, settings, blinks),
                               store, failed)

    # ------------------------------------------------------------------ analyses
    def compute_erp(self, spec: C.ErpSpec, done) -> bool:
        ses = self.session
        epochs, recipe = ses.epochs, ses.epoch_recipe

        def store(res):
            res["recipe"] = recipe
            ses.erp = res
            done(res)
        return self.worker.run("Computing ERPs…", lambda: C.run_erp(epochs, spec), store)

    def cluster_test(self, diffs, tfce: bool, n_permutations: int, tmin, tmax, done):
        self.worker.run("Cluster test across subjects…",
                        lambda: C.group_cluster_test(diffs, tfce=tfce, tmin=tmin, tmax=tmax,
                                                     n_permutations=n_permutations), done)

    def compute_bandpower(self, spec: C.BandPowerSpec, done) -> bool:
        ses = self.session
        raw, table, epochs, recipe = ses.eeg_input(), ses.event_table, ses.epochs, \
            ses.epoch_recipe

        def store(res):
            res["recipe"] = recipe
            # the baseline came from the same data and events as the epochs
            res["consistent"] = recipe is not None and raw is recipe["data"] \
                and table is recipe["table"]
            ses.bp, ses.tc = res, None
            done(res)
        return self.worker.run("Computing band power (MNE PSD)…",
                               lambda: C.run_bandpower(epochs, raw, table, spec), store)

    def compute_timecourse(self, band: str, done) -> bool:
        ses = self.session
        r, epochs, raw = ses.bp, ses.epochs, ses.eeg_input()

        def store(res):
            ses.tc = res
            done(res)
        return self.worker.run(f"Computing {band} time course…",
                               lambda: C.run_timecourse(r, epochs, raw, band), store)

    def compute_indices(self, spec: C.IndicesSpec, done) -> bool:
        ses = self.session
        r, rec, gev, epochs = ses.bp, ses.rec, ses.gaze_events, ses.epochs

        def store(res):
            res["recipe"] = r.get("recipe")
            res["bp"] = r
            res["consistent"] = r.get("consistent", False) and gev is \
                (r.get("recipe") or {}).get("gaze_events", gev)
            ses.idx = res
            done(res)
        return self.worker.run(
            "Computing indices…",
            lambda: C.run_indices(r, epochs, rec.gaze if rec is not None else None, gev, spec),
            store)
