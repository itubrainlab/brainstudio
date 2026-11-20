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
        report = self.worker.report  # progress bar: the run / subject being processed

        def work():
            rec = S.load_step(step, report)
            if rec is not None and rec.has_gaze:
                params = S.detection_params(rec.gaze, gaze_preset, gaze_params)
                rec.gaze_events = S.detect_gaze_events(rec.gaze, params, rec.segments,
                                                       report)
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
            store(S.detect_gaze_events(gaze, params, rec.segments))
            return
        self.worker.run("Detecting fixations, saccades and blinks (pymovements)…",
                        lambda: S.detect_gaze_events(gaze, params, rec.segments,
                                                     self.worker.report), store)

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

    def set_event_filter(self, conditions, match) -> bool:
        return self.session.set_event_filter(conditions, match)

    def remove_trigger_events(self):
        self.session.remove_trigger_events()

    def set_montage(self, source, names, bads=None) -> dict:
        return self.session.set_montage(source, names, bads)

    # ------------------------------------------------------------------ preprocessing
    def _blinks(self):
        ev = self.session.gaze_events
        return ev.blinks if ev is not None else None

    def preprocess(self, p: C.PreprocParams, use_ica: bool, done=None, failed=None) -> bool:
        """Filter, remove each subject's ICA components, interpolate its bad channels;
        subject by subject, with the power spectra and the blink check before and after."""
        ses, blinks = self.session, self._blinks()
        raw, settings = ses.raw, ses.subject_settings(use_ica)
        parts = [sub.part for sub in ses.subjects]
        p.per_subject = {k: {"bads": v["bads"], "exclude": v["exclude"]}
                         for k, v in settings.items()}
        recipe = ses.snapshot_pre_recipe(p, settings)

        report = self.worker.report  # progress bar: the subject being processed

        def job():
            clean = C.preprocess_subjects(raw, parts, p, settings, lambda k, n, label: report(
                k, n, f"Preprocessing: {label} ({k + 1} of {n})…"))
            spectra = C.psd_check_subjects(raw, clean, parts, lambda k, n, label: report(
                k, n, f"Spectra before / after: {label} ({k + 1} of {n})…"))
            return (clean, *spectra,
                    C.blink_check(raw, clean, C.replace(p, bads=clean.info["bads"]), blinks))

        def store(res):
            clean, before, after, ses.blink_check = res
            ses.set_clean(clean, p, before, after, recipe=recipe)
            if done:
                done()
        n = len(parts)
        return self.worker.run("Preprocessing" + (f" {n} subjects one by one…" if n > 1
                                                  else "…"), job, store, failed)

    def detect_bad_channels(self, p: C.BadChannelParams, from_event: bool, done=None):
        """Peak-to-peak bad channels of every subject, added to those its files mark bad."""
        ses, raw = self.session, self.session.raw
        subs = list(ses.subjects)

        report, n = self.worker.report, len(subs)

        def job():
            out = []
            for k, sub in enumerate(subs):
                if n > 1:
                    report(k, n, f"Bad channels: {sub.label} ({k + 1} of {n})…")
                tmin = ses.subject_first_event(sub) if from_event else None
                one = C.part_raw(raw, sub.part, C.eeg_names(raw)) if len(subs) > 1 else raw
                out.append(C.detect_bad_channels(one, p, tmin))
            return out

        def store(dfs):
            for sub, df in zip(subs, dfs):
                sub.bad_stats = df
                keep = [b for b in sub.part["bads"]]
                sub.bads = keep + [b for b in df.loc[df["bad"], "channel"] if b not in keep]
            ses.notify("bads")
            if done:
                done()
            ses.mark_stale("preprocessing")
        self.worker.run("Measuring peak-to-peak per channel"
                        + (f", {len(subs)} subjects…" if len(subs) > 1 else "…"), job, store)

    def fit_ica(self, notch, ip: C.ICAParams, from_event: bool, done=None):
        """ICA on 1–30 Hz (ERP guide) of each subject on its own (heads differ), with blink
        scores against the eye tracker; one subject's data in memory at a time."""
        import mne
        ses, raw, h = self.session, self.session.raw, None
        eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True)]
        subs = list(ses.subjects)
        plan = [(sub, list(sub.bads), ses.subject_first_event(sub) if from_event else None,
                 ses.subject_blinks(sub)) for sub in subs]

        report, n = self.worker.report, len(plan)

        def job():
            fits, errors = {}, []
            for k, (sub, bads, tmin, blinks) in enumerate(plan):
                if n > 1:
                    report(k, n, f"Fitting ICA: {sub.label} ({k + 1} of {n})…")
                one = C.part_raw(raw, sub.part) if len(subs) > 1 else raw
                try:
                    ica, fit_raw = C.fit_ica(one, bads, h, notch, ip, tmin=tmin)
                except ValueError as e:  # e.g. too few good channels: no ICA here
                    errors.append(f"{sub.label}: {e}")
                    continue
                del one
                spec = dict(bads=list(bads), h_freq=h, notch=notch, params=ip, tmin=tmin,
                            steps=dict(fit_raw.fit_steps))
                fits[sub.key] = (ica, fit_raw, C.score_blink_components(ica, fit_raw, blinks,
                                                                        eog), spec)
            if not fits:
                raise ValueError("; ".join(errors) or "no subject could be fitted")
            return fits, errors

        def store(res):
            fits, errors = res
            ses.set_icas(fits)
            ses.ica_errors = errors
            if done:
                done()
        what = "99 % variance" if ip.n_components < 1 else f"{ip.n_components} components"
        self.worker.run(f"Fitting ICA ({ip.method}, {what})"
                        + (f" for {len(subs)} subjects one by one…" if len(subs) > 1 else "…"),
                        job, store)

    def rescore_ica(self, done):
        """New eye-tracker blinks: rescore every subject's ICA components and redo the
        blink check. ``done(scores_changed, check_changed)``."""
        import mne
        ses, blinks = self.session, self._blinks()
        raw, clean, p = ses.raw, ses.clean, ses.params
        eog = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True)]
        plan = [(sub, sub.ica, sub.fit_raw, ses.subject_blinks(sub)) for sub in ses.subjects]

        report, n = self.worker.report, len(plan)

        def job():
            scores = {}
            for k, (sub, ica, fit_raw, b) in enumerate(plan):
                if ica is not None:
                    if n > 1:
                        report(k, n, f"Rescoring ICA: {sub.label} ({k + 1} of {n})…")
                    scores[sub.key] = C.score_blink_components(ica, fit_raw, b, eog)
            check = C.blink_check(raw, clean, C.replace(p, bads=clean.info["bads"]), blinks) \
                if clean is not None else (None, None)
            return scores, check

        def store(res):
            scores, check = res
            new_scores = False
            for sub, ica, _, _ in plan:
                if sub.key in scores and sub.ica is ica:
                    sub.scores, new_scores = scores[sub.key], True
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
