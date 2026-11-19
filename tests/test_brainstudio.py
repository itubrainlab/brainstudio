"""Tests on synthetic data (BIDS + XDF round trips, gaze detection, FRPs)."""

from brainstudio import core as C
from brainstudio import session as S
import numpy as np
import mne
import pandas as pd
import pytest
from brainstudio.demo import simulate, write_bids, write_xdf_demo
from brainstudio.io import load_bids_run, load_xdf, scan_bids
from brainstudio.core import clean_timestamps as _clean_timestamps
from brainstudio.io.xdf_writer import write_xdf


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    write_bids(root / "bids", subjects=("01",))
    write_xdf_demo(root / "demo.xdf")
    return root


def test_bids_scan_and_load(demo):
    runs = scan_bids(demo / "bids")
    assert len(runs) == 1 and runs[0].eeg is not None and runs[0].gaze
    rec = load_bids_run(demo / "bids", runs[0])
    assert rec.has_eeg and rec.has_gaze
    assert rec.gaze.times[0] == pytest.approx(0.5, abs=1e-6)  # StartTime alignment
    assert rec.gaze.sfreq == pytest.approx(500, rel=1e-3)
    ev = rec.all_events()
    # EEG events.tsv and eye-tracker messages are merged without duplicates
    assert (ev.description == "stimulus").sum() == len(simulate(seed=0).markers) // 2


def test_xdf_load(demo):
    rec = load_xdf(demo / "demo.xdf")
    assert len(rec.raw.ch_names) == 32
    assert rec.gaze.times[0] == pytest.approx(0.5, abs=0.01)
    assert rec.all_events().description.iloc[0] == "stimulus"
    assert rec.all_events().onset.iloc[0] == pytest.approx(3.0, abs=0.01)


def test_fixation_detection_matches_truth():
    sim = simulate(duration=60, seed=3)
    from brainstudio.model import GazeData
    g = GazeData(sim.gaze_t, sim.gx, sim.gy, sim.pupil)
    fix = S.detect_gaze_events(g).fixations
    err = np.abs(fix.onset.to_numpy()[:, None] - sim.fix_onsets[None]).min(1)
    assert np.median(err) < 0.03
    assert 0.8 < len(fix) / len(sim.fix_onsets) < 1.4


def test_aoi_stats_use_pymovements():
    import pandas as pd
    fix = pd.DataFrame({"onset": [0, 1], "offset": [.2, 1.3], "duration": [.2, .3],
                        "x": [100, 900], "y": [100, 500]})
    st = C.aoi_statistics(fix, C.make_aois({"a": (50, 50, 100, 100)}))
    assert st.n_fixations.iloc[0] == 1 and st.dwell_time.iloc[0] == pytest.approx(.2)


def test_frp_recovers_lambda_response(demo):
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = C.run_preprocessing(rec.raw, C.PreprocParams(l_freq=1, h_freq=30))
    ev = S.detect_gaze_events(rec.gaze)
    fix = ev.fixations[ev.fixations.duration >= 0.1].assign(name="fixation")
    evk = C.epochs_from_table(raw, fix, -0.2, 0.6).average()
    oz = evk.copy().pick(["Oz"]).data[0]
    assert 0.06 < evk.times[np.argmax(oz)] < 0.14  # simulated P1 at ~90 ms


def test_dejitter_follows_rate_change():
    """A stream whose rate changes midway (seen in Gazepoint recordings) must keep its
    timestamps: one straight line through them misplaces samples by seconds."""
    rng = np.random.default_rng(0)
    true = np.r_[np.arange(150_000) / 149.28, 150_000 / 149.28 + np.arange(60_000) / 148.53]
    stamped = true + 2000 + rng.uniform(-0.01, 0.01, len(true))  # jitter
    clean, _, info = _clean_timestamps({"time_stamps": stamped,
                                        "info": {"nominal_srate": ["60"]}})
    assert np.abs(clean - 2000 - true).max() < 0.03
    assert np.all(np.diff(clean) >= 0)
    assert any("not constant" in m for m in info["notes"])
    # a steady stream is still a straight line (jitter removed, no note)
    steady = 2000 + np.arange(100_000) / 256 + rng.uniform(-0.01, 0.01, 100_000)
    clean, _, info = _clean_timestamps({"time_stamps": steady,
                                        "info": {"nominal_srate": ["256"]}})
    assert np.abs(np.diff(clean, 2)).max() < 1e-6
    assert not any("not constant" in m for m in info["notes"])


def test_foreign_clock_samples_recover_offset(tmp_path):
    """Samples stamped on another clock (seen in real Gazepoint recordings)."""
    n, sr, off = 3000, 150.0, 325222.957
    ts = 1000 + np.arange(n) / sr
    for i in (500, 1500, 1501):
        ts[i] += off
    stream = {"time_stamps": ts, "info": {"nominal_srate": [str(sr)]}}
    clean, idx, info = _clean_timestamps(stream)
    assert len(idx) == n - 3
    assert info["foreign_offset"] == pytest.approx(off, abs=0.01)
    # loader maps the stream onto the EEG clock
    eeg_t = 1000 + off + np.arange(int(20 * 256)) / 256
    write_xdf(tmp_path / "f.xdf", [
        dict(name="EEG", type="EEG", srate=256, format="float32", labels=["Cz", "Pz"],
             timestamps=eeg_t, data=np.zeros((len(eeg_t), 2))),
        dict(name="Tracker", type="gaze", srate=sr, format="float32",
             labels=["FPOGX", "FPOGY", "FPOGV"], timestamps=ts,
             data=np.column_stack([np.full(n, .5), np.full(n, .25), np.ones(n)])),
    ])
    rec = load_xdf(tmp_path / "f.xdf")
    assert rec.gaze.times[0] == pytest.approx(0.0, abs=0.02)
    assert np.nanmedian(rec.gaze.x) == pytest.approx(960)
    assert np.nanmedian(rec.gaze.y) == pytest.approx(270)


def test_event_table_parsing_and_context(demo):
    import mne
    import pandas as pd
    from brainstudio.model import Recording

    ev = pd.DataFrame({"onset": [1.0, 2.0, 3.0, 9.0, 10.0],
                       "duration": 0.0,
                       "description": ["TASK_START|n=1|cond=AI", "PAGE|page=2", "PAGE|page=3",
                                       "TASK_END|n=1|cond=AI", "PAGE|page=4"]})
    rec = Recording(events=ev)
    fix = pd.DataFrame({"onset": [2.5, 9.5], "offset": [2.7, 9.8], "duration": [.2, .3],
                        "x": [10.0, 10.0], "y": [10.0, 10.0]})
    from brainstudio.model import GazeEvents
    ge = GazeEvents.empty()
    ge.fixations = fix
    tab = S.event_table(rec, ge)
    assert pd.api.types.is_numeric_dtype(tab["page"])
    sel = C.filter_events(tab, query='phase == "TASK" and cond == "AI"', names={"fixation"})
    assert list(sel.onset) == [2.5]  # the 9.5 s fixation is after TASK_END
    assert len(C.filter_events(tab, query="page >= 3", names={"PAGE"})) == 2
    with pytest.raises(ValueError):
        C.filter_events(tab, query="cond ==")


def test_montage_and_metadata_epochs(demo):
    import mne

    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = rec.raw.copy()
    raw.set_montage(None)
    names = list(raw.ch_names)
    raw.rename_channels({n: f"ch{i}" for i, n in enumerate(names)})
    rep = C.apply_montage(raw, C.montage_from_source("standard_1005"), names)
    assert rep["n_renamed"] == len(names) and rep["n_positioned"] == 30
    assert raw.get_channel_types(["HEOG"]) == ["eog"]

    ge = S.detect_gaze_events(rec.gaze)
    tab = S.event_table(rec, ge)
    ep = C.epochs_from_table(C.run_preprocessing(raw, C.PreprocParams(l_freq=1, h_freq=30)),
                             C.filter_events(tab, names={"fixation", "stimulus"}), -0.2, 0.5)
    assert set(ep.event_id) == {"fixation", "stimulus"} and ep.metadata is not None


def test_channel_reference(demo):
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    assert C.parse_reference("keep", rec.raw) is None
    assert C.parse_reference("CAR", rec.raw) == "average"
    assert C.parse_reference("cz", rec.raw) == ["Cz"]
    assert C.parse_reference("TP9, Cz", rec.raw.copy().rename_channels({"T7": "TP9"})) == [
        "TP9", "Cz"]
    with pytest.raises(ValueError, match="not found"):
        C.parse_reference("M1", rec.raw)
    out = C.run_preprocessing(rec.raw, C.PreprocParams(l_freq=None, h_freq=None, notch=None, reference="Cz"))
    assert np.abs(out.get_data(picks=["Cz"])).max() < 1e-12  # reference channel is flat
    pair = C.run_preprocessing(rec.raw, C.PreprocParams(l_freq=None, h_freq=None, notch=None, reference="O1, O2"))
    o = pair.get_data(picks=["O1", "O2"])
    assert np.abs(o.sum(0)).max() < 1e-12  # linked reference: O1 = -O2




def test_gtec_64_layout_montage():
    import mne
    assert "gtec_64" in C.builtin_montages()
    m = C.montage_from_source("gtec_64")
    assert len(m.ch_names) == 64
    assert m.ch_names[:3] == ["Fp1", "Fpz", "Fp2"] and m.ch_names[18] == "FC3"
    assert m.ch_names[-4:] == ["F9", "F10", "A1", "A2"]
    info = mne.create_info([f"EEG{i:03d}" for i in range(1, 65)], 256.0, "eeg")
    raw = mne.io.RawArray(np.zeros((64, 10)), info, verbose="error")
    rep = C.apply_montage(raw, m, list(m.ch_names))
    assert rep["n_renamed"] == 64 and rep["n_positioned"] == 64 and not rep["missing"]
    assert raw.ch_names[29] == "Cz"


def test_pipeline_bads_ica_groups_bandpower(demo):
    """Preprocessing → epoching → analysis on synthetic data with real blink artifacts."""

    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = rec.raw.copy().load_data()
    raw._data[raw.ch_names.index("T7")] = 0.0                                   # flat
    raw._data[raw.ch_names.index("C4")] += np.random.default_rng(0).normal(
        0, 400e-6, raw.n_times)                                                  # noisy
    bad = C.detect_bad_channels(raw)
    assert {"T7", "C4"} <= set(bad.loc[bad.bad, "channel"])
    assert bad.bad.sum() <= 4

    ev = S.detect_gaze_events(rec.gaze)
    bads = ["T7", "C4"]
    ica, fit_raw = C.fit_ica(raw, bads, 40.0, None, C.ICAParams(n_components=10))
    assert not set(bads) & set(ica.ch_names)
    sc = C.score_blink_components(ica, fit_raw, ev.blinks)
    assert sc.suggested.any()
    assert sc.loc[sc.suggested, "blink_t"].max() == sc.blink_t.max()
    params = C.PreprocParams(bads=bads, l_freq=1, h_freq=40, reference="average",
                             ica_exclude=sc.loc[sc.suggested, "component"].tolist())
    clean = C.run_preprocessing(raw, params, ica)
    assert clean.info["bads"] == bads and raw.info["bads"] == []  # input untouched
    before = C.blink_locked_average(C.filtered_copy(raw, 1, 40, None), ev.blinks, ["Fp1", "Fp2"])
    after = C.blink_locked_average(clean, ev.blinks, ["Fp1", "Fp2"])
    assert np.ptp(after[1]) < 0.5 * np.ptp(before[1])                            # blinks gone

    table = S.event_table(rec, ev)
    groups = {"stim": table[table.name == "stimulus"],
              "fix": table[table.name == "fixation"].iloc[:200]}
    eps, summary = C.build_group_epochs(clean, groups, C.EpochSettings(-0.2, 1.0, None),
                                         ev.blinks)
    assert set(eps) == {"stim", "fix"} and (summary.kept > 0).all()
    assert eps["stim"].baseline is None and "group" in eps["stim"].metadata
    assert not set(bads) & set(eps["stim"].ch_names)

    s = C.BandSettings(tmin=0.0, tmax=1.0)
    base = C.BandBaseline(np.array([1.0]), 0.0, 20.0)
    per = []
    for g, ep in eps.items():
        bp = C.band_power(ep, s)
        assert bp.shape == (len(ep), len(ep.ch_names), len(s.bands)) and np.nanmin(bp) > 0
        bb, owner = C.baseline_band_power(clean, base, s, ep.ch_names, 1.0)
        assert len(bb) == 20 and (owner == 0).all()
        ref, sd = C.reference_for(ep.metadata.onset.to_numpy(), base, bb, owner)
        assert np.isfinite(C.normalise(bp, ref, sd, "dB")).mean() > 0.9
        idx, _ = C.eeg_indices(bp, ep.ch_names, s.bands, ref=ref)
        assert idx["engagement β/(α+θ)"].notna().all()
        assert idx["α ERD % (parietal)"].notna().all()
        idx["group"] = g
        per.append(idx)
    import pandas as pd
    tab = C.comparison_table(pd.concat(per), ["θ/α", "α ERD % (parietal)"], ["stim", "fix"])
    assert list(tab.columns[:3]) == ["measure", "stim mean", "stim sem"]
    assert tab["q (FDR)"].notna().all() and (tab["q (FDR)"] >= tab["p"] - 1e-12).all()




def test_xdf_stream_roles_override(demo):
    rec = load_xdf(demo / "demo.xdf")
    roles = {s["name"]: s for s in rec.streams}
    assert {s["role"] for s in rec.streams} >= {"EEG", "gaze", "markers"}
    assert all(s["role"] == s["auto_role"] for s in rec.streams)
    gaze_key = next(s["key"] for s in rec.streams if s["role"] == "gaze")
    eeg_key = next(s["key"] for s in rec.streams if s["role"] == "EEG")
    no_gaze = load_xdf(demo / "demo.xdf", roles={gaze_key: "ignore"})
    assert no_gaze.gaze is None and no_gaze.raw is not None
    assert next(s for s in no_gaze.streams if s["key"] == gaze_key)["role"] == "ignore"
    no_eeg = load_xdf(demo / "demo.xdf", roles={eeg_key: "ignore"})
    assert no_eeg.raw is None and no_eeg.gaze is not None
    with pytest.raises(ValueError, match="Only one stream can be EEG"):
        load_xdf(demo / "demo.xdf", roles={gaze_key: "EEG"})
    assert roles  # every stream is listed with its metadata


def test_numeric_string_stream_is_gaze(tmp_path):
    """Gazepoint-style streams send numbers as text at a regular rate: read as data."""
    t = np.arange(0, 20, 1 / 60)
    x = 0.5 + 0.1 * np.sin(t)
    rows = [[f"{a:.5f}", f"{0.4:.5f}", "1", f"{25 + np.sin(k):.3f}", f"{26:.3f}"]
            for k, a in enumerate(x)]
    eeg_t = np.arange(0, 20, 1 / 100)
    write_xdf(tmp_path / "s.xdf", [
        dict(name="GazepointGaze", type="gaze", srate=60, format="string",
             labels=["FPOGX", "FPOGY", "FPOGV", "LPD", "RPD"], timestamps=t, data=rows),
        dict(name="EEG", type="EEG", srate=100, format="float32", labels=["Cz", "Pz"],
             timestamps=eeg_t, data=np.random.default_rng(0).normal(0, 10, (len(eeg_t), 2))),
        dict(name="markers", type="Markers", srate=0, format="string", labels=["m"],
             timestamps=np.array([1.0, 2.0]), data=["a", "b"]),
    ])
    rec = load_xdf(tmp_path / "s.xdf")
    roles = {s["name"]: s["role"] for s in rec.streams}
    assert roles == {"GazepointGaze": "gaze", "EEG": "EEG", "markers": "markers"}
    assert rec.gaze is not None and np.nanmean(rec.gaze.x) == pytest.approx(0.5 * 1920, rel=0.05)
    assert np.nanmedian(rec.gaze.pupil) == pytest.approx(25.5, abs=0.6)


def test_multicolumn_markers_pair_and_keep_every_column(tmp_path):
    """start/stop rows → events with duration; every column → an event-table column;
    gaze events inside a trial inherit its phase and attributes."""
    eeg_t = np.arange(0, 30, 1 / 100)
    rows = [["start", "trials", "clip1.mp4", "sad", ""], ["stop", "trials", "clip1.mp4", "sad",
                                                          "angry"],
            ["start", "trials", "clip2.mp4", "happy", ""], ["end", "trials", "clip2.mp4",
                                                            "happy", "happy"],
            ["start", "fixation", "clip2.mp4", "happy", ""]]
    write_xdf(tmp_path / "m.xdf", [
        dict(name="EEG", type="EEG", srate=100, format="float32", labels=["Cz", "Pz"],
             timestamps=eeg_t, data=np.random.default_rng(0).normal(0, 10, (len(eeg_t), 2))),
        dict(name="exp_stim", type="Markers", srate=0, format="string",
             labels=["c1", "c2", "c3", "c4", "c5"],
             timestamps=np.array([2.0, 5.0, 10.0, 12.5, 20.0]), data=rows),
    ])
    rec = load_xdf(tmp_path / "m.xdf")
    st = next(s for s in rec.streams if s["name"] == "exp_stim")
    auto = st["auto_options"]["markers"]
    assert auto["columns"][:2] == ["flag", "name"]
    tab = S.event_table(rec)
    trials = tab[tab.name == "trials"].reset_index(drop=True)
    assert list(trials.duration) == pytest.approx([3.0, 2.5])
    assert {"c3", "c4", "c5"} <= set(tab.columns)
    assert list(trials.c5) == ["angry", "happy"]                     # response from stop row
    assert (tab.name == "trials_end").sum() == 2
    assert (tab.name == "fixation_marker").sum() == 1   # unclosed start kept; no clash
    assert (tab.name == "fixation").sum() == 0          # with gaze "fixation" events
    # user options: rename and re-assign columns
    opts = {st["key"]: {"names": ["action", "event", "stimulus", "emotion", "response"],
                        "columns": ["flag", "name", "attribute", "attribute", "ignore"],
                        "end_events": False}}
    tab2 = S.event_table(load_xdf(tmp_path / "m.xdf", options=opts))
    assert "emotion" in tab2 and "response" not in tab2 and "trials_end" not in set(tab2.name)
    # phase inheritance from events with a duration; "fixation" is not a *_ON phase
    assert __import__("brainstudio.core").core._START.match("fixation") is None


def test_load_many_xdf_with_entities(demo, tmp_path):
    """Several files → one recording; events carry subject/session/run from file names."""
    import shutil
    from brainstudio.io import entities_from_path, load_many
    a = tmp_path / "sub-01_ses-1_task-free_run-01_eeg.xdf"
    b = tmp_path / "sub-01_ses-2_task-free_run-02_eeg.xdf"
    shutil.copy(demo / "demo.xdf", a)
    shutil.copy(demo / "demo.xdf", b)
    assert entities_from_path(b) == {"subject": "01", "session": "2", "bids_task": "free",
                                     "run": "02", "file": b.name}
    one = load_xdf(a)
    rec = load_many([a, b])
    assert rec.raw.n_times == 2 * one.raw.n_times
    assert "BAD boundary" in set(rec.raw.annotations.description)
    assert len(rec.segments) == 2 and rec.segments[1]["start"] == pytest.approx(
        one.raw.n_times / one.raw.info["sfreq"])
    tab = S.event_table(rec)
    stim = tab[tab.name == "stimulus"]
    assert set(stim.session) == {"1", "2"} and set(stim.run) == {"01", "02"}  # text kept
    assert (stim.session == "2").sum() == (stim.session == "1").sum()
    assert rec.gaze.times[-1] > rec.segments[1]["start"]


def test_trigger_channel_codes_and_mask():
    import mne
    sf = 250
    n = sf * 20
    data = np.random.default_rng(0).normal(0, 1e-5, (2, n))
    trig = np.zeros(n)
    for t, c in [(2, 1), (5, 2), (8, 1), (11, 3), (14, 258)]:
        trig[int(t * sf):int(t * sf) + 5] = c
    raw = mne.io.RawArray(np.vstack([data, trig * 1e-6]),
                          mne.create_info(["Fz", "Cz", "TRIG"], sf, "eeg"), verbose="error")
    raw._orig_units = {"TRIG": "uV"}        # as EDF readers keep it
    assert C.guess_trigger_channel(raw) == "TRIG"
    assert C.type_channels_by_name(raw) == {"TRIG": "stim"}   # re-typed on load
    assert C.guess_trigger_channel(raw) == "TRIG"
    ev = C.trigger_codes(raw, "TRIG")
    assert ev.code.tolist() == [1, 2, 1, 3, 258]
    assert ev.onset.tolist() == pytest.approx([2, 5, 8, 11, 14])
    assert C.trigger_codes(raw, "TRIG", mask=255).code.tolist() == [1, 2, 1, 3, 2]
    assert C.trigger_codes(raw, "TRIG", mode="threshold").onset.tolist() == pytest.approx([14])


def test_component_statistics_p300(demo):
    """Rhino-style component analysis finds the simulated P300 (stimulus vs fixation)."""
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    ev = S.detect_gaze_events(rec.gaze)
    raw = C.run_preprocessing(rec.raw, C.PreprocParams(l_freq=0.1, h_freq=30))
    tab = S.event_table(rec, ev)
    eps, summ = C.build_group_epochs(
        raw, {"stim": tab[tab.name == "stimulus"], "fix": tab[tab.name == "fixation"].iloc[::4]},
        C.EpochSettings(-0.2, 0.8, None, delay_ms=0))
    assert "rejected by" in summ
    eps = {g: e.copy().apply_baseline((-0.2, 0)) for g, e in eps.items()}
    comp = C.COMPONENT_PRESETS["P300"]
    s, per = C.component_analysis(eps, comp)
    stim = s[s.group == "stim"].iloc[0]
    assert 250 <= stim["peak latency ms"] <= 500 and stim["peak µV"] > 0
    diff = s.iloc[-1]
    assert diff["group"] == "stim − fix" and diff["p"] < 0.01 and diff["window mean µV"] > 0
    assert set(per.group) == {"stim", "fix"} and len(per) == sum(len(e) for e in eps.values())
    with pytest.raises(ValueError, match="None of"):
        C.component_analysis(eps, C.Component("x", 0.1, 0.2, ("Nope",), 1))


def test_two_edfs_with_trigger_channel(tmp_path):
    """EDF files joined; a trigger channel becomes events with session/run columns."""
    import mne
    from brainstudio.io import load_many
    sf = 256
    paths = []
    for ses in ("1", "2"):
        n = sf * 30
        data = np.random.default_rng(int(ses)).normal(0, 5e-6, (3, n))
        trig = np.zeros(n)
        for k, t in enumerate(np.arange(2, 28, 3)):
            trig[int(t * sf):int(t * sf) + 10] = 1 + k % 2
        raw = mne.io.RawArray(np.vstack([data, trig]),
                              mne.create_info(["Fz", "Cz", "Pz", "TRIG"], sf,
                                              ["eeg", "eeg", "eeg", "misc"]), verbose="error")
        p = tmp_path / f"sub-07_ses-{ses}_task-odd_run-1_eeg.edf"
        mne.export.export_raw(p, raw, fmt="edf", physical_range=(-1, 3), overwrite=True,
                              verbose="error")
        paths.append(p)
    rec = load_many(paths)
    assert rec.raw.times[-1] == pytest.approx(60, abs=0.1)
    ev = C.trigger_codes(rec.raw, "TRIG")
    assert len(ev) == 18 and set(ev.code) == {1, 2}
    rec.events = pd.DataFrame({"onset": ev.onset, "duration": 0.0,
                               "description": [f"code{c}|trigger_code={c}" for c in ev.code]})
    tab = S.event_table(rec)
    t1 = tab[tab.name == "code1"]
    assert sorted(set(t1.session)) == ["1", "2"] and len(t1) == 10


def test_gtec_unicorn_layout():
    import mne
    m = C.montage_from_source("gtec_unicorn")
    assert m.ch_names == ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
    extra = ["ACC X", "GYR Z", "Battery", "Counter", "Validation"]
    raw = mne.io.RawArray(np.zeros((13, 10)), mne.create_info(
        [f"EEG {i}" for i in range(1, 9)] + extra, 250.0, "eeg"), verbose="error")
    rep = C.apply_montage(raw, m, list(m.ch_names) + extra)
    assert rep["n_positioned"] == 8 and not rep["missing"]
    assert set(raw.get_channel_types(extra)) == {"misc"}


def test_bad_channels_and_ica_ignore_data_before_first_event(demo):
    """Noise before the first event marks no channel bad when that part is ignored,
    and ICA is fitted from the first event on (blinks still align)."""
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    ev = S.detect_gaze_events(rec.gaze)
    t0 = C.first_event_time(S.event_table(rec, ev))
    assert t0 == pytest.approx(3.0, abs=0.01)
    raw = rec.raw.copy().load_data()
    n0 = int(t0 * raw.info["sfreq"])
    raw._data[raw.ch_names.index("Cz"), :n0] += np.random.default_rng(0).normal(0, 2e-3, n0)
    p = C.BadChannelParams(max_bad_fraction=0.02)  # 3 noisy s of 120 = 2.5 % of the windows
    assert "Cz" in set(C.detect_bad_channels(raw, p).query("bad").channel)
    assert "Cz" not in set(C.detect_bad_channels(raw, p, tmin=t0).query("bad").channel)
    ica, fit_raw = C.fit_ica(raw, [], 40.0, None, C.ICAParams(n_components=8), tmin=t0)
    assert C.fit_times(fit_raw)[0] == pytest.approx(t0, abs=0.01)
    assert C.score_blink_components(ica, fit_raw, ev.blinks).suggested.any()


def test_iir_highpass_with_boundary_at_data_edge():
    """A boundary annotation at the very start/end used to crash MNE's IIR path
    (0-sample stretch); the result must equal MNE's filter without the annotation."""
    import mne
    sf = 250
    x = np.random.default_rng(0).normal(0, 1e-5, (3, sf * 20))
    raw = mne.io.RawArray(x, mne.create_info(["Fz", "Cz", "Pz"], sf, "eeg"), verbose="error")
    ref = raw.copy().filter(0.1, None, method="iir", iir_params=C.IIR_PARAMS,
                            verbose="error").filter(None, 40, verbose="error")
    for onset in (0.0, raw.times[-1] + 1 / sf):
        r = raw.copy()
        r.set_annotations(mne.Annotations([onset], [0], ["EDGE boundary"]))
        out = C.run_preprocessing(r, C.PreprocParams(l_freq=0.1, h_freq=40, notch=None))
        assert np.allclose(out.get_data(), ref.get_data(), atol=1e-12)


def test_join_recordings_with_different_bad_channels():
    import mne

    from brainstudio.io import concatenate
    from brainstudio.model import Recording
    info = mne.create_info(["Fz", "Cz", "Pz"], 100.0, "eeg")
    a = mne.io.RawArray(np.zeros((3, 500)), info, verbose="error")
    b = a.copy()
    a.info["bads"], b.info["bads"] = ["Cz"], ["Pz"]
    joined = concatenate([Recording(raw=a), Recording(raw=b)])
    assert joined.raw.info["bads"] == ["Cz", "Pz"] and joined.raw.n_times == 1000
    assert "bad channels combined" in joined.description


def test_bids_dataset_info_bads_and_positions(tmp_path):
    """BIDS: bad channels (with reasons) from channels.tsv, dataset/sidecar info, and the
    subject-level space-…_electrodes.tsv found for every run."""
    from brainstudio.io import load_bids_runs
    root = tmp_path / "bids"
    write_bids(root, subjects=("01",))
    ch = next(root.rglob("*_channels.tsv"))
    df = pd.read_csv(ch, sep="\t", dtype=str, keep_default_na=False)
    df.loc[df.name == "Cz", ["status", "status_description"]] = ["bad", "broken lead"]
    df.loc[df.name == "Pz", "status"] = "bad"
    df.to_csv(ch, sep="\t", index=False)
    run = scan_bids(root)[0]
    rec = load_bids_runs(root, [run])
    assert set(rec.raw.info["bads"]) == {"Cz", "Pz"}
    assert rec.metadata["bad_channels"]["Cz"] == "broken lead"
    assert rec.metadata["dataset"]["Name"] == "brAInStudio synthetic free-viewing"
    assert "Bad channels (channels.tsv): Pz, Cz (broken lead)" in rec.description
    assert rec.metadata["montage_source"]  # template or electrodes file
    # electrodes.tsv at subject level with a space- entity (no task/run)
    el = run.eeg.with_name("sub-01_space-CapTrak_electrodes.tsv")
    el.write_text("name\tx\ty\tz\nCz\t0\t0\t0.1\n")
    assert __import__("brainstudio.core", fromlist=["x"]).electrodes_file(run.eeg) == el


def test_ica_uses_the_kept_channels_not_the_file_bad_list(demo):
    """Channels the file marks bad but the user keeps must be in the ICA fit."""
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = rec.raw.copy().load_data()
    raw.info["bads"] = raw.ch_names[:20]          # e.g. from channels.tsv
    ica, fit_raw = C.fit_ica(raw, bads=[], h_freq=40.0, p=C.ICAParams(n_components=20))
    assert ica.n_components_ == 20 and len(ica.ch_names) > 20
    assert fit_raw.info["bads"] == []


def test_ica_fit_is_bounded_on_long_data(demo):
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    ica, fit_raw = C.fit_ica(rec.raw, ["HEOG", "VEOG"], 40.0, None,
                             C.ICAParams(n_components=8, max_samples=5000))
    assert ica.n_components_ == 8
    assert "fitted on every" in fit_raw.fit_note       # subsampled over the whole recording
    assert fit_raw.n_times > 5000                        # scoring still sees all the data


def test_bids_events_tsv_columns_and_column_filters(tmp_path):
    """Every events.tsv column becomes a filterable event column; column conditions."""
    f = tmp_path / "sub-01_task-x_run-1_events.tsv"
    f.write_text("onset\tduration\tSOA\tlabel\ttrial_type\tanswering_time\tsample\n"
                 "1.0\t0.9\t0.5\thuman\teerie\t1.5\t256\n"
                 "4.5\t1.0\t0.4\tanimated\tnot_eerie\t2.8\t1160\n"
                 "8.3\t1.1\t0.5\thuman\tnot_eerie\tn/a\t2128\n")
    ev = C.bids_events(f)
    assert list(ev.onset) == [1.0, 4.5, 8.3]
    assert ev.description[0] == "eerie|SOA=0.5|label=human|answering_time=1.5"
    rec = __import__("brainstudio.model", fromlist=["x"]).Recording(events=ev)
    t = S.event_table(rec)
    assert {"soa", "label", "answering_time"} <= set(t.columns)
    human = C.filter_events(t, conditions=[("label", "=", "human")])
    assert list(human.name) == ["eerie", "not_eerie"]
    fast = C.filter_events(t, conditions=[("label", "=", "human"),
                                           ("answering_time", "<", "2")])
    assert list(fast.name) == ["eerie"]
    assert len(C.filter_events(t, conditions=[("label", "one of", "human, animated")])) == 3
    with pytest.raises(ValueError, match="compares numbers"):
        C.filter_events(t, conditions=[("label", "<", "x")])


def test_epoch_summary_counts_each_reason():
    import mne
    sf = 100.0
    x = np.random.default_rng(0).normal(0, 1e-6, (2, int(sf * 60)))
    x[0, int(30.3 * sf)] = 1e-3                       # spike inside the epoch at 30 s
    raw = mne.io.RawArray(x, mne.create_info(["Cz", "Pz"], sf, "eeg"), verbose="error")
    raw.set_annotations(mne.Annotations([40.0], [1.0], ["BAD_acq_gap"]))
    onsets = [0.05,                                    # window starts before the recording
              10.0, 10.0,                              # two events on the same sample
              30.0,                                    # amplitude (spike on Cz)
              40.2,                                    # inside a BAD span
              ] + [12.0 + k for k in range(8)]         # plain events
    table = pd.DataFrame({"onset": onsets, "duration": 0.0, "name": "e", "kind": "marker",
                          "description": "e"})
    _, summ = C.build_group_epochs(raw, {"g": table},
                                    C.EpochSettings(-0.2, 0.8, reject_uv=100,
                                                     autoreject=False))
    r = summ.iloc[0]
    assert r["selected"] == 13 and r["outside recording"] == 1 and r["same sample"] == 1
    assert r["rejected (BAD annotation)"] + r["rejected (amplitude)"] + r["kept"] == 11
    assert r["kept"] + r["outside recording"] + r["same sample"] \
        + r["rejected (amplitude)"] + r["rejected (BAD annotation)"] == r["selected"]
    assert "over cap" not in summ
    assert "outside/dup" not in summ


def test_erp_reference_choices(demo):
    """Each component selects its usual reference; mastoids fall back to as recorded."""
    import mne
    assert C.COMPONENT_REFERENCE["P300"] == "mastoids" and C.COMPONENT_REFERENCE["N170"] == \
        "average"
    assert C.resolve_erp_reference("mastoids", ["Cz", "TP9", "TP10"])[0] == ["TP9", "TP10"]
    ref, label = C.resolve_erp_reference("mastoids", ["Fz", "Cz", "Pz"])
    assert ref is None and label.startswith("as recorded") and "no good mastoid" in label
    assert C.resolve_erp_reference("average", ["Cz"])[0] == "average"
    assert C.resolve_erp_reference("recorded", ["Cz"])[0] is None
    with pytest.raises(ValueError, match="not among"):
        C.resolve_erp_reference("M1, M2", ["Cz", "Pz"])
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = rec.raw.copy().load_data().pick("eeg")
    tab = S.event_table(rec)
    eps, _ = C.build_group_epochs(raw, {"s": tab[tab.name == "stimulus"]},
                                   C.EpochSettings(-0.2, 0.8, None))
    avg = C.reference_epochs(eps["s"], "average")
    assert np.allclose(avg.get_data().mean(1), 0, atol=1e-12)        # average reference
    assert C.reference_epochs(eps["s"], None) is eps["s"]             # as recorded
    assert np.allclose(eps["s"].get_data().mean(1), eps["s"].get_data().mean(1))
    pair = C.reference_epochs(eps["s"], ["T7", "T8"])
    assert np.allclose(pair.get_data(picks=["T7"]) + pair.get_data(picks=["T8"]), 0,
                       atol=1e-12)                                    # mean of T7/T8 = 0
    _ = mne


def test_band_power_baseline_is_per_subject():
    """Epochs are normalised against their own subject's baseline, never pooled."""
    # baseline events: subject a at 0 s and 100 s, subject b at 200 s
    base = C.BandBaseline(np.array([0.0, 100.0, 200.0]), 0, 10, "mean", "rest",
                           np.array(["a", "a", "b"]))
    owner = np.array([0, 0, 1, 1, 2, 2])                       # 2 segments per event
    base_bp = np.array([1.0, 1.0, 3.0, 3.0, 10.0, 10.0])[:, None, None]  # a: 1 and 3, b: 10
    onsets = np.array([50.0, 150.0, 250.0, 260.0])
    subj = np.array(["a", "a", "b", "c"])                      # c has no baseline
    mean, sd = C.reference_for(onsets, base, base_bp, owner, subj)
    assert mean[:3, 0, 0].tolist() == [2.0, 2.0, 10.0]          # own subject's mean only
    assert np.isnan(mean[3, 0, 0])                              # no borrowing across subjects
    base.mode = "nearest"
    mean, _ = C.reference_for(onsets, base, base_bp, owner, subj)
    assert mean[:3, 0, 0].tolist() == [1.0, 3.0, 10.0]          # nearest event of the subject
    assert C.subjects_without_baseline(base, subj) == ["c"]
    pooled, _ = C.reference_for(onsets, C.BandBaseline(base.onsets, 0, 10), base_bp, owner)
    assert np.allclose(pooled[:, 0, 0], np.mean([1, 1, 3, 3, 10, 10]))  # no labels: pooled


def test_group_erp_averages_subjects_first():
    """Grand average = mean of per-subject averages, whatever each subject's epoch count;
    the across-subject test is a paired t-test on the subject means."""
    from scipy import stats as sst
    rng = np.random.default_rng(1)
    info = mne.create_info(["Cz", "Pz"], 100.0, "eeg")
    groups = {}
    for g, amp in (("A", 4e-6), ("B", 1e-6)):
        data, subj = [], []
        for s, n in (("01", 5), ("02", 40), ("03", 12), ("04", 9)):
            base = amp * (1 + 0.3 * int(s))
            x = rng.normal(0, 1e-6, (n, 2, 80))
            x[:, :, 40:60] += base
            data.append(x)
            subj += [s] * n
        ep = mne.EpochsArray(np.concatenate(data), info, tmin=-0.2, verbose="error")
        ep.metadata = pd.DataFrame({"subject": subj})
        groups[g] = ep
    comp = C.Component("X", 0.2, 0.39, ("Cz", "Pz"), +1)
    res = C.group_component_analysis(groups, comp)
    subs = list(res["evokeds"]["A"].values())
    assert np.allclose(res["grand"]["A"].data, np.mean([e.data for e in subs], 0))
    # pooled epochs would weigh subject 02 (40 epochs) eight times more than subject 01
    pooled = groups["A"].average().data
    assert not np.allclose(res["grand"]["A"].data, pooled)
    sm = res["summary"].set_index("group")
    assert sm.loc["A", "subjects"] == 4
    per = res["subjects"].pivot(index="subject", columns="group", values="window mean µV")
    t = sst.ttest_rel(per["A"], per["B"])
    assert sm.loc["A − B", "t"] == pytest.approx(t.statistic)
    assert sm.loc["A − B", "p"] == pytest.approx(t.pvalue)
    assert len(res["diffs"]) == 4 and len(res["subject_tests"]) == 4


def test_subject_spans_merge_runs():
    from brainstudio.core import segment_label, subject_spans
    segs = [{"start": 0, "end": 10, "subject": "01", "run": "1"},
            {"start": 10, "end": 20, "subject": "01", "run": "2"},
            {"start": 20, "end": 35, "subject": "02", "run": "1"}]
    spans = subject_spans(segs)
    assert [(s["label"], s["start"], s["end"]) for s in spans] == [
        ("sub-01", 0.0, 20), ("sub-02", 20.0, 35)]
    assert segment_label(segs[1]) == "sub-01 · run-2"
    assert subject_spans([{"start": 0, "end": 5}]) == []


def test_script_literals_are_exact_python():
    from brainstudio import script as SC
    spec = C.GroupSpec("A", ["stimulus", "x" * 40], query="cond == 'AI'",
                       conditions=[["subject", "=", "01"]],
                       recordings={"file": [f"sub-{i:02d}_task-x_eeg.vhdr" for i in range(9)]})
    code = SC.assign("spec = ", spec)
    assert max(len(line) for line in code.splitlines()) <= SC.WIDTH
    ns = {"GroupSpec": C.GroupSpec}  # scripts hold the class itself, no module prefix
    exec(code, ns)
    assert ns["spec"] == spec
    assert SC.lit(np.float64(0.1)) == "0.1" and SC.lit(np.int64(3)) == "3"


def test_exported_script_repeats_the_erp(tmp_path):
    """The short exported ERP script (standard libraries only, no brAInStudio) builds the
    same epochs and per-subject ERP values as the app."""
    import runpy
    import sys
    from brainstudio import io as bio
    from brainstudio import script as SC
    from brainstudio.demo import write_bids
    root = tmp_path / "bids"
    write_bids(root, subjects=("01", "02"))
    runs = bio.scan_bids(root)
    rec = bio.load_bids_runs(root, runs, None)
    step = {"kind": "load_bids", "root": str(root), "runs": [list(r.key) for r in runs],
            "screen_size": None}
    step["recording"] = S.recording_info(rec, step)
    layout, _ = C.read_channel_layout(C.layout_montages()["gtec_64"])
    names = layout[:len(rec.raw.ch_names)]
    C.apply_montage(rec.raw, C.montage_from_source("gtec_64"), names)  # a montage step
    det = C.DetectionParams()
    gev = S.detect_gaze_events(rec.gaze, det)
    table = S.event_table(rec, gev, None)
    ip = C.ICAParams(n_components=8, clean="threshold", iclabel=False)
    ica, fit_raw = C.fit_ica(rec.raw, [], None, 50.0, ip)
    pp = C.PreprocParams(ica_exclude=[0])
    clean = C.run_preprocessing(rec.raw, pp, ica)
    specs = [C.GroupSpec("stim", ["stimulus"]), C.GroupSpec("resp", ["response"])]
    st = C.EpochSettings(autoreject=False)
    epochs, _ = C.build_group_epochs(
        clean, {g.label: C.select_events(table, g) for g in specs}, st, gev.blinks)
    spec = C.ErpSpec(["stim", "resp"], C.COMPONENT_PRESETS["P300"], (-0.1, 0.0), "average")
    res = C.run_erp(epochs, spec)
    load = {"steps": [step, {"kind": "montage", "source": "gtec_64", "names": names}],
            "gaze": {"params": det, "aois": {}}}
    recipe = {"load": load, "groups": specs, "settings": st,
              "pre": {"load": load, "params": pp, "ica": {"steps": fit_raw.fit_steps}}}
    path = tmp_path / "erp.py"
    code = SC.build_script("erp", recipe, spec, {"ch_names": list(epochs["stim"].ch_names),
                                                 "n_subjects": 2})
    path.write_text(code, encoding="utf-8")
    body = code.split('"""', 2)[2]
    assert "brainstudio" not in body and len(code.splitlines()) < 260
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "brainstudio"}
    for k in saved:  # the script must not need brainstudio
        sys.modules[k] = None
    import matplotlib
    matplotlib.use("Agg")
    try:
        g = runpy.run_path(str(path), run_name="exported")
    finally:
        sys.modules.update(saved)
    assert {k: len(v) for k, v in g["epochs"].items()} == {k: len(v) for k, v in epochs.items()}
    app = res["group"]["subjects"].pivot(index="subject", columns="group",
                                         values="window mean µV")
    assert np.allclose(g["means"].loc[app.index, app.columns], app, rtol=0, atol=1e-9)


def test_text_files_are_read_and_written_as_utf8():
    """Windows' default encoding (cp1252) cannot decode the package's own sources, which
    exported scripts copy from: every text read/write must name its encoding."""
    import re
    from pathlib import Path
    root = Path(C.__file__).parent
    bad = []
    for f in root.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.(read_text|write_text)\(", line) and "encoding=" not in line:
                bad.append(f"{f.relative_to(root)}:{i}")
    assert not bad, bad


def test_gaze_preset_follows_the_data():
    """Gazepoint gaze with only the fixation-filtered POG gets the Gazepoint preset."""
    from brainstudio.model import GazeData
    t = np.arange(100) / 150.0
    fpog = GazeData(t, np.zeros(100), np.zeros(100), extra={"fpogx": t, "fpogy": t})
    best = GazeData(t, np.zeros(100), np.zeros(100), extra={"fpogx": t, "bpogx": t})
    assert S.suggest_preset(fpog) == "gazepoint_fpog"
    assert S.suggest_preset(best) == "standard"
    assert C.preset_name(C.PRESETS["gazepoint_fpog"][1]) == "gazepoint_fpog"
    assert C.preset_name(C.DetectionParams(fix_dispersion=55.0)) is None
