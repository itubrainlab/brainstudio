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
    # (events.tsv is the reference: its columns, e.g. stim_file, are kept)
    name = ev.description.str.split("|").str[0]
    assert (name == "stimulus").sum() == len(simulate(seed=0).markers) // 2
    assert ev.description.str.contains("stim_file=").sum() == (name == "stimulus").sum()


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


def test_exported_script_reads_multicolumn_markers(tmp_path):
    """The exported script pairs start/stop rows of a multi-column marker stream like the
    app, so groups of paired events ('trials') select the same epochs."""
    import runpy
    import sys

    from brainstudio import script as SC
    eeg_t = np.arange(0, 30, 1 / 100)
    rows = [["start", "trials", "clip1.mp4", "sad", ""], ["stop", "trials", "clip1.mp4", "sad",
                                                          "angry"],
            ["start", "trials", "clip2.mp4", "happy", ""], ["end", "trials", "clip2.mp4",
                                                            "happy", "happy"]]
    path = tmp_path / "m.xdf"
    write_xdf(path, [
        dict(name="EEG", type="EEG", srate=100, format="float32", labels=["Cz", "Pz"],
             timestamps=eeg_t, data=np.random.default_rng(0).normal(0, 10, (len(eeg_t), 2))),
        dict(name="exp_stim", type="Markers", srate=0, format="string",
             labels=["c1", "c2", "c3", "c4", "c5"],
             timestamps=np.array([2.0, 5.0, 10.0, 12.5]), data=rows)])
    rec = load_xdf(path)
    step = {"kind": "load", "path": str(path), "screen_size": (1920, 1080), "roles": None,
            "options": None}
    step["recording"] = S.recording_info(rec, step)
    table = S.event_table(rec)
    specs = [C.GroupSpec("A", ["trials"]), C.GroupSpec("B", ["trials_end"])]
    st = C.EpochSettings(autoreject=False)
    epochs, _ = C.build_group_epochs(rec.raw, {g.label: C.select_events(table, g)
                                               for g in specs}, st)
    spec = C.ErpSpec(["A", "B"], C.COMPONENT_PRESETS["P300"], (-0.1, 0.0), "recorded")
    recipe = {"load": {"steps": [step], "gaze": {"params": None, "aois": {}}},
              "groups": specs, "settings": st, "pre": None}
    code = SC.build_script("erp", recipe, spec, {"ch_names": ["Cz", "Pz"], "n_subjects": 1})
    (tmp_path / "erp.py").write_text(code, encoding="utf-8")
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "brainstudio"}
    for k in saved:  # the script must not need brainstudio
        sys.modules[k] = None
    import matplotlib
    matplotlib.use("Agg")
    try:
        g = runpy.run_path(str(tmp_path / "erp.py"), run_name="exported")
    finally:
        sys.modules.update(saved)
    assert {k: len(v) for k, v in g["epochs"].items()} == {"A": 2, "B": 2} \
        == {k: len(v) for k, v in epochs.items()}
    ev = g["events"]
    assert list(ev.loc[ev.name == "trials", "duration"]) == pytest.approx([3.0, 2.5])
    assert list(ev.loc[ev.name == "trials", "c5"]) == ["angry", "happy"]


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


def test_bids_gazepoint_physio_next_to_a_face_video_stream(tmp_path):
    """A Gazepoint file (``recording-gaze``, sidecar ``Type: gaze``, an ``onset`` column,
    0–1 BPOGX/BPOGY, ``display_resolution``) is the eye tracker, not the OpenFace stream
    beside it (``Type: videostream``, gaze_0_x …); events.tsv at the subject level."""
    import gzip
    import json
    from pathlib import Path
    from brainstudio.io import load_bids_runs
    root = tmp_path / "bids"
    write_bids(root, subjects=("01",), stimuli=False)
    eeg = root / "sub-01" / "eeg"
    for f in eeg.glob("*recording-eye1*"):
        f.unlink()
    (eeg / "sub-01_task-freeview_events.tsv").rename(
        root / "sub-01" / "sub-01_task-freeview_events.tsv")
    beh = root / "sub-01" / "beh"
    beh.mkdir()
    n, rate = 600, 60.0
    t = np.arange(n) / rate
    x, y = 0.5 + 0.1 * np.sin(t), 0.5 + 0.1 * np.cos(t)
    gaze_cols = ["onset", "TIME", "BPOGX", "BPOGY", "BPOGV"]
    face_cols = ["onset", "frame", "gaze_0_x", "gaze_0_y", "gaze_angle_x"]
    for rec, kind, cols, rows in (
            ("gaze", "gaze", gaze_cols, np.c_[t, t + 100, x, y, np.ones(n)]),
            ("videostream", "videostream", face_cols,
             np.c_[t, np.arange(n), -5 + 0 * t, 1 + 0 * t, 0 * t])):
        stem = beh / f"sub-01_task-freeview_recording-{rec}_physio"
        with gzip.open(f"{stem}.tsv.gz", "wt") as fh:
            np.savetxt(fh, rows, delimiter="\t")
        meta = {"SamplingFrequency": rate, "StartTime": 0, "Columns": cols, "Type": kind,
                "display_resolution": "1280x1024"}
        Path(f"{stem}.json").write_text(json.dumps(meta))
    run = scan_bids(root)[0]
    assert [g.name for g in run.gaze] == ["sub-01_task-freeview_recording-gaze_physio.tsv.gz"]
    rec = load_bids_runs(root, [run])
    g = rec.gaze
    assert tuple(g.screen_size) == (1280, 1024)
    assert np.allclose(g.times, t) and np.allclose(g.x, x * 1280) and np.allclose(g.y, y * 1024)
    assert len(rec.all_events()) > 0


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
    # three groups: repeated-measures ANOVA on the subjects that have every group
    three = C.group_component_analysis({**groups, "C": groups["A"].copy()}, comp)
    last = three["summary"].iloc[-1]
    assert last["test"] == "repeated-measures ANOVA" and 0 <= last["p"] <= 1


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


def test_group_labels_from_events_and_filters():
    """Groups are named after their events and filters, uniquely; typed names stay."""
    from brainstudio import script as SC
    G = C.GroupSpec
    assert C.group_label(G("", ["stimulus"])) == "stimulus"
    assert C.group_label(G("", ["stimulus", "response"])) == "response/stimulus"
    assert C.group_label(G("", list("abcde"))) == "a/b/c +2"
    assert C.group_label(G("", [])) == "no events"
    assert C.group_label(G("", ["stimulus"], conditions=[["emotion", "=", "happy"]])) \
        == "stimulus · emotion=happy"
    assert C.group_label(G("", ["stimulus"], conditions=[
        ["rt", "<", "2"], ["emotion", "one of", "happy, sad"], ["word", "contains", "ab"],
        ["x", "=", ""]])) == "stimulus · rt<2 · emotion∈{happy,sad} · word~ab"
    assert C.group_label(G("", ["stimulus"], text="face", query="rt  < 2")) \
        == "stimulus · /face/ · rt < 2"
    # recordings are the same for every group: not part of the name
    assert C.group_label(G("x", ["stimulus"], recordings={"subject": ["01"]})) == "stimulus"
    long = C.group_label(G("", ["stimulus"], query="a == 1 and " * 20))
    assert len(long) == C.GROUP_LABEL_MAX and long.endswith("…")
    specs = [G("Group A", ["stimulus"]), G("stimulus", ["response"]), G("Group C", ["stimulus"]),
             G("mine", ["blink"])]
    labels = C.auto_group_labels(specs, [True, False, True, False])
    assert labels == ["stimulus (2)", "stimulus", "stimulus (3)", "mine"]
    assert C.auto_group_labels(specs[:1] + specs[2:3], [True, True]) == \
        ["stimulus", "stimulus (2)"]
    # automatic labels are plain keys: they survive the exported scripts' literals
    spec = G(labels[0], ["stimulus"], conditions=[["rt", "≤", "2"]])
    ns = {"GroupSpec": G}
    exec(SC.assign("spec = ", spec), ns)
    assert ns["spec"] == spec


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


def test_bids_stimuli_follow_the_events(tmp_path):
    """Stimuli named by the events (BIDS stim_file by default, or any column mapped to the
    files) are on the timeline from the event's onset; video frames match the time."""
    from brainstudio.demo import write_bids
    root = write_bids(tmp_path / "bids", subjects=("01",))
    rec = load_bids_run(root, scan_bids(root)[0])
    ses = S.Session()
    ses.set_recording(rec)
    tab = ses.event_table
    # events.tsv kept (its stim_file column); tracker messages of the same events dropped
    assert (tab.name == "stimulus").sum() == 19 and "stim_file" in tab
    assert ses.stimulus_column == "stim_file" and len(ses.stimulus_files) == 4
    st = ses.stimuli
    assert len(st) == 19 and set(st.kind) == {"image", "video"}
    first = tab[tab.name == "stimulus"].onset.iloc[0]
    assert st.onset.iloc[0] == pytest.approx(first)
    assert (st.end.to_numpy()[:-1] == st.onset.to_numpy()[1:]).all()  # until the next one
    assert C.stimulus_at(st, first - 0.01) is None
    v = st[st.kind == "video"].iloc[0]
    assert C.stimulus_at(st, v.onset + 1.0)["file"] == "videos/clip1.mp4"
    reader = C.StimulusReader()
    for t in (1.0, 0.2, 2.0):  # backwards and forwards: the dot moves 7 px per frame
        img = reader.frame(v.path, "video", t)
        x = np.flatnonzero(img[180].max(1) > 200).mean()
        assert x == pytest.approx(40 + int(round(t * 25)) * 7, abs=8)
    assert C.stimulus_rect((720, 1280), (1920, 1080)) == (320.0, 180.0, 1280.0, 720.0)
    assert C.stimulus_rect((2160, 3840), (1920, 1080)) == (0.0, 0.0, 1920.0, 1080.0)
    # the user maps another column: every "response" shows one picture, for 0.5 s
    ses.set_stimuli(ses.stimulus_folder, "name", {"response": "images/scene2.png"})
    st = ses.stimuli
    assert len(st) == (tab.name == "response").sum() and set(st.file) == {"images/scene2.png"}


def test_stimulus_column_and_value_matching():
    """Event values name files by path, file name or name without extension; numbers
    match too (82 → 82.png), and the column naming the most files is proposed."""
    files = ["2.png", "82.png", "faces/smile.jpg", "clip.mp4"]
    assert C.match_stimuli(["82", "smile", "faces/smile.jpg", "CLIP.MP4", "7"], files) == {
        "82": "82.png", "smile": "faces/smile.jpg", "faces/smile.jpg": "faces/smile.jpg",
        "CLIP.MP4": "clip.mp4"}
    tab = pd.DataFrame({"onset": [1.0, 2.0, 3.0], "duration": [0.9, 1.0, 0.0],
                        "kind": "marker", "name": "x",
                        "trial_number": [1, 2, 3], "image_id": [82.0, 2.0, 82.0]})
    assert C.guess_stimulus_column(tab, files) == "image_id"
    st = C.stimulus_timeline(tab, "/data/stimuli", "image_id",
                             C.match_stimuli(["82", "2"], files), end=10.0)
    assert list(st.file) == ["82.png", "2.png", "82.png"]
    assert list(st.end) == pytest.approx([1.9, 3.0, 10.0])  # duration, else next / end
    assert C.guess_stimulus_column(tab.assign(stim_file=["a", "b", "c"]), files) == "stim_file"


def _gazepoint(rate, seconds=20.0, seed=0):
    """Gazepoint-like gaze: fixations of 0.3 s at random places (FPOG… columns) with
    noise, 0–1 coordinates."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    fid = (t // 0.3).astype(int)
    centre = rng.uniform(0.2, 0.8, (fid.max() + 1, 2))
    x = centre[fid, 0] + rng.normal(0, 0.003, len(t))
    y = centre[fid, 1] + rng.normal(0, 0.003, len(t))
    start = fid * 0.3
    data = np.c_[x, y, fid, t - start, centre[fid, 0], centre[fid, 1], np.ones(len(t))]
    return C.gaze_table(t, data, ["BPOGX", "BPOGY", "FPOGID", "FPOGD", "FPOGX", "FPOGY",
                                  "FPOGV"], (1000, 1000), sfreq=rate)


def test_gaze_rates_resampling_and_tracker_fixations():
    """A rate that disagrees with the timestamps (a period written as the rate) is
    replaced by the measured one; joined parts at different rates are upsampled to the
    highest (linear, gaps kept) and detected each on its own; the tracker's own fixations
    replace I-DT and do not depend on the rate."""
    t = np.arange(200) / 62.0
    assert C.checked_rate(1 / 62.0, t) == pytest.approx(62.0)
    assert C.checked_rate(60.0, t) == 60.0  # within 10 %: the stated rate
    # linear interpolation; a loss keeps its edges (the nearer sample decides)
    g = pd.DataFrame({"t": [0.0, 0.1, 0.2, 0.3], "x": [0.0, 10.0, np.nan, 30.0],
                      "y": [0.0, 1.0, 2.0, 3.0], "ch_id": [1.0, 1.0, 2.0, 2.0]})
    r = C.resample_gaze(g, 20.0)
    assert np.allclose(r["t"], np.arange(7) * 0.05)
    assert np.allclose(r["x"], [0, 5, 10, 10, np.nan, np.nan, 30], equal_nan=True)
    assert list(r["ch_id"]) == [1, 1, 1, 1, 2, 2, 2] and r.attrs["sfreq"] == 20.0
    # the tracker's fixations keep their start, end at the nearest new sample, and a
    # fixation valid on one sample only survives
    for slow in (_gazepoint(60.0), _gazepoint(60.0).assign(
            ch_fpogv=lambda d: np.where(d["ch_fpogd"] < 0.28, 0.0, 1.0))):
        native = C.tracker_fixations(slow, (1000, 1000))
        up = C.tracker_fixations(C.resample_gaze(slow, 150.0), (1000, 1000))
        assert len(up) == len(native) > 60
        assert np.allclose(up["onset"], native["onset"], atol=1e-9)
        assert np.allclose(up["offset"], native["offset"], atol=0.5 / 150 + 1e-9)
    # the tracker's fixations: 0.3 s each, at their place in pixels, at any rate
    for rate in (60.0, 150.0):
        fix = C.tracker_fixations(_gazepoint(rate), (1000, 1000))
        assert len(fix) == 67 and np.allclose(fix["duration"][:-1], 0.3, atol=1.5 / rate)
        assert fix["x"].between(200, 800).all()
    slow, fast = _gazepoint(60.0), _gazepoint(150.0, seed=1)
    raw, events, gaze, segments, notes = C.join(
        [(None, pd.DataFrame(columns=["onset", "duration", "description"]), g_, {})
         for g_ in (slow, fast)])
    assert any(n.startswith("gaze upsampled to 150.0 Hz") for n in notes)
    assert gaze.attrs["sfreq"] == pytest.approx(150.0)
    first = gaze[gaze["t"] < segments[1]["start"]]
    assert np.median(np.diff(first["t"])) == pytest.approx(1 / 150.0)
    ev = C.detect_gaze_events(gaze, C.DetectionParams(), [s["start"] for s in segments])
    assert ev.fixations.attrs["source"] == "eye tracker" and len(ev.fixations) == 2 * 67
    idt = C.detect_gaze_events(fast, C.DetectionParams(tracker_fixations=False))
    assert idt.fixations.attrs["source"] == "I-DT"


def test_event_filter_conditions():
    df = pd.DataFrame({"onset": [1.0, 2.0, 3.0, 4.0],
                       "name": ["stimulus", "response", "Stimulus", "BAD boundary"],
                       "subject": ["acl", "acl", "agk", ""]})
    f = lambda *conds, match="all": list(C.filter_rows(df, conds, match)["onset"])  # noqa: E731
    assert f((None, "contains", "stim")) == [1.0, 3.0]
    assert f((None, "contains", "STIM agk")) == [3.0]          # every word, any column
    assert f(("name", "contains", "acl")) == []
    assert f(("name", "is", "stimulus"), ("subject", "is not", "agk")) == [1.0]
    assert f(("onset", "≥", "2"), ("onset", "<", "4")) == [2.0, 3.0]
    assert f(("onset", "is", "3")) == [3.0]
    assert f(("name", "starts with", "bad"), ("subject", "is", "agk"), match="any") == [3.0, 4.0]
    assert f(("subject", "is empty", "")) == [4.0]
    assert f((None, "does not contain", "acl")) == [3.0, 4.0]
    assert f(("onset", ">", "x")) == [] and f(("name", "is", "  ")) == [1.0, 2.0, 3.0, 4.0]


def test_event_filter_applies_everywhere_and_in_scripts(tmp_path):
    """Applied, the event filter leaves out markers in the session's table (gaze events
    stay), in the markers shown, and in the exported script's epochs."""
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
    rec.gaze_events = S.detect_gaze_events(rec.gaze, C.DetectionParams(), rec.segments)
    rec.detected_with = C.DetectionParams()
    ses = S.Session()
    ses.set_recording(rec, step)
    marker = lambda t: t[t["kind"] == "marker"]  # noqa: E731
    n_all, n_gaze = len(marker(ses.event_table)), (ses.event_table["kind"] != "marker").sum()
    conds = [("onset", "≥", "30"), ("subject", "is", "02")]
    assert ses.set_event_filter(conds, "any") and not ses.set_event_filter(conds, "any")
    kept = marker(ses.event_table)
    assert 0 < len(kept) < n_all and (ses.event_table["kind"] != "marker").sum() == n_gaze
    assert ((kept["onset"] >= 30) | (kept["subject"] == "02")).all()
    assert len(ses.shown_events()) == len(kept) < len(rec.all_events())
    assert len(marker(ses.event_table_all)) == n_all
    specs = [C.GroupSpec("stim", ["stimulus"]), C.GroupSpec("resp", ["response"])]
    st = C.EpochSettings(autoreject=False)
    epochs, _ = C.build_group_epochs(
        rec.raw, {g.label: C.select_events(ses.event_table, g) for g in specs}, st,
        ses.gaze_events.blinks)
    spec = C.ErpSpec(["stim", "resp"], C.COMPONENT_PRESETS["P300"], (-0.1, 0.0), "average")
    recipe = {"load": ses.load_recipe(), "groups": specs, "settings": st, "pre": None}
    assert SC.check_recipe(recipe) is None
    code = SC.build_script("erp", recipe, spec, {"ch_names": list(epochs["stim"].ch_names),
                                                 "n_subjects": 2})
    path = tmp_path / "erp.py"
    path.write_text(code, encoding="utf-8")
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
    unfiltered = {g_.label: len(C.select_events(ses.event_table_all, g_)) for g_ in specs}
    assert all(len(epochs[k]) < unfiltered[k] for k in unfiltered)  # the filter mattered
    bad = dict(recipe, load=dict(recipe["load"], event_filter=C.event_filter(
        [("phase", "is", "x")], "all", ses.event_table_all)))
    assert "phase" in SC.check_recipe(bad)


def test_preprocessing_per_subject_matches_the_exported_script(tmp_path):
    """Each subject is preprocessed on its own (its bad channels, its ICA, its removed
    components), the same as preprocessing that subject alone; the exported script
    (standard libraries only) gives the same data."""
    from brainstudio import io as bio
    from brainstudio import script as SC
    root = tmp_path / "bids"
    write_bids(root, subjects=("01", "02"), stimuli=False)
    runs = bio.scan_bids(root)
    rec = bio.load_bids_runs(root, runs, None)
    step = {"kind": "load_bids", "root": str(root), "runs": [list(r.key) for r in runs],
            "screen_size": None}
    ses = S.Session()
    ses.set_recording(rec, step)
    raw = rec.raw
    assert [s.label for s in ses.subjects] == ["sub-01", "sub-02"]
    assert ses.subjects[1].part["start"] == ses.subjects[0].part["stop"] > 0
    ses.subjects[0].bads = [raw.ch_names[3]]  # a different bad channel per subject
    ses.subjects[1].bads = [raw.ch_names[7]]
    ip = C.ICAParams(n_components=5, clean="threshold", iclabel=False)
    fits = {}
    for sub in ses.subjects:
        ica, fit_raw = C.fit_ica(C.part_raw(raw, sub.part), sub.bads, None, None, ip,
                                 tmin=ses.subject_first_event(sub))
        assert len(fit_raw.times) < raw.n_times / 2  # this subject's data only
        fits[sub.key] = (ica, fit_raw, C.score_blink_components(ica, fit_raw), dict(
            bads=list(sub.bads), params=ip, steps=dict(fit_raw.fit_steps)))
    ses.set_icas(fits)
    ses.subjects[0].exclude, ses.subjects[1].exclude = [0], [1, 2]
    p = C.PreprocParams(l_freq=1.0, h_freq=30.0, notch=None, interpolate_bads=True)
    settings = ses.subject_settings()
    p.per_subject = {k: {"bads": v["bads"], "exclude": v["exclude"]}
                     for k, v in settings.items()}
    parts = [s.part for s in ses.subjects]
    clean = C.preprocess_subjects(raw, parts, p, settings)
    assert clean.info["bads"] == [] and "2 subjects" in p.describe()
    for sub in ses.subjects:  # each subject as if it were preprocessed alone
        alone = C.run_preprocessing(C.part_raw(raw, sub.part),
                                    C.replace(p, bads=sub.bads, ica_exclude=sub.exclude),
                                    sub.ica)
        a, b = sub.part["start"], sub.part["stop"]
        assert np.array_equal(alone.get_data(), clean.get_data()[:, a:b])
    before, after = C.psd_check_subjects(raw, clean, parts)
    assert before.get_data().shape == after.get_data().shape
    # the exported script's loading and preprocessing cells give the same data
    step["recording"] = S.recording_info(rec, step)
    recipe = {"load": {"steps": [step], "gaze": {"params": None, "aois": {}}},
              "pre": ses.snapshot_pre_recipe(p, settings)}
    load, imp = SC._loading(recipe)
    pre, imp2 = SC._preprocessing(recipe)
    code = "\n".join([SC._IMPORT[m] for m in sorted(imp | imp2) if m in SC._IMPORT]
                     + ["import json", "import warnings", "from pathlib import Path",
                        "mne.set_log_level('error')"] + load + pre)
    assert "SUBJECTS" in code and "brainstudio" not in code
    g = {}
    exec(compile(code, "exported", "exec"), g)
    assert np.allclose(g["data"].get_data(), clean.get_data(), rtol=0, atol=1e-12)
    edges = lambda r: sorted(zip(*[np.round(r.annotations.onset, 6), r.annotations.description]))  # noqa: E731
    script_edges = [e for e in edges(g["data"]) if C.is_data_annotation(pd.Series([e[1]]))[0]]
    assert script_edges == edges(clean)  # the joins' boundaries once, as in the app
