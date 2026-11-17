"""Tests on synthetic data (BIDS + XDF round trips, gaze detection, FRPs)."""

import numpy as np
import pytest

from brainstudio import eeg as E
from brainstudio import gaze as G
from brainstudio.demo import simulate, write_bids, write_xdf_demo
from brainstudio.io import load_bids_run, load_xdf, scan_bids
from brainstudio.io.xdf import _clean_timestamps
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
    fix = G.detect_fixations(g)
    err = np.abs(fix.onset.to_numpy()[:, None] - sim.fix_onsets[None]).min(1)
    assert np.median(err) < 0.03
    assert 0.8 < len(fix) / len(sim.fix_onsets) < 1.4


def test_aoi_stats_use_pygaze():
    import pandas as pd
    fix = pd.DataFrame({"onset": [0, 1], "offset": [.2, 1.3], "duration": [.2, .3],
                        "x": [100, 900], "y": [100, 500]})
    st = G.aoi_statistics(fix, {"a": G.make_aoi("rectangle", (50, 50), (100, 100))})
    assert st.n_fixations.iloc[0] == 1 and st.dwell_time.iloc[0] == pytest.approx(.2)


def test_frp_recovers_lambda_response(demo):
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = E.preprocess(rec.raw, E.FilterParams(1, 30))
    ev = G.detect_events(rec.gaze)
    evk = E.fixation_related_epochs(raw, ev.fixations, min_duration=0.1).average()
    oz = evk.copy().pick(["Oz"]).data[0]
    assert 0.06 < evk.times[np.argmax(oz)] < 0.14  # simulated P1 at ~90 ms


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

    from brainstudio import events as EV
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
    tab = EV.build_event_table(rec, ge)
    assert pd.api.types.is_numeric_dtype(tab["page"])
    sel = EV.filter_events(tab, query='phase == "TASK" and cond == "AI"', names={"fixation"})
    assert list(sel.onset) == [2.5]  # the 9.5 s fixation is after TASK_END
    assert len(EV.filter_events(tab, query="page >= 3", names={"PAGE"})) == 2
    with pytest.raises(ValueError):
        EV.filter_events(tab, query="cond ==")


def test_montage_and_metadata_epochs_contrast(demo):
    import mne

    from brainstudio import events as EV

    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = rec.raw.copy()
    raw.set_montage(None)
    names = list(raw.ch_names)
    raw.rename_channels({n: f"ch{i}" for i, n in enumerate(names)})
    rep = E.apply_montage(raw, E.montage_from_source("standard_1005"), names)
    assert rep["n_renamed"] == len(names) and rep["n_positioned"] == 30
    assert raw.get_channel_types(["HEOG"]) == ["eog"]

    ge = G.detect_events(rec.gaze)
    tab = EV.build_event_table(rec, ge)
    ep = E.epochs_from_table(E.preprocess(raw, E.FilterParams(1, 30)),
                             EV.filter_events(tab, names={"fixation", "stimulus"}), -0.2, 0.5)
    assert set(ep.event_id) == {"fixation", "stimulus"} and ep.metadata is not None
    a, b = ep["fixation"], ep["stimulus"]
    evs = E.contrast_evokeds(a, b, ("fix", "stim"))
    assert "fix − stim" in evs
    st = E.cluster_contrast(a, b, roi=["Pz"], n_permutations=200)
    assert len(st["clusters"]) == len(st["p_values"])
    for m in st["clusters"]:  # always boolean masks over time
        assert m.dtype == bool and m.shape == st["times"].shape
    # the stimulus P300 at Pz should differ from fixations (simulated effect)
    assert min(st["p_values"]) < 0.05


def test_channel_reference(demo):
    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    assert E.parse_reference("keep", rec.raw) is None
    assert E.parse_reference("CAR", rec.raw) == "average"
    assert E.parse_reference("cz", rec.raw) == ["Cz"]
    assert E.parse_reference("TP9, Cz", rec.raw.copy().rename_channels({"T7": "TP9"})) == [
        "TP9", "Cz"]
    with pytest.raises(ValueError, match="not found"):
        E.parse_reference("M1", rec.raw)
    out = E.preprocess(rec.raw, E.FilterParams(None, None, None, "Cz"))
    assert np.abs(out.get_data(picks=["Cz"])).max() < 1e-12  # reference channel is flat
    pair = E.preprocess(rec.raw, E.FilterParams(None, None, None, "O1, O2"))
    o = pair.get_data(picks=["O1", "O2"])
    assert np.abs(o.sum(0)).max() < 1e-12  # linked reference: O1 = -O2


def test_spatial_contrast_with_channel_reference(demo):
    """Single-channel reference → flat channel must not poison the F statistic."""
    import warnings

    from brainstudio import events as EV

    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = E.preprocess(rec.raw, E.FilterParams(1, 30, None, "Cz"))
    tab = EV.build_event_table(rec, G.detect_events(rec.gaze))
    ep = E.epochs_from_table(raw, EV.filter_events(tab, names={"fixation", "stimulus"}),
                             -0.1, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no 0/0 in the F test
        st = E.cluster_contrast(ep["fixation"], ep["stimulus"], n_permutations=100,
                                spatial=True)
    assert "Cz" in st["excluded"] and "Cz" not in st["ch_names"]
    for m in st["clusters"]:
        assert m.dtype == bool and m.shape == (len(st["times"]), len(st["ch_names"]))
    with pytest.raises(ValueError, match="flat"):
        E.cluster_contrast(ep["fixation"], ep["stimulus"], roi=["Cz"], n_permutations=10)


def test_gtec_64_layout_montage():
    import mne
    assert "gtec_64" in E.builtin_montages()
    m = E.montage_from_source("gtec_64")
    assert len(m.ch_names) == 64
    assert m.ch_names[:3] == ["Fp1", "Fpz", "Fp2"] and m.ch_names[18] == "FC3"
    assert m.ch_names[-4:] == ["F9", "F10", "A1", "A2"]
    info = mne.create_info([f"EEG{i:03d}" for i in range(1, 65)], 256.0, "eeg")
    raw = mne.io.RawArray(np.zeros((64, 10)), info, verbose="error")
    rep = E.apply_montage(raw, m, list(m.ch_names))
    assert rep["n_renamed"] == 64 and rep["n_positioned"] == 64 and not rep["missing"]
    assert raw.ch_names[29] == "Cz"


def test_pipeline_bads_ica_groups_bandpower(demo):
    """Preprocessing → epoching → analysis on synthetic data with real blink artifacts."""
    from brainstudio import bandpower as BP
    from brainstudio import epoching as EP
    from brainstudio import events as EV
    from brainstudio import preprocessing as P
    from brainstudio import stats as ST

    rec = load_bids_run(demo / "bids", scan_bids(demo / "bids")[0])
    raw = rec.raw.copy().load_data()
    raw._data[raw.ch_names.index("T7")] = 0.0                                   # flat
    raw._data[raw.ch_names.index("C4")] += np.random.default_rng(0).normal(
        0, 400e-6, raw.n_times)                                                  # noisy
    bad = P.detect_bad_channels(raw)
    assert {"T7", "C4"} <= set(bad.loc[bad.bad, "channel"])
    assert bad.bad.sum() <= 4

    ev = G.detect_events(rec.gaze)
    bads = ["T7", "C4"]
    ica, fit_raw = P.fit_ica(raw, bads, 40.0, None, P.ICAParams(n_components=10))
    assert not set(bads) & set(ica.ch_names)
    sc = P.score_blink_components(ica, fit_raw, ev.blinks)
    assert sc.suggested.any()
    assert sc.loc[sc.suggested, "blink_t"].max() == sc.blink_t.max()
    params = P.PreprocParams(bads=bads, l_freq=1, h_freq=40, reference="average",
                             ica_exclude=sc.loc[sc.suggested, "component"].tolist())
    clean = P.run_preprocessing(raw, params, ica)
    assert clean.info["bads"] == bads and raw.info["bads"] == []  # input untouched
    before = P.blink_locked_average(P._filtered(raw, 1, 40, None), ev.blinks, ["Fp1", "Fp2"])
    after = P.blink_locked_average(clean, ev.blinks, ["Fp1", "Fp2"])
    assert np.ptp(after[1]) < 0.5 * np.ptp(before[1])                            # blinks gone

    table = EV.build_event_table(rec, ev)
    groups = {"stim": table[table.name == "stimulus"],
              "fix": table[table.name == "fixation"].iloc[:200]}
    eps, summary = EP.build_group_epochs(clean, groups, EP.EpochSettings(-0.2, 1.0, None),
                                         ev.blinks)
    assert set(eps) == {"stim", "fix"} and (summary.kept > 0).all()
    assert eps["stim"].baseline is None and "group" in eps["stim"].metadata
    assert not set(bads) & set(eps["stim"].ch_names)

    s = BP.BandSettings(tmin=0.0, tmax=1.0)
    base = BP.BandBaseline(np.array([1.0]), 0.0, 20.0)
    per = []
    for g, ep in eps.items():
        bp = BP.band_power(ep, s)
        assert bp.shape == (len(ep), len(ep.ch_names), len(s.bands)) and np.nanmin(bp) > 0
        bb, owner = BP.baseline_band_power(clean, base, s, ep.ch_names, 1.0)
        assert len(bb) == 20 and (owner == 0).all()
        ref, sd = BP.reference_for(ep.metadata.onset.to_numpy(), base, bb, owner)
        assert np.isfinite(BP.normalise(bp, ref, sd, "dB")).mean() > 0.9
        idx, _ = BP.eeg_indices(bp, ep.ch_names, s.bands, ref=ref)
        assert idx["engagement β/(α+θ)"].notna().all()
        assert idx["α ERD % (parietal)"].notna().all()
        idx["group"] = g
        per.append(idx)
    import pandas as pd
    tab = ST.comparison_table(pd.concat(per), ["θ/α", "α ERD % (parietal)"], ["stim", "fix"])
    assert list(tab.columns[:3]) == ["measure", "stim mean", "stim sem"]
    assert tab["q (FDR)"].notna().all() and (tab["q (FDR)"] >= tab["p"] - 1e-12).all()
