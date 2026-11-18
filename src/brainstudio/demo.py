"""Synthetic EEG + eye-tracking data (free-viewing task) for demos and tests.

``python -m brainstudio.demo OUTDIR`` writes a small BIDS dataset (two
subjects, BrainVision EEG + BIDS eye-tracking physio) and an XDF file with
EEG, gaze and marker streams.
"""

from __future__ import annotations

import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from .eeg import set_standard_montage

SCREEN = (1920, 1080)
CH_NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
            "Pz", "P4", "P8", "PO3", "PO4", "O1", "Oz", "O2", "HEOG", "VEOG"]
# AOI targets the simulated observer looks at (centres in px)
TARGETS = [(480, 270), (1440, 270), (960, 540), (480, 810), (1440, 810)]


@dataclass
class SimData:
    eeg: np.ndarray  # (n_ch, n) volts
    eeg_sfreq: float
    gaze_t: np.ndarray  # seconds, EEG time base
    gx: np.ndarray
    gy: np.ndarray
    pupil: np.ndarray
    gaze_sfreq: float
    markers: list[tuple[float, str]]
    fix_onsets: np.ndarray


def _pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def simulate(duration=120.0, eeg_sfreq=250.0, gaze_sfreq=500.0, gaze_offset=0.5,
             seed=0) -> SimData:
    rng = np.random.default_rng(seed)
    # ---------------------------------------------------------------- gaze
    n_g = int((duration - gaze_offset) * gaze_sfreq)
    gt = gaze_offset + np.arange(n_g) / gaze_sfreq
    gx, gy = np.empty(n_g), np.empty(n_g)
    pos = np.array(TARGETS[2], float)
    i, fix_onsets = 0, []
    while i < n_g:
        # saccade (minimum-jerk) to the next fixation point
        tgt = np.array(TARGETS[rng.integers(len(TARGETS))], float) + rng.normal(0, 60, 2)
        tgt = np.clip(tgt, 20, np.array(SCREEN) - 20)
        amp = np.linalg.norm(tgt - pos)
        sac_n = max(int((0.021 + 0.0022 * amp / 35) * gaze_sfreq), 3)
        s = np.linspace(0, 1, sac_n)
        prof = 10 * s**3 - 15 * s**4 + 6 * s**5
        k = min(sac_n, n_g - i)
        gx[i:i + k] = (pos[0] + (tgt[0] - pos[0]) * prof)[:k]
        gy[i:i + k] = (pos[1] + (tgt[1] - pos[1]) * prof)[:k]
        i += k
        pos = tgt
        fix_onsets.append(gt[min(i, n_g - 1)])
        fix_n = int(rng.gamma(4, 0.065) * gaze_sfreq) + int(0.08 * gaze_sfreq)
        k = min(fix_n, n_g - i)
        drift = np.cumsum(rng.normal(0, 0.4, (k, 2)), axis=0)
        gx[i:i + k] = pos[0] + drift[:, 0]
        gy[i:i + k] = pos[1] + drift[:, 1]
        i += k
    gx += rng.normal(0, 2.0, n_g)
    gy += rng.normal(0, 2.0, n_g)
    pupil = 3000 + 200 * np.sin(2 * np.pi * gt / 30) + rng.normal(0, 10, n_g)
    # stimulus markers every 6 s with pupil dilation response
    markers = []
    for t in np.arange(3.0, duration - 3, 6.0):
        markers.append((float(t), "stimulus"))
        markers.append((float(t + rng.uniform(0.6, 1.2)), "response"))
        resp = np.clip(gt - t, 0, None)
        pupil += 250 * (resp / 0.9) * np.exp(1 - resp / 0.9) * (gt > t)
    # blinks: missing data
    blink_times = []
    for bt in np.cumsum(rng.exponential(4.0, 60)) + gaze_offset:
        if bt > duration - 1:
            break
        m = (gt >= bt) & (gt < bt + rng.uniform(0.1, 0.25))
        gx[m] = gy[m] = pupil[m] = np.nan
        blink_times.append(bt)
    fix_onsets = np.asarray(fix_onsets)

    # ---------------------------------------------------------------- EEG
    n_e = int(duration * eeg_sfreq)
    et = np.arange(n_e) / eeg_sfreq
    n_ch = len(CH_NAMES)
    eeg = np.vstack([_pink(n_e, rng) for _ in range(n_ch)]) * 8e-6
    occ = np.array([n.startswith(("O", "PO", "P")) for n in CH_NAMES])
    alpha = np.sin(2 * np.pi * 10 * et + rng.uniform(0, 2 * np.pi))
    alpha *= 1 + 0.5 * np.sin(2 * np.pi * et / 20)
    eeg[occ] += 10e-6 * alpha
    # lambda response (occipital P1 ~ 90 ms) at fixation onsets
    kern_t = np.arange(0, 0.4, 1 / eeg_sfreq)
    lam = (np.exp(-((kern_t - 0.09) / 0.025) ** 2) - 0.6 * np.exp(-((kern_t - 0.17) / 0.04) ** 2))
    imp = np.zeros(n_e)
    np.add.at(imp, np.clip((fix_onsets * eeg_sfreq).astype(int), 0, n_e - 1), 1)
    lam_sig = np.convolve(imp, lam)[:n_e]
    eeg[occ] += 6e-6 * lam_sig
    # stimulus P300 (parietal)
    p3 = np.exp(-((kern_t - 0.32) / 0.07) ** 2)
    imp = np.zeros(n_e)
    for t, d in markers:
        if d == "stimulus":
            imp[int(t * eeg_sfreq)] = 1
    par = np.array([n in ("Pz", "P3", "P4", "CP1", "CP2", "Cz") for n in CH_NAMES])
    eeg[par] += 8e-6 * np.convolve(imp, p3)[:n_e]
    # EOG / frontal artifacts from eye movements and blinks
    gx_e = np.interp(et, gt, np.nan_to_num(gx, nan=SCREEN[0] / 2))
    gy_e = np.interp(et, gt, np.nan_to_num(gy, nan=SCREEN[1] / 2))
    heog = (gx_e - SCREEN[0] / 2) * 0.15e-6
    veog = (gy_e - SCREEN[1] / 2) * 0.12e-6
    blink_sig = np.zeros(n_e)
    for bt in blink_times:
        blink_sig += 150e-6 * np.exp(-((et - bt - 0.08) / 0.06) ** 2)
    eeg[CH_NAMES.index("HEOG")] += heog
    eeg[CH_NAMES.index("VEOG")] += veog + blink_sig
    for name, w in (("Fp1", 0.6), ("Fp2", 0.6), ("F7", 0.3), ("F8", 0.3)):
        eeg[CH_NAMES.index(name)] += w * (blink_sig + veog)
    eeg[CH_NAMES.index("F7")] += 0.5 * heog
    eeg[CH_NAMES.index("F8")] -= 0.5 * heog
    eeg += 3e-6 * np.sin(2 * np.pi * 50 * et)  # line noise
    return SimData(eeg, eeg_sfreq, gt, gx, gy, pupil, gaze_sfreq, sorted(markers), fix_onsets)


def make_raw(sim: SimData) -> mne.io.RawArray:
    types = ["eog" if "EOG" in n else "eeg" for n in CH_NAMES]
    info = mne.create_info(CH_NAMES, sim.eeg_sfreq, types)
    info["line_freq"] = 50
    raw = mne.io.RawArray(sim.eeg, info, verbose="error")
    raw.set_meas_date(1_700_000_000)
    set_standard_montage(raw)
    on = [t for t, _ in sim.markers]
    raw.set_annotations(mne.Annotations(on, [0.0] * len(on), [d for _, d in sim.markers],
                                        orig_time=raw.info["meas_date"]))
    return raw


def write_bids(root: str | Path, subjects=("01", "02")) -> Path:
    from mne_bids import BIDSPath, make_dataset_description, write_raw_bids

    root = Path(root)
    for k, sub in enumerate(subjects):
        sim = simulate(seed=k)
        raw = make_raw(sim)
        bp = BIDSPath(subject=sub, task="freeview", datatype="eeg", root=root)
        write_raw_bids(raw, bp, format="BrainVision", allow_preload=True,
                       event_id={"stimulus": 1, "response": 2}, overwrite=True,
                       verbose="error")
        # eye tracking (BIDS 1.10 physio, eye-tracker clock in ms)
        base = bp.copy().update(suffix=None, extension=None).basename
        d = bp.directory
        clock0 = 4_200_000.0
        ts = clock0 + (sim.gaze_t - sim.gaze_t[0]) * 1000.0
        rows = np.column_stack([ts, sim.gx, sim.gy, sim.pupil])
        fname = d / f"{base}_recording-eye1_physio.tsv.gz"
        with gzip.open(fname, "wt") as f:
            for r in rows:
                f.write("\t".join("n/a" if not np.isfinite(v) else f"{v:.2f}" for v in r) + "\n")
        (d / f"{base}_recording-eye1_physio.json").write_text(json.dumps({
            "PhysioType": "eyetrack", "SamplingFrequency": sim.gaze_sfreq,
            "StartTime": float(sim.gaze_t[0]), "RecordedEye": "right",
            "Columns": ["timestamp", "x_coordinate", "y_coordinate", "pupil_size"],
            "timestamp": {"Units": "ms", "Origin": "System startup"},
            "x_coordinate": {"Units": "pixel"}, "y_coordinate": {"Units": "pixel"},
            "pupil_size": {"Units": "arbitrary"},
            "ScreenResolution": list(SCREEN), "ScreenSize": [0.53, 0.30],
            "ScreenDistance": 0.65, "Manufacturer": "Synthetic",
        }, indent=2))
        ev = d / f"{base}_recording-eye1_physioevents.tsv.gz"
        with gzip.open(ev, "wt") as f:
            for t, desc in sim.markers:
                f.write(f"{clock0 + (t - sim.gaze_t[0]) * 1000:.1f}\tn/a\t{desc}\n")
        (d / f"{base}_recording-eye1_physioevents.json").write_text(json.dumps({
            "Columns": ["onset", "duration", "message"],
            "onset": {"Units": "ms"}}, indent=2))
    make_dataset_description(path=root, name="brAInStudio synthetic free-viewing",
                             dataset_type="raw", overwrite=True, verbose="error")
    return root


def write_xdf_demo(path: str | Path, lsl_t0: float = 5123.4) -> Path:
    from .io.xdf_writer import write_xdf

    sim = simulate(duration=60.0, seed=7)
    et = lsl_t0 + np.arange(sim.eeg.shape[1]) / sim.eeg_sfreq
    gt = lsl_t0 + sim.gaze_t
    types = ["EOG" if "EOG" in n else "EEG" for n in CH_NAMES]
    write_xdf(path, [
        dict(name="Synthetic EEG", type="EEG", srate=sim.eeg_sfreq, format="float32",
             labels=CH_NAMES, units=["microvolts"] * len(CH_NAMES), ch_types=types,
             timestamps=et, data=(sim.eeg * 1e6).T),
        dict(name="Synthetic Eyetracker", type="Gaze", srate=sim.gaze_sfreq,
             format="float32", labels=["gaze_x", "gaze_y", "pupil_diameter"],
             timestamps=gt, data=np.column_stack([sim.gx, sim.gy, sim.pupil])),
        dict(name="Markers", type="Markers", srate=0, format="string", labels=["marker"],
             timestamps=[lsl_t0 + t for t, _ in sim.markers if t < 60],
             data=[d for t, d in sim.markers if t < 60]),
    ])
    return Path(path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "demo_data")
    out.mkdir(parents=True, exist_ok=True)
    print("BIDS dataset:", write_bids(out / "bids"))
    print("XDF file:   ", write_xdf_demo(out / "freeview.xdf"))


if __name__ == "__main__":
    main()
