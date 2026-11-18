# BrainStudio

Analysis and visualisation of simultaneous **EEG + eye-tracking** recordings, built on
[MNE-Python](https://mne.tools) / [MNE-BIDS](https://mne.tools/mne-bids) for EEG and
[PyGaze](https://www.pygaze.org) for gaze (AOIs), with a PySide6 + pyqtgraph GUI.

## Install & run

```bash
uv sync
```

```bash
uv run brainstudio --demo
```

```bash
uv run brainstudio /path/to/bids_root
```

```bash
uv run brainstudio /path/to/recording.xdf --screen 1920x1080
```

`--demo` writes a synthetic free-viewing dataset (BIDS + XDF) to a temp folder and
opens it. `uv run brainstudio-demo OUTDIR` writes it to a folder of your choice.

## Supported inputs

| Input | EEG | Gaze | Events |
|---|---|---|---|
| **BIDS dataset** (folder) | `*_eeg.{vhdr,edf,bdf,set,fif}` via `mne_bids.read_raw_bids` | BIDS eye tracking (`*_recording-<eye>_physio.tsv.gz` + JSON, `PhysioType: eyetrack`), legacy `*_eyetrack.tsv`, or MNE `eyegaze`/`pupil` channels | `events.tsv`, `*_physioevents.tsv.gz` messages |
| **XDF** (LSL) | stream of type EEG/ExG (else the largest numeric stream) | stream scored by type/name/labels (gaze, eye, pupil, POG…) | string marker streams |
| EEG file | any MNE-readable file | MNE eyetrack channels | annotations |
| EyeLink `.asc` | – | via `mne.io.read_raw_eyelink` | messages |

You can attach a separate eye-tracking file to a loaded recording (**File ▸ Attach eye-tracking file**).

### Time alignment
Everything lives on one time base: t = 0 is the first EEG sample.

* **BIDS**: gaze uses the sidecar `StartTime` (seconds relative to the EEG). Tracker
  timestamps (`timestamp` column, unit from the sidecar or inferred from
  `SamplingFrequency`) define the sample times.
* **XDF**: pyxdf applies the LSL clock offsets stored in the file. BrainStudio then
  de-jitters each regular stream itself, one linear fit per gap-free segment, and uses
  the *effective* sampling rate. Some recordings have samples stamped on another clock,
  as in Gazepoint streams where a few samples carry the LSL clock instead of the
  tracker clock. BrainStudio detects these, removes them, and uses them to estimate the
  offset between clocks. A stream that doesn't overlap the EEG is then mapped onto the
  EEG clock with that offset. If no offset can be found, stream starts are aligned and
  the info panel shows a **WARNING**. Every step is listed in the info panel.
* **Gaze offset** (Controls panel) shifts gaze relative to the EEG by hand.

Gaze coordinates are in screen pixels. Normalised coordinates (Gazepoint, Tobii, Pupil
Labs) are scaled to the screen size. Validity flags (e.g. `FPOGV`) and 0/0 samples count
as missing.

## GUI

BrainStudio is styled like the lab's other app, brAInStream:
* the dark style with blue highlights and the brAIn lab red;
* a top bar with the br**AI**n studio wordmark;
* icons drawn in code.

The style sheet is BrainStudio's own; no g.Pype files are used.

**Dataset dock** (left):
* a **ribbon** of buttons for the File menu's functions: BIDS, File, Eye data, Montage,
  Stimulus, Export, Snapshot;
* the recording tree;
* for XDF files, a **Streams** table. It lists every stream with its channels and rate
  and the **role** it plays: *EEG*, *gaze*, *markers* or *ignore*. Roles are picked
  automatically. To change them, pick another role and press **Reload with these
  roles**. **Automatic** goes back to the automatic roles. At most one stream can be
  EEG and one gaze; hover a stream to see its channel labels.

Three main tabs:

* **Viewer** (shows the preprocessed EEG by default; switch to raw in the Controls dock)
  * Stacked EEG traces with event markers and shaded saccades, blinks and fixations.
  * Gaze x/y and pupil traces, linked on the same time axis.
  * Overview bar showing gaze validity and events.
  * Playback controls.
  * Screen view: live gaze point and trail, numbered scanpath, fixation heatmap,
    optional stimulus image, and draggable **AOIs**.
  * **Live topomap** of the EEG around the cursor (needs channel positions).
* **EEG** is a three-stage pipeline. Every analysis runs in MNE.
  * Each stage takes its input from the stage before. A banner at the top of each stage
    says what that input is and whether the stage's results are up to date (●), out of
    date (▲) or not run yet (○).
  * Changing a setting marks that stage, and every stage after it, out of date.
  * **Run pipeline ▶▶** runs everything from the first stage that is out of date.
  * Long computations run in the background.
  * The stages:
    1. **Preprocessing**
       * *Channels*: automated peak-to-peak rejection. The data are cut into windows and
         band-passed/notched like the output will be. A channel is bad when too many of
         its windows are noisy (ptp above a threshold) or flat, or when it is a
         robust-z outlier. Tick or untick channels by hand. Bad channels are left out of
         ICA, the average reference, epochs and statistics, or interpolated with
         spherical splines.
       * *Filtering & reference*: high-pass, low-pass, notch; keep, average, one
         channel, or the mean of several (`M1, M2`).
       * *ICA blink removal* (`mne.preprocessing.ICA`, picard):
         * Fitted on a ≥ 1 Hz copy of the good channels.
         * Components are scored against the eye tracker's blinks, using a
           blink-locked t-score.
         * They are also scored against EOG channels or, without EOG, the frontal
           electrodes (`find_bads_eog`).
         * Suggested components are pre-ticked.
         * MNE component, source and property plots are available.
       * Checks: spectrum before/after, per-channel noisy/flat %, and the
         blink-locked frontal EEG before/after.
    2. **Epoching**
       * Map events into named **groups** (conditions). Each group has its own
         checklist and filters (see *Selecting events*).
       * Set the epoch start and end around each event, peak-to-peak and flat
         rejection, optionally drop epochs containing an eye-tracker blink, and cap the
         epochs per group.
       * Shows a summary (selected, outside, rejected, kept) and a timeline of where
         each group's events fall.
       * Save one `-epo.fif` per group.
       * Epochs are *not* baseline-corrected here. Each analysis picks its own baseline.
    3. **Analysis** of the ticked groups:
       * *ERP*:
         * Baseline = a window **before the event** (e.g. −0.2…0 s).
         * ROI waveforms ± SEM; with two groups, the A − B difference and a cluster
           permutation test, on the ROI or spatio-temporal (`mne.stats`).
         * Mean and peak amplitude and peak latency in a measurement window, compared
           across groups.
         * MNE compare-evokeds, joint plots and topomaps.
       * *Band power*:
         * Editable bands. Power in a window of the epoch from `Epochs.compute_psd`
           (Welch or multitaper).
         * Baseline = a window **after a baseline event**, e.g. 5…115 s after
           `BASELINE_START`. It is cut into segments as long as the analysis window.
           Either the mean of all baseline events or the nearest preceding one is
           used, and results are expressed in dB, % change, ratio or z.
         * Bar charts ± SEM, a Morlet ERD/ERS time course on the same baseline, and
           MNE topomaps per band, group and difference.
       * *Cognitive indices* per epoch, with editable regions:
         * engagement β/(α+θ) (Pope 1995)
         * workload = frontal-midline θ / parietal α (Gevins & Smith 2003)
         * θ/α, frontal θ/β, frontal alpha asymmetry
         * parietal α ERD % and frontal-midline θ ERS % relative to the baseline
         * eye metrics: task-evoked pupil dilation, blink rate, mean fixation duration
       * Groups are compared with a Welch t-test and Mann-Whitney test (two groups) or
         ANOVA and Kruskal-Wallis (more), with Hedges' g and FDR q-values. These are
         epoch-level statistics within one recording.
       * Every result exports to CSV/FIF with a provenance text file.
* **Gaze**:
  * fixation, saccade and blink summary
  * fixation-duration and amplitude histograms, main sequence
  * per-AOI fixation count, dwell time and time to first fixation, computed with
    `pygaze.plugins.aoi.AOI` (CSV export)
* **Controls dock**: viewer options, which EEG to display (preprocessed or raw),
  gaze-detection parameters, gaze offset, screen size. "Open in MNE browser" adds blinks
  and saccades as annotations.

### Selecting events
All markers and detected gaze events go into one table
(`brainstudio.events.build_event_table`):

* `name`, `kind` (marker, fixation, saccade or blink), `onset`, `duration`.
* Structured markers like `READING_PAGE|page=2|task=JF|cond=AI` are split into
  `name=READING_PAGE` plus columns `page`, `task`, `cond` (numeric where possible).
* Gaze events have `x`, `y`, `amplitude`, `peak_velocity` and `aoi`, the AOI drawn in
  the screen view.
* Every event inherits a `phase`, the innermost `X_START`…`X_END` span it falls in,
  together with that span's attributes. That makes queries like "fixations while
  reading in the AI condition" possible.

Each selector combines:
* a **checklist** of event types, with live counts after filtering;
* a **text filter**: a regex on the raw description;
* a **metadata filter**: a pandas `query`, the same syntax as MNE's
  `epochs['…']`. Examples: `phase == "READING" and cond == "AI"`,
  `duration > 0.2 and aoi == "TOP"`, `page in [1, 2]`.

The *Columns* menu lists the available columns with example values.

### Montage
**File ▸ Channel names & montage… (Ctrl+M)**:
* Choose a built-in MNE montage or a montage file (`.elc .sfp .loc .bvef .csd .txt`,
  BIDS `*_electrodes.tsv`, …).
* **Channel layouts** are listed first. A layout gives the electrode for each amplifier
  input, so picking one also fills in the channel names. `gtec_64` is the g.tec 64-channel
  extended 10-20 "standard" setup: 1 Fp1 … 60 O2, 61 F9, 62 F10, 63 A1, 64 A2, ground
  at AFz. It is preselected for recordings with 64 unnamed channels. Positions come from
  MNE's 10-05 template. To add your own layout, put a text file with one
  `<input> <name>` per line in `~/.brainstudio/montages/` (see
  `src/brainstudio/montages/gtec_64.txt`).
* Optionally rename channels in recording order. Edit the list, load names from a text
  file, or fill them from the montage's file order. Channels named *EOG/ECG/EMG* get
  those channel types.
* *Check* reports how many EEG channels got positions. *Plot sensors* shows the layout
  with MNE.

Topomaps, joint plots and spatio-temporal tests need positions.

### Colours
Every chart follows the app's light or dark palette and updates when the system
setting changes. That covers pyqtgraph plots, the topomap, and MNE's own matplotlib
figures (through rcParams). A single colour-blind-safe series palette is shared, so
group A and B have the same colours in the app and in MNE plots.

Shortcuts: ←/→ page, Space play, +/- EEG scale, PgUp/PgDn scroll channels, click a time
plot to move the cursor.

## Library use

```python
from brainstudio.io import load_xdf
from brainstudio import gaze as G, events as EV, preprocessing as P, epoching as EP
from brainstudio import bandpower as BP, stats as ST, eeg as E

rec = load_xdf("recording.xdf")      # roles={"4": "ignore"} overrides a stream's role
print([(st["name"], st["role"]) for st in rec.streams])
gtec = E.montage_from_source("gtec_64")                         # names + positions
E.apply_montage(rec.raw, gtec, gtec.ch_names)
ev = G.detect_events(rec.gaze)                                  # fixations / saccades / blinks
table = EV.build_event_table(rec, ev)                           # markers + gaze events

# 1 preprocessing
bad = P.detect_bad_channels(rec.raw, P.BadChannelParams(ptp_max_uv=250, notch=50))
bads = bad.loc[bad.bad, "channel"].tolist()
ica, fit_raw = P.fit_ica(rec.raw, bads, h_freq=40, notch=50)
scores = P.score_blink_components(ica, fit_raw, ev.blinks)      # blink-locked + EOG/frontal
pp = P.PreprocParams(bads=bads, l_freq=1, h_freq=40, notch=50, reference="average",
                     ica_exclude=scores.loc[scores.suggested, "component"].tolist())
clean = P.run_preprocessing(rec.raw, pp, ica)

# 2 epoching: named groups of events
fx = EV.filter_events(table, query='phase == "READING"', names={"fixation"})
groups = {"AI": fx[fx.cond == "AI"], "NoAI": fx[fx.cond == "NoAI"]}
epochs, summary = EP.build_group_epochs(clean, groups, EP.EpochSettings(-0.5, 1.5, 150))

# 3 analysis: ERP (pre-event baseline) and band power (baseline after a baseline event)
erp = {g: e.copy().apply_baseline((-0.2, 0)) for g, e in epochs.items()}
stats = E.cluster_contrast(erp["AI"], erp["NoAI"], roi=["Pz", "P3", "P4"])
s = BP.BandSettings(tmin=0, tmax=1.5)
base = BP.BandBaseline(table.loc[table.name == "BASELINE_START", "onset"].values, 5, 115)
per = []
for g, ep in epochs.items():
    bp = BP.band_power(ep, s)                                     # (epochs, channels, bands) µV²
    bb, owner = BP.baseline_band_power(clean, base, s, ep.ch_names, 1.5)
    ref, sd = BP.reference_for(ep.metadata.onset.values, base, bb, owner)
    idx, notes = BP.eeg_indices(bp, ep.ch_names, s.bands, ref=ref)  # engagement, workload…
    per.append(idx.assign(group=g))
import pandas as pd
print(ST.comparison_table(pd.concat(per), ["engagement β/(α+θ)", "α ERD % (parietal)"],
                          ["AI", "NoAI"]))
```

## Notes on the gaze algorithms
The pip release of PyGaze (`python-pygaze`) contains the experiment runtime and the
`AOI` plugin, but not the offline event detectors. Those live in PyGazeAnalyser, which
isn't on PyPI. BrainStudio reimplements the same algorithms in vectorised form:
* blinks: runs of missing samples (`blink_min`–`blink_max`)
* fixations: dispersion-based, staying within *radius* of the fixation start for at
  least *min fixation*
* saccades: velocity threshold

Thresholds are in pixels; tune them for your screen and tracker. Some trackers export a
fixation-filtered point of gaze (e.g. Gazepoint `FPOG*`) rather than raw gaze. With
that data the trace is step-like, so velocity-based saccade counts come out low;
fixation and blink detection still work.

## Tests

```bash
uv run pytest
```
