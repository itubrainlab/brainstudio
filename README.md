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

* **Viewer tab**
  * Stacked EEG traces with event markers and shaded saccades, blinks and fixations.
  * Gaze x/y and pupil traces, linked on the same time axis.
  * Overview bar showing gaze validity and events.
  * Playback controls.
  * Screen view: live gaze point and trail, numbered scanpath, fixation heatmap,
    optional stimulus image, and draggable **AOIs**.
  * **Live topomap** of the EEG around the cursor (needs channel positions).
* **Analysis tab** (all EEG analysis runs in MNE)
  * *Gaze statistics*:
    * fixation, saccade and blink summary
    * fixation-duration and amplitude histograms, main sequence
    * per-AOI fixation count, dwell time and time to first fixation, computed with
      `pygaze.plugins.aoi.AOI` (CSV export)
  * *Spectrum*: PSD per channel (`Raw.compute_psd`) and band topomaps.
  * *Event-related potentials*: an `mne.Epochs` locked to any selection of events
    (see below), with the event table attached as **epochs metadata**. Butterfly + GFP
    plot, MNE joint plot, epochs image, and saving `-epo.fif` with metadata.
  * *Contrast*: two event groups A and B, each chosen with its own checklist and
    filters.
    * Evokeds use `mne.combine_evoked` for the A − B difference wave and are shown with
      ±SEM on an ROI.
    * Statistics are a cluster-based permutation test with `mne.stats`: either
      `permutation_cluster_test` on the ROI mean or `spatio_temporal_cluster_test` with
      the montage adjacency. Significant clusters are shaded.
    * MNE `plot_compare_evokeds` and difference joint/topomaps.
    * Export: evokeds and epochs as `.fif`, clusters as `.csv`, plus a text file
      describing the selection.
* **Controls dock**
  * MNE filtering: high-pass, low-pass, notch.
  * Re-referencing (`set_eeg_reference`): *keep* (as recorded), *average*, any EEG
    channel from the list, or several typed with commas (`M1, M2`) for their mean,
    e.g. linked mastoids. The list follows channel renames from the montage dialog.
    ERPs and contrasts use the re-referenced data.
  * Gaze-detection parameters, gaze offset, screen size.
  * "Open in MNE browser" adds blinks and saccades as annotations.

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
from brainstudio.io import scan_bids, load_bids_run, load_xdf
from brainstudio import gaze as G, eeg as E

rec = load_xdf("recording.xdf")              # or load_bids_run(root, scan_bids(root)[0])
print(rec.summary(), rec.description)
ev = G.detect_events(rec.gaze)               # fixations / saccades / blinks DataFrames
raw = E.preprocess(rec.raw, E.FilterParams(1, 40))
from brainstudio import events as EV
table = EV.build_event_table(rec, ev)          # markers + gaze events, parsed attributes
sel = EV.filter_events(table, query='phase == "READING"', names={"fixation"})
epochs = E.epochs_from_table(raw, sel, -0.2, 0.6, reject_uv=150)   # metadata attached
a, b = epochs['cond == "AI"'], epochs['cond == "NoAI"']           # MNE metadata queries
evokeds = E.contrast_evokeds(a, b, ("AI", "NoAI"))                 # incl. "AI − NoAI"
stats = E.cluster_contrast(a, b, roi=["O1", "Oz", "O2"], n_permutations=1000)
aoi = G.aoi_statistics(ev.fixations, {"left": G.make_aoi("rectangle", (0, 0), (960, 1080))})
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
