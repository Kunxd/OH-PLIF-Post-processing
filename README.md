# OH-PLIF post-processing

Turbulent flame speed analysis from OH-PLIF image stacks (Bachelor thesis:
*Chemiluminescence vs. OH-LIF: Evaluating Turbulent Flame Speed Estimation
Methods in Premixed Flames*). Pipeline after Wabel et al. / Driscoll:
Wabel local thresholding → per-position c̄ → ray-scanned c̄ contours →
surface area by revolution and flame height. S_T = ṁ / (ρ_u · A).

## Files
Two versions of the analysis. They are identical except for how each frame is
binarised, read the same `config_local.py` and write to separate folders.

| file | purpose |
|---|---|
| `ohplif1.py` | **Version 1 — moving window.** Each pixel vs. the maximum in a `WINDOW_SIZE` × `WINDOW_SIZE` square. The full batch script (v8): all flames, parallel, resumable. Results in `OUTDIR`. Called `run_wabel_batch.py`, then `main.py`, until Nov 2026; same code, only renamed |
| `ohplif2.py` | **Version 2 — watershed.** Each pixel vs. the signal at the nearest watershed ridge (Wabel's PhD thesis 2017, Sec. 2.4.1.4, p. 57; 25 %, 4 passes as in Wabel et al. PCI 2017, p. 1803). Does not copy `ohplif1.py`: it loads it and swaps only the threshold, so every fix in `ohplif1.py` applies to both. Results in `OUTDIR_ohplif2`. Needs scikit-image. Was `wabel_new.py` + `wabel_watershed.py` |
| `spe_reader.py` | LightField / WinSpec `.spe` reader |
| `estimate_nozzle_center.py` | estimates each flame's nozzle centre from the saved c̄ maps |
| `thesis_figures.py` | **Clean figures for the thesis** from finished results of either version; nothing is reprocessed. Per instantaneous composite: the matched frames as raw bg-subtracted signal, as binary (burnt 1 white, unburnt 0 black) — both without lines — with the flame front + pockets re-detected by a flood fill from the nozzle on the burnt gas in grey (`_front`), and the same front in blue on the OH signal in a white→red map (`_front_red`; composites where the fill leaks out through a gap go to `front_failed/`). Average composites with the c̄ field and the measured c-lines, including pos0 as a grey "not measured" band with the estimated c-line in grey (`SHOW_POS0`); no axis line, position marks or debug overlays. `python3 thesis_figures.py OUTDIR` → `OUTDIR/thesis_figures/`; settings (colours, axes in mm, width, pos0, …) at the top of the file; `--from-spe` re-reads the raw frames from the .spe files. Writes only into its own `thesis_figures/` folder; figures already there are skipped (an interrupted run continues), `--redo` draws everything again (needed after changing a setting) |
| `thesis_matrices.py` | **All flames in one figure**, next to the output of `thesis_figures.py` and drawn with its own functions and settings; nothing is reprocessed. *Condition matrices* like the OH\* photo matrix (rows d_j = 6/9/12 mm, columns CH4 Re_j 15k/30k \| H2 15k/30k/60k), one per picture type: raw, binary, front, front_red, c̄, c̄ + one c-line (one matrix per level), all c-lines (c̄ with pos0). All cells share one mm scale (nozzle axis centred, heights aligned) and the same ± half-width. The instantaneous matrices show the same frames per cell (first composite whose front closes). *Position-window figure* per fuel: one column per condition, one panel per measurement position, OH signal in white→red with the front in blue, columns equally wide and as close as the widest front allows (+0.5 mm), any ambient air inside a panel painted red. A cell whose picture you don't like can be swapped for another composite: `--swap "H2,9,30k=instant_2_front.png"`, the `SWAP` setting, or `make_matrices(folder, swap={...})` from Python. Flame → cell from the folder name (`CH4`/`H2`, `D<n>`, `Re<n>`; check with `--list`). `python3 thesis_matrices.py OUTDIR/thesis_figures` → `OUTDIR/thesis_figures/matrices/threshold_XX/` |
| `make_synthetic_results.py` | Fake results folder in exactly the `ohplif1.py` format (15 invented flames, with a leaking front, an open tip and a non-round Re as test cases) for testing the figure scripts without real data: `python3 make_synthetic_results.py /tmp/synth && python3 thesis_matrices.py /tmp/synth` |
| `centre_overview.py` | overview of the measured flame axis of all flames (near-nozzle = the composites' synthetic axis, whole-flame, per position, tilt) from the saved `c_lines.npz`; nothing reprocessed. `python3 centre_overview.py OUTDIR` → `OUTDIR/centre_overview/` (Excel, overview figures, per-flame check images). Works on the results of both versions |
| `white_bg.py` | supervisor's core-isolation script, unchanged; only an added block that reads `OUTDIR` from `config_local.py` and saves figures to `OUTDIR/white_bg/`. `python3 white_bg.py` = original synthetic test, `--real` = binarized frames under `OUTDIR` |
| `config_local.example.py` | template for your machine-specific settings |

## Setup (once per computer)
```bash
git clone https://github.com/Kunxd/OH-PLIF-Post-processing.git
cd OH-PLIF-Post-processing
pip3 install -r requirements.txt
cp config_local.example.py config_local.py   # then edit paths in config_local.py
```

## Run
```bash
git pull                     # get the latest version
python3 ohplif1.py           # version 1 (moving window)  → OUTDIR
python3 ohplif2.py           # version 2 (watershed)      → OUTDIR_ohplif2
```
Set `DRY_RUN = True` in `config_local.py` first to check which flames and
positions are found. An interrupted run is resumed by simply starting it again.

`ohplif2.py` ignores `WINDOW_SIZE`; its own optional settings
(`WATERSHED_PRE_MEDIAN`, `WATERSHED_CONNECTIVITY`, `WATERSHED_N_ITER`,
`OHPLIF2_OUTDIR`) are listed in `config_local.example.py`. It takes about
0.3–0.6 s per frame on real data (`ohplif1.py`: 0.03 s). Results of the earlier
`wabel_new.py` in `OUTDIR_wabel_new` are reused (not redone) as long as no
`OUTDIR_ohplif2` folder exists. To run the helper scripts on version-2 results,
pass the folder, e.g. `python3 centre_overview.py /path/to/OUTDIR_ohplif2`.

## Input naming
`<flame name>_pos<N> <anything>.spe` for data, `<flame name>_pos<N>_bg <anything>.spe`
for backgrounds. The flame name (everything before `_pos<N>`) becomes the output
folder name; the nozzle diameter is read from `D<n>` in it.

## Output (v8.5: threshold → flame)
Everything that depends on the binarization threshold is stored per threshold
(`THRESHOLD_FRAC` can be one value or a list, e.g. `[0.20, 0.25, 0.30]`):
```
OUTDIR/
├── ALL_conditions_results.xlsx        all flames × all thresholds (merged "Threshold 25%" column groups)
├── threshold_25/
│   ├── ALL_conditions_results_25.xlsx all flames, this threshold only (when several thresholds)
│   └── <flame>/
│       ├── average composites/        c̄ composite without / per / with all c-lines, cbar_composite.npy, c_lines.npz
│       ├── instantaneous/             stitched instantaneous fronts on the raw bg-subtracted signal
│       ├── instantaneous_binarized/   the same stitched frames as binary images
│       ├── posN/                      cbar.npy, cbar_map.png, instant_fronts.npz, overlay_*.png, binary_*.png
│       ├── <flame>_results.xlsx       A and h per c̄ level, area breakdown, instantaneous A, input files
│       └── user_settings_log.txt
├── threshold_30/ …
├── _run_data/<flame>/                 bookkeeping shared by all thresholds (don't edit):
│                                      posN/done.json (resume marker), bg_mean.npy, settings_posN.txt
├── logs/  code_used/
```
Each frame is read once and binarized once per threshold. Adding a threshold
later only processes the new threshold; existing ones are reused.
Results from v8.4 and older (`OUTDIR/<flame>/…`) are moved into this layout
automatically on the next run (nothing is reprocessed; the old flame Excel is
kept as `_run_data/<flame>/old_<flame>_results.xlsx`).
Instantaneous composites (visualisation only — positions were not recorded simultaneously): per position, frames are matched on their binary images (`STITCH_BAND_PX` rows at each junction, best chain over all positions); the matched binaries are stitched, and the front is traced once on the stitched binary (since v8.7). The front is therefore continuous across junctions, and a mismatch between positions shows as a short horizontal step (its size is printed at each junction).
With `INSTANT_AREA = True`, the inner flame front of each stitched composite (outer product/air edge removed, pockets kept) is revolved 180° per side like the c-lines → `A_inst`.

Run logs in `OUTDIR/logs/`, the code and `config_local.py` of each run in `OUTDIR/code_used/`. `estimate_nozzle_center.py` reads `OUTDIR/threshold_XX/` (first threshold, or give one: `python3 estimate_nozzle_center.py OUTDIR 0.30`).
