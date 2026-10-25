# OH-PLIF post-processing

Turbulent flame speed analysis from OH-PLIF image stacks (Bachelor thesis:
*Chemiluminescence vs. OH-LIF: Evaluating Turbulent Flame Speed Estimation
Methods in Premixed Flames*). Pipeline after Wabel et al. / Driscoll:
Wabel local thresholding → per-position c̄ → ray-scanned c̄ contours →
surface area by revolution and flame height. S_T = ṁ / (ρ_u · A).

## Files
| file | purpose |
|---|---|
| `run_wabel_batch.py` | main batch script (v8): all flames, parallel, resumable |
| `spe_reader.py` | LightField / WinSpec `.spe` reader |
| `estimate_nozzle_center.py` | estimates each flame's nozzle centre from the saved c̄ maps |
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
python3 run_wabel_batch.py
```
Set `DRY_RUN = True` in `config_local.py` first to check which flames and
positions are found. An interrupted run is resumed by simply starting it again.

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
With `INSTANT_AREA = True`, the inner flame front of each stitched composite (outer product/air edge removed, pockets kept) is revolved 180° per side like the c-lines → `A_inst`.

Run logs in `OUTDIR/logs/`. `estimate_nozzle_center.py` reads `OUTDIR/threshold_XX/` (first threshold, or give one: `python3 estimate_nozzle_center.py OUTDIR 0.30`).
