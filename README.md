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

## Output (per flame, in `OUTDIR/<flame name>/`)
- `average composites/` – c̄ composite without / per / with all c-lines, `cbar_composite.npy`, `c_lines.npz`
- `instantaneous/` – stitched instantaneous fronts (PNG + `.npy` + frame info `.json`)
- `<flame>_results.xlsx` – A and h per c̄ level, area breakdown, input files
- `user_settings_log.txt` – settings used per position
- `posN/` – `cbar.npy`, instantaneous fronts, overlays, `done.json` (resume marker)

Combined over all flames: `OUTDIR/ALL_conditions_results.xlsx`; run logs in `OUTDIR/logs/`.
