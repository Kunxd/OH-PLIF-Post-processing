#!/usr/bin/env python3
"""
make_synthetic_results.py — a fake OH-PLIF results folder, in exactly the format that
ohplif1.py writes, for testing thesis_figures.py / thesis_matrices.py without real data.

  python3 make_synthetic_results.py OUTDIR [--res 0.1]

Writes OUTDIR/threshold_25/<flame>/ for the 15 conditions of the campaign
(CH4: d_j = 6/9/12 mm × Re_j = 15k/30k;  H2: d_j = 6/9/12 mm × Re_j = 15k/30k/60k):
  average composites/cbar_composite.npy, c_lines.npz       (c̄ field + ray-scanned c-lines)
  instantaneous/instant_raw_composite_<n>.npy, ..._info.json
  instantaneous_binarized/instant_binary_composite_<n>.npy, ..._info.json
and an empty OUTDIR/_run_data/.

The flames are invented: an unburnt core that narrows to a tip at height L, wrapped in a
burnt (OH) band whose outer edge widens downstream and, near the nozzle, is as wide as
the pilot (products out to r = 16 mm, narrowing downstream); both edges are wrinkled by a
multi-scale random field whose amplitude grows with height and Re_j (H2: more
small-scale wrinkling).  Every position of an instantaneous composite is its own random
realisation (as in the experiment, where positions were not recorded simultaneously),
so the stitched front shows small steps at the junctions.  Built-in test cases:
  * CH4_D6_Re30000: composite #1 has a gap in the burnt band → the front re-scan leaks
    (tests the "no closed front" path); composite #2 is fine.
  * H2_D12_Re60000: the measured positions end below the flame tip (open tip).
  * H2_D12_Re29874: Re in the name is not a round number (nearest-column mapping).
  * a few isolated burnt specks in the ambient air of every binary image.
"""
import os
import json
import math
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.stats import norm

LEVELS = [0.4, 0.5, 0.6, 0.7]
THR = 0.25
STEP_MM = 25.0          # one traverse step = height of one cropped position strip
OFFSET_MM = 20.0        # nozzle exit → bottom of pos1 (pos0 extension rows)
WIDTH_MM = 60.0
AXIS_MM = 29.0          # nozzle axis 29 mm from the left edge → image not centred on it
PILOT_R_MM = 16.0       # 32 mm pilot of the same mixture: OH-containing products around the jet
N_COMP = 2

CONDITIONS = [("CH4", d, re) for d in (6, 9, 12) for re in (15000, 30000)] + \
             [("H2", d, re) for d in (6, 9, 12) for re in (15000, 30000, 60000)]


def flame_name(fuel, d, re):
    if fuel == "H2" and d == 12 and re == 30000:
        re = 29874                                       # non-round Re in the name
    return f"{fuel}_D{d}_Re{re}_phi{'1' if fuel == 'CH4' else '0.4'}"


def flame_length(fuel, d, re):
    k = re / 1000.0
    return d * (6.0 + 0.25 * k) if fuel == "CH4" else d * (5.0 + 0.2 * k)


def n_positions(fuel, d, re, L):
    n = math.ceil((L + 12.0 - OFFSET_MM) / STEP_MM)
    if fuel == "H2" and d == 12 and re == 60000:
        n -= 2                                           # tip above the last position
    return max(1, n)


def mean_edges(z, fuel, d, re, L):
    """Mean core half-width R and outer OH-band half-width Ro [mm] at heights z."""
    zeta = z / L
    R = np.where(zeta < 1, 0.5 * d * np.clip(1 - zeta ** 1.3, 0, None) ** 0.6, -(z - L) * 0.5)
    Ro = np.maximum(0.5 * d + 2.5 + 0.06 * z,      # products of the jet flame
                    PILOT_R_MM - 0.04 * z)          # + pilot products (narrowing downstream)
    return R, Ro


def amplitude(z, fuel, d, re, L):
    return 1.25 * (0.25 + 0.9 * np.minimum(z / L, 1.2)) * (re / 15000) ** 0.35 * (d / 9) ** 0.3


def noise(rng, H, W, res, fuel):
    scales = ([(0.3, 0.6, 0.6), (1.0, 2.5, 0.5), (3.0, 7.0, 0.5)] if fuel == "H2"
              else [(0.4, 1.0, 0.3), (1.2, 3.0, 0.5), (3.0, 8.0, 0.65)])
    tot = np.zeros((H, W))
    for sr, sz, w in scales:
        f = gaussian_filter(rng.standard_normal((H, W)), (sz / res, sr / res), mode="reflect")
        tot += w * f / f.std()
    return tot / math.sqrt(sum(w * w for *_, w in scales))


def strip(rng, fuel, d, re, L, z_rows, r_cols, res, energy, leak=False):
    """One position: binary (burnt = True) and bg-subtracted raw signal."""
    H, W = len(z_rows), len(r_cols)
    Z = z_rows[:, None]
    Rabs = np.abs(r_cols)[None, :]
    R, Ro = mean_edges(Z, fuel, d, re, L)
    A = amplitude(Z, fuel, d, re, L)
    core = (R - Rabs + A * noise(rng, H, W, res, fuel)) > 0
    outer = (Ro - Rabs + 0.8 * A * noise(rng, H, W, res, fuel)) > 0
    burnt = outer & ~core
    if leak:                                             # cut the right-hand band
        row = H // 2
        right = r_cols > 0
        burnt[row - 4:row + 4, right] = False
    # a few isolated specks in the ambient air
    amb = np.argwhere(~outer)
    for i in rng.choice(len(amb), size=min(6, len(amb)), replace=False):
        y, x = amb[i]
        burnt[y:y + 2, x:x + 2] = True
    # raw: OH peak near the front, decays into the products; sheet profile; noise
    dist = distance_transform_edt(~core) * res             # distance from the unburnt core
    sig = burnt * (0.12 + 0.88 * np.exp(-dist / 1.5))     # OH peak at the front, low in products
    sig *= np.exp(-np.clip(Z - L, 0, None) / 80.0)
    sheet = 0.7 + 0.3 * np.sin(np.pi * (np.arange(H) + 0.5) / H)[:, None]
    lateral = 1.0 - 0.004 * (r_cols[None, :] - r_cols[0])
    raw = 1500 * energy * sig * sheet * lateral
    raw = gaussian_filter(raw, 0.7) + rng.normal(0, 40, raw.shape)
    return burnt, np.clip(raw, 0, None).astype(np.float32)


def cbar_field(fuel, d, re, L, z_rows, r_cols):
    """Analytic ensemble average: P(outside the core and inside the outer edge)."""
    Z = z_rows[:, None]
    Rabs = np.abs(r_cols)[None, :]
    R, Ro = mean_edges(Z, fuel, d, re, L)
    A = amplitude(Z, fuel, d, re, L)
    return norm.cdf((Rabs - R) / A) * norm.cdf((Ro - Rabs) / (0.8 * A))


def rayscan(cb, nc, level, close_px=5):
    """Bottom → top: first crossing c̄ ≥ level left and right of the axis; stops when
    the arms meet (closed tip).  Points (column, row)."""
    left, right = [], []
    ic = int(round(nc))
    for row in range(cb.shape[0] - 1, -1, -1):
        line = cb[row]
        if line[ic] >= level:
            break
        li = np.flatnonzero(line[:ic][::-1] >= level)
        ri = np.flatnonzero(line[ic:] >= level)
        if li.size == 0 or ri.size == 0:
            break
        l, r = ic - 1 - li[0], ic + ri[0]
        left.append((l, row))
        right.append((r, row))
        if r - l <= close_px:
            break
    return np.array(left, float).reshape(-1, 2), np.array(right, float).reshape(-1, 2)


def extend(arm, nc, d, res, h_real, ext_px, side):
    """Linear pos0 extension from the bottom of the measured arm to the nozzle lip."""
    if len(arm) == 0:
        return arm
    x0, y0 = arm[0]
    lip = nc + side * 0.5 * d / res
    rows = np.arange(h_real, h_real + ext_px)
    xs = x0 + (lip - x0) * (rows - y0) / (h_real + ext_px - 1 - y0)
    return np.vstack([np.column_stack([xs, rows])[::-1], arm])


def write_flame(root, fuel, d, re, res, seed):
    rng = np.random.default_rng(seed)
    name = flame_name(fuel, d, re)
    L = flame_length(fuel, d, re)
    npos = n_positions(fuel, d, re, L)
    fdir = os.path.join(root, "threshold_25", name)
    ad, idir, bdir = (os.path.join(fdir, s) for s in
                      ("average composites", "instantaneous", "instantaneous_binarized"))
    for p in (ad, idir, bdir):
        os.makedirs(p, exist_ok=True)
    W = int(round(WIDTH_MM / res))
    nc = AXIS_MM / res
    step = int(round(STEP_MM / res))
    ext_px = int(round(OFFSET_MM / res))
    r_cols = (np.arange(W) - nc) * res
    h_real = npos * step

    # rows top → bottom; z of composite row k (pos0 rows below)
    def z_of_rows(a, b):
        return (h_real + ext_px - 1 - np.arange(a, b)) * res

    # ── average composite
    cb_real = cbar_field(fuel, d, re, L, z_of_rows(0, h_real), r_cols)
    cb = np.vstack([cb_real, np.zeros((ext_px, W))])
    np.save(os.path.join(ad, "cbar_composite.npy"), cb)
    bounds = [(k * step, npos - k) for k in range(npos)]        # (row offset, pos) top → bottom
    dct = {"levels": np.array(LEVELS), "threshold_frac": np.array(THR),
           "nozzle_center_px": np.array(nc), "spatial_res_mm": np.array(res),
           "composite_h_real_px": np.array(h_real), "ext_px": np.array(ext_px),
           "nozzle_diam_mm": np.array(float(d)), "height_offset_mm": np.array(0.0),
           "positions_top_to_bottom": np.array([p for _, p in bounds]),
           "pos_row_offsets_px": np.array([y for y, _ in bounds])}
    H = cb.shape[0]
    for lv in LEVELS:
        l, r = rayscan(cb_real, nc, lv)
        for arm, pts, side in (("left", l, -1), ("right", r, +1)):
            full = extend(pts, nc, d, res, h_real, ext_px, side)
            dct[f"c{lv}_{arm}_rayscan_px"] = pts
            dct[f"c{lv}_{arm}_px"] = full
            dct[f"c{lv}_{arm}_mm"] = np.column_stack([(full[:, 0] - nc) * res,
                                                      (H - 1 - full[:, 1]) * res])
    np.savez(os.path.join(ad, "c_lines.npz"), **dct)

    # ── instantaneous composites
    for ci in range(1, N_COMP + 1):
        bi_strips, raw_strips, rows, used, srcs = [], [], [], {}, {}
        for k, (y0, p) in enumerate(bounds):
            leak = (name.startswith("CH4_D6_Re30000") and ci == 1 and p == 1)
            frame = int(rng.integers(0, 50))
            bi, raw = strip(rng, fuel, d, re, L, z_of_rows(y0, y0 + step), r_cols, res,
                            energy=rng.uniform(0.6, 1.3), leak=leak)
            bi_strips.append(bi)
            raw_strips.append(raw)
            rows.append({"row_offset": y0, "pos": p, "frame": frame})
            used[f"pos{p}"] = frame
            srcs[f"pos{p}"] = {"file": f"{name}_pos{p} synthetic.spe", "frame_in_file": frame}
        BI, RAW = np.vstack(bi_strips), np.vstack(raw_strips)
        np.save(os.path.join(bdir, f"instant_binary_composite_{ci}.npy"), BI)
        np.save(os.path.join(idir, f"instant_raw_composite_{ci}.npy"), RAW)
        info = {"flame": name, "composite": ci, "threshold_frac": THR,
                "frames_used": used, "source_frames": srcs, "junction_mismatch_px": {},
                "note": "SYNTHETIC test data (make_synthetic_results.py)",
                "rows_top_to_bottom": rows}
        for dd in (idir, bdir):
            with open(os.path.join(dd, f"instant_composite_{ci}_info.json"), "w") as f:
                json.dump(info, f, indent=1)
    return name, L, npos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--res", type=float, default=0.1, help="mm per pixel (default 0.1)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, "_run_data"), exist_ok=True)
    for i, (fuel, d, re) in enumerate(CONDITIONS):
        name, L, npos = write_flame(a.outdir, fuel, d, re, a.res, seed=1000 + i)
        print(f"{name:28s} L = {L:6.1f} mm   {npos} positions")


if __name__ == "__main__":
    main()
