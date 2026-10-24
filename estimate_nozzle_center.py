#!/usr/bin/env python3
"""
estimate_nozzle_center.py — estimate the nozzle centre (flame axis) of every
finished flame from its saved c̄ maps.  No raw .spe data needed.

Method
  1. Stack the flame's pos*/cbar.npy files (pos1 at the bottom), exactly like
     the composite in run_wabel_batch.py.
  2. Ray-scan the c̄ = LEVEL line from a start centre (NOZZLE_CENTER).
  3. In every row where both the left and the right arm were found, take the
     midpoint between them.  The top part of the line (near the tip, where the
     line closes) is ignored, because it is distorted by a wrong start centre.
  4. New centre = median of these midpoints.  Repeat steps 2–4 from the new
     centre until it stops moving, so the result does not depend on the
     (possibly wrong) start value.

Also reported: centre per position and the flame tilt (midpoint drifting
with height).  A check image is saved in each flame folder.

Usage (in the repo folder):
    python3 estimate_nozzle_center.py                  # uses OUTDIR from config_local.py
    python3 estimate_nozzle_center.py /path/to/results # any results folder (also old v7 output)
    python3 estimate_nozzle_center.py /path/to/results 0.30   # use the 30 % threshold c̄ maps
With several thresholds, the c̄ maps of the first one in THRESHOLD_FRAC are used
unless a threshold is given.
"""
import os, sys, re, glob, csv, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_wabel_batch as W          # same scan function + your config_local.py settings

LEVEL          = 0.5    # c̄ line used for the midpoint
IGNORE_TOP     = 0.25   # ignore this fraction of the line length near the tip
MIN_WIDTH_PX   = 20     # ignore rows where the two arms are closer than this
MAX_ITER       = 10
WARN_SHIFT_PX  = 5      # flag flames whose centre differs more than this from NOZZLE_CENTER


CBAR_NAME = "cbar.npy"    # set in main(): cbar_t25.npy etc. (v8.4+), cbar.npy (older)


def _cbar_path(posdir):
    for name in (CBAR_NAME, "cbar.npy"):
        f = os.path.join(posdir, name)
        if os.path.isfile(f):
            return f
    return None


def find_flame_dirs(root):
    """Folders that contain pos<N>/cbar*.npy (v8.4 per-threshold, older v8 and v7 output)."""
    found = []
    for d, subdirs, files in os.walk(root):
        if any(re.fullmatch(r"pos\d+", s) and _cbar_path(os.path.join(d, s)) for s in subdirs):
            found.append(d)
    return sorted(found)


def load_stack(fdir):
    pos = sorted(int(m.group(1)) for s in os.listdir(fdir)
                 for m in [re.fullmatch(r"pos(\d+)", s)]
                 if m and _cbar_path(os.path.join(fdir, s)))
    maps = {p: np.load(_cbar_path(os.path.join(fdir, f"pos{p}"))) for p in pos}
    widths = {m.shape[1] for m in maps.values()}
    if len(widths) > 1:
        raise ValueError(f"positions have different widths {widths}")
    strips, bounds, y = [], [], 0
    for p in reversed(pos):                      # top position first
        strips.append(maps[p]); bounds.append((y, y + maps[p].shape[0], p))
        y += maps[p].shape[0]
    return np.vstack(strips), bounds, pos


def midpoints(stack, nc):
    left, right, info = W.scan_cbar_contour_status(stack, int(round(nc)), LEVEL)
    L = {int(y): int(x) for x, y in left}
    R = {int(y): int(x) for x, y in right}
    ys = np.array(sorted(set(L) & set(R)))
    if len(ys) == 0:
        return None, info
    xl = np.array([L[y] for y in ys]); xr = np.array([R[y] for y in ys])
    keep = (xr - xl) >= MIN_WIDTH_PX
    if len(ys):
        y_tip, y_bot = ys.min(), ys.max()
        keep &= ys >= y_tip + IGNORE_TOP * (y_bot - y_tip)
    if keep.sum() < 10:
        return None, info
    return (ys[keep], (xl[keep] + xr[keep]) / 2.0), info


def estimate(stack, nc0):
    nc = float(nc0); hist = [nc]
    for it in range(MAX_ITER):
        mp, info = midpoints(stack, nc)
        if mp is None:
            return None
        new = float(np.median(mp[1]))
        hist.append(new)
        if abs(new - nc) < 0.5:
            nc = new; break
        nc = new
    return nc, mp, hist, info


def save_check_image(fdir, stack, bounds, nc_old, nc_new, mp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, stack.shape[0] / stack.shape[1] * 8), dpi=110)
    ax.imshow(stack, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ax.axvline(nc_old, color="goldenrod", ls="--", lw=1.2, label=f"NOZZLE_CENTER (config) = {nc_old:.0f}")
    ax.axvline(nc_new, color="lime", ls="-", lw=1.2, label=f"estimated centre = {nc_new:.0f}")
    ax.plot(mp[1], mp[0], ".", color="lime", ms=1.5, alpha=0.6)
    for y0, y1, p in bounds:
        if y0 > 0:
            ax.axhline(y0, color="orange", ls=":", lw=0.5)
        ax.text(stack.shape[1] - 5, y0 + 15, f"pos{p}", ha="right", va="top", fontsize=7, color="orange")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{os.path.basename(fdir)} — c̄={LEVEL} midpoints", fontsize=9)
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    global CBAR_NAME
    root = sys.argv[1] if len(sys.argv) > 1 else W.OUTDIR
    t = float(sys.argv[2]) if len(sys.argv) > 2 else W.thresholds()[0]
    CBAR_NAME = f"cbar_{W.thr_tag(t)}.npy"
    print(f"c̄ maps       : {CBAR_NAME} (threshold {W.thr_pct(t)} %), or cbar.npy for older results")
    if not os.path.isdir(root):
        print(f"ERROR: folder not found: {root}"); sys.exit(1)
    nc0 = W.NOZZLE_CENTER; res = W.SPATIAL_RES
    flames = find_flame_dirs(root)
    print(f"Results folder : {root}")
    print(f"Start centre   : NOZZLE_CENTER = {nc0} px   (c̄ = {LEVEL} line, top "
          f"{int(IGNORE_TOP * 100)} % near the tip ignored)")
    print(f"Flames found   : {len(flames)}\n")
    if not flames:
        print("No folders with pos*/cbar*.npy found."); sys.exit(1)

    rows = []
    hdr = f"{'flame':<40} {'centre':>7} {'shift':>7} {'shift':>7} {'tilt':>7}   per position"
    print(hdr); print(f"{'':<40} {'[px]':>7} {'[px]':>7} {'[mm]':>7} {'[deg]':>7}"); print("-" * 110)
    for fdir in flames:
        name = os.path.relpath(fdir, root)
        try:
            stack, bounds, pos = load_stack(fdir)
            r = estimate(stack, nc0)
            if r is None:
                print(f"{name:<40}  could not find a usable c̄={LEVEL} line"); continue
            nc, (ys, mids), hist, info = r
            # per position
            per = []
            for y0, y1, p in bounds:
                m = (ys >= y0) & (ys < y1)
                per.append((p, float(np.median(mids[m])) if m.sum() >= 5 else None))
            per.sort()
            # tilt: midpoint vs height (rows → px), positive = leaning right going up
            slope = np.polyfit(-ys.astype(float), mids, 1)[0]
            tilt = math.degrees(math.atan(slope))
            shift = nc - nc0
            flag = "  <-- check" if abs(shift) > WARN_SHIFT_PX else ""
            pstr = "  ".join(f"p{p}:{c:.0f}" if c is not None else f"p{p}:-" for p, c in per)
            print(f"{name:<40} {nc:7.1f} {shift:+7.1f} {shift * res:+7.2f} {tilt:+7.2f}   {pstr}{flag}")
            if info["status"] != "closed":
                print(f"{'':<40} (note: c̄={LEVEL} line status '{info['status']}')")
            save_check_image(fdir, stack, bounds, nc0, nc, (ys, mids),
                             os.path.join(fdir, "nozzle_center_check.png"))
            rows.append(([name, round(nc, 1), round(shift, 1), round(shift * res, 3),
                          round(tilt, 3), len(ys), len(hist) - 1],
                         {p: ("" if c is None else round(c, 1)) for p, c in per}))
        except Exception as e:
            print(f"{name:<40}  ERROR: {type(e).__name__}: {e}")

    out = os.path.join(root, "nozzle_centers.csv")
    maxpos = max((max(per) for _, per in rows), default=0)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flame", "centre_px", "shift_px", "shift_mm", "tilt_deg", "rows_used",
                    "iterations"] + [f"centre_pos{i}_px" for i in range(1, maxpos + 1)])
        for base, per in rows:
            w.writerow(base + [per.get(i, "") for i in range(1, maxpos + 1)])
    print(f"\nTable saved  → {out}")
    print("Check images → nozzle_center_check.png in each flame folder")
    print("(green line = estimated centre, dashed gold = current NOZZLE_CENTER)")


if __name__ == "__main__":
    main()
