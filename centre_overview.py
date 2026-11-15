#!/usr/bin/env python3
"""
centre_overview.py — where is the flame axis really?

Reads a results folder of main.py (or wabel_new.py) and gives an overview of the
measured flame centre of every flame, compared with the configured NOZZLE_CENTER.
Nothing is reprocessed, no .spe files are needed and no result file is changed.

Usage
    python3 centre_overview.py RESULTS_DIR
    python3 centre_overview.py RESULTS_DIR --level 0.5 --rows 5
    python3 centre_overview.py RESULTS_DIR --out /some/other/folder --no-flame-plots
    python3 centre_overview.py              # RESULTS_DIR = OUTDIR from config_local.py

RESULTS_DIR is the OUTDIR of the analysis, or any folder below it (one
threshold_25/ folder, one flame folder).  Every
.../<flame>/average composites/c_lines.npz below it is used (v8 layouts: v8.5+,
v8.4 and older).  Output goes to RESULTS_DIR/centre_overview/ unless --out is given.

What is measured
  main.py finds each c-line with a ray scan: in every row it starts at
  NOZZLE_CENTER and walks left and right until c̄ reaches the level.  In a row
  where both the left and the right point exist, their midpoint
  (x_left + x_right) / 2 is where the flame axis is in that row.  The scan stops
  by itself as soon as NOZZLE_CENTER is no longer inside the unburnt core, so in
  every stored row NOZZLE_CENTER lies inside the core and the midpoint does not
  depend on the configured value.

  near-nozzle centre   mean midpoint over the lowest --rows rows of pos1.
                       Same rows and formula as the "synthetic axis" main.py draws
                       with DEBUG_SYNTH_AXIS = True (DEBUG_SYNTH_AXIS_ROWS = 5).
  whole-flame centre   median midpoint over the flame, without the top 25 % of the
                       line near the tip and rows where the arms are < 20 px apart
                       (same rule as estimate_nozzle_center.py, without its re-scan).
  per-position centre  median midpoint inside each position (same rows as above).
  tilt                 straight-line fit of midpoint vs height; + = the axis leans
                       to the right going up.
  drift                sideways movement of that fit between the lowest and the
                       highest row used [mm].
  All shifts are measured from the NOZZLE_CENTER stored in each c_lines.npz;
  + = the measured centre lies to the right (larger column number).

Output (RESULTS_DIR/centre_overview/)
  centre_overview.xlsx         "Overview"  one row per flame and threshold (--level)
                               "Per level" one row per flame, threshold and c̄ level
  centre_midpoints.csv         every midpoint used (row, height, left, right, midpoint)
  centre_overview_t25.png      all flames: near-nozzle and whole-flame shift
  centre_profiles_t25.png      all flames: axis position vs height
  flames_t25/<flame>.png       per flame: c̄ map with the measured axis, and the
                               axis position vs height for every c̄ level
Needs numpy, matplotlib and openpyxl (all in requirements.txt).
"""
import os
import re
import sys
import csv
import math
import argparse

import numpy as np

# ── defaults (all can be changed on the command line) ────────────────────
REF_LEVEL        = 0.5    # c̄ level shown in the overview figures / Overview sheet
NEAR_NOZZLE_ROWS = 5      # = DEBUG_SYNTH_AXIS_ROWS in main.py
IGNORE_TOP       = 0.25   # = estimate_nozzle_center.py
MIN_WIDTH_PX     = 20     # = estimate_nozzle_center.py
WARN_SHIFT_PX    = 5      # flag: near-nozzle centre further than this from NOZZLE_CENTER
SMOOTH_ROWS      = 21     # running median along the height for the profile lines (plot only)

AVG_DIR  = "average composites"            # main.py AVG_DIR_NAME
OUT_NAME = "centre_overview"
SKIP_DIRS = {"_run_data", "code_used", "logs", "white_bg", OUT_NAME}

# main.py colours, so a c̄ level has the same hue in every figure:
LEVEL_COLORS  = ['black', 'limegreen', 'dodgerblue', 'orange', 'magenta',
                 'cyan', 'red', 'yellow']                 # c-lines on the c̄ map
MARKER_COLORS = ['dimgray', 'forestgreen', 'royalblue', 'darkorange', 'darkmagenta',
                 'darkcyan', 'darkred', 'gold']           # same hues, darker: on white
AXIS_COLOR = 'goldenrod'                                  # NOZZLE_CENTER line, as in main.py
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


# ══════════════════════════════════════════════════════════════════════
#  Reading
# ══════════════════════════════════════════════════════════════════════
def find_clines(root, out_dir):
    found = []
    out_dir = os.path.normpath(os.path.abspath(out_dir))
    for d, subdirs, files in os.walk(root):
        subdirs[:] = sorted(s for s in subdirs if s not in SKIP_DIRS
                            and os.path.normpath(os.path.abspath(os.path.join(d, s))) != out_dir)
        if os.path.basename(d) == AVG_DIR and "c_lines.npz" in files:
            found.append(os.path.join(d, "c_lines.npz"))
    return found


def flame_and_threshold_from_path(path):
    """v8.5+: threshold_25/<flame>/average composites   v8.4: <flame>/threshold_25/average
    composites   ≤ v8.3: <flame>/average composites."""
    parts = os.path.normpath(path).split(os.sep)
    i = len(parts) - 2                                  # index of "average composites"
    flame = parts[i - 1]
    thr = None
    if re.fullmatch(r"threshold_\d+(\.\d+)?", flame):   # v8.4 layout
        thr = float(flame.split("_", 1)[1]) / 100.0
        flame = parts[i - 2]
    else:
        for p in reversed(parts[:i - 1]):
            m = re.fullmatch(r"threshold_(\d+(?:\.\d+)?)", p)
            if m:
                thr = float(m.group(1)) / 100.0
                break
    return flame, thr


def _scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = z[key]
    return v.item() if v.ndim == 0 else v


def load_clines(path):
    z = np.load(path, allow_pickle=False)
    flame, thr_path = flame_and_threshold_from_path(path)
    thr = _scalar(z, "threshold_frac", None)
    thr = float(thr) if thr is not None and np.isfinite(thr) else thr_path
    levels = {}
    for k in z.files:
        m = re.fullmatch(r"c(.+)_left_rayscan_px", k)
        if m and f"c{m.group(1)}_right_rayscan_px" in z.files:
            levels[float(m.group(1))] = (np.asarray(z[k]).reshape(-1, 2),
                                         np.asarray(z[f"c{m.group(1)}_right_rayscan_px"]).reshape(-1, 2))
    diam = _scalar(z, "nozzle_diam_mm", np.nan)
    return {
        "path": path, "dir": os.path.dirname(path), "flame": flame, "thr": thr,
        "nc": float(_scalar(z, "nozzle_center_px")),
        "res": float(_scalar(z, "spatial_res_mm")),
        "h_real": int(_scalar(z, "composite_h_real_px")),
        "ext_px": int(_scalar(z, "ext_px", 0)),
        "height_offset": float(_scalar(z, "height_offset_mm", 0.0)),
        "diam": None if diam is None or not np.isfinite(diam) else float(diam),
        "positions": [int(p) for p in np.atleast_1d(_scalar(z, "positions_top_to_bottom"))],
        "offsets": [int(o) for o in np.atleast_1d(_scalar(z, "pos_row_offsets_px"))],
        "levels": dict(sorted(levels.items())),
        "mtime": os.path.getmtime(path),
    }


# ══════════════════════════════════════════════════════════════════════
#  Measuring
# ══════════════════════════════════════════════════════════════════════
def row_position(F, ys):
    """Position number of every composite row (offsets are listed top → bottom)."""
    offs = np.array(F["offsets"]); pos = np.array(F["positions"])
    order = np.argsort(offs)
    idx = np.searchsorted(offs[order], ys, side="right") - 1
    return pos[order][np.clip(idx, 0, len(offs) - 1)]


def height_mm(F, ys):
    """Same height definition as the c-lines in main.py (mm above the nozzle exit)."""
    H = F["h_real"] + F["ext_px"]
    return (H - 1 - np.asarray(ys, float)) * F["res"] + F["height_offset"]


def measure_level(F, left, right, n_rows):
    L = {int(y): int(x) for x, y in left}
    R = {int(y): int(x) for x, y in right}
    ys = np.array(sorted(set(L) & set(R)), dtype=int)
    r = {"ys": ys, "n_both": len(ys), "near": None, "near_n": 0, "whole": None,
         "whole_n": 0, "tilt": None, "drift": None, "zrange": None, "per_pos": {}}
    if len(ys) == 0:
        r.update(xl=np.empty(0), xr=np.empty(0), mid=np.empty(0), z=np.empty(0),
                 pos=np.empty(0, int), near_mask=np.zeros(0, bool), valid=np.zeros(0, bool),
                 wide=np.zeros(0, bool))
        return r
    xl = np.array([L[y] for y in ys], float); xr = np.array([R[y] for y in ys], float)
    mid = (xl + xr) / 2.0
    z = height_mm(F, ys)
    pos = row_position(F, ys)

    # near nozzle — identical to main.synthetic_axis_column()
    first_row = max(F["h_real"] - n_rows, max(F["offsets"]))
    near_mask = ys >= first_row
    if near_mask.any():
        r["near"] = float(np.mean(mid[near_mask])); r["near_n"] = int(near_mask.sum())

    # whole flame — estimate_nozzle_center.py row rule
    wide = (xr - xl) >= MIN_WIDTH_PX          # narrower rows: scan closed, midpoint = NOZZLE_CENTER
    valid = wide.copy()
    y_tip, y_bot = ys.min(), ys.max()
    valid &= ys >= y_tip + IGNORE_TOP * (y_bot - y_tip)
    if valid.sum() >= 10:
        r["whole"] = float(np.median(mid[valid])); r["whole_n"] = int(valid.sum())
        zp = z[valid] / F["res"]                                  # height in px
        slope = np.polyfit(zp, mid[valid], 1)[0]                  # px sideways per px up
        r["tilt"] = math.degrees(math.atan(slope))
        r["zrange"] = (float(z[valid].min()), float(z[valid].max()))
        r["drift"] = slope * (zp.max() - zp.min()) * F["res"]
        for p in F["positions"]:
            m = valid & (pos == p)
            if m.sum() >= 5:
                r["per_pos"][p] = float(np.median(mid[m]))
    r.update(xl=xl, xr=xr, mid=mid, z=z, pos=pos, near_mask=near_mask, valid=valid, wide=wide)
    return r


def measure(F, n_rows):
    F["m"] = {lv: measure_level(F, l, r, n_rows) for lv, (l, r) in F["levels"].items()}
    return F


def pick_level(F, want):
    if not F["levels"]:
        return None
    return min(F["levels"], key=lambda lv: abs(lv - want))


# ══════════════════════════════════════════════════════════════════════
#  Tables
# ══════════════════════════════════════════════════════════════════════
def _r(v, n=1):
    return "" if v is None else round(float(v), n)


def _shift(F, x, mm=False):
    if x is None:
        return None
    return (x - F["nc"]) * (F["res"] if mm else 1.0)


def thr_pct(t):
    return "-" if t is None else f"{t * 100:g}"


def thr_num(t):
    return "" if t is None else round(t * 100, 3)


def note_for(F, m):
    notes = []
    if m["near"] is None:
        notes.append("no row with both arms in the lowest pos1 rows")
    elif abs(m["near"] - F["nc"]) > WARN_SHIFT_PX:
        notes.append(f"check: near-nozzle centre > {WARN_SHIFT_PX} px from NOZZLE_CENTER")
    if m["whole"] is None:
        notes.append("too few rows for the whole-flame centre")
    return "; ".join(notes)


def overview_rows(flames, want, all_pos):
    hdr = ["flame", "threshold [%]", "c̄ level", "NOZZLE_CENTER [px]",
           "near-nozzle centre [px]", "near-nozzle shift [px]", "near-nozzle shift [mm]",
           "whole-flame centre [px]", "whole-flame shift [px]", "whole-flame shift [mm]",
           "tilt [deg]", "drift over height [mm]", "height used from [mm]", "height used to [mm]",
           "near-nozzle shift, all c̄ levels: min [px]", "near-nozzle shift, all c̄ levels: max [px]"]
    hdr += [f"pos{p} centre [px]" for p in all_pos] + ["note"]
    rows = []
    for F in flames:
        lv = pick_level(F, want)
        if lv is None:
            rows.append([F["flame"], thr_num(F["thr"]), "", F["nc"]] + [""] * (len(hdr) - 5)
                        + ["no c-lines in c_lines.npz"])
            continue
        m = F["m"][lv]
        nears = [_shift(F, F["m"][l]["near"]) for l in F["m"] if F["m"][l]["near"] is not None]
        zr = m["zrange"] or (None, None)
        rows.append([F["flame"], thr_num(F["thr"]), lv, _r(F["nc"]),
                     _r(m["near"]), _r(_shift(F, m["near"])), _r(_shift(F, m["near"], True), 3),
                     _r(m["whole"]), _r(_shift(F, m["whole"])), _r(_shift(F, m["whole"], True), 3),
                     _r(m["tilt"], 2), _r(m["drift"], 3), _r(zr[0], 2), _r(zr[1], 2),
                     _r(min(nears)) if nears else "", _r(max(nears)) if nears else ""]
                    + [_r(m["per_pos"].get(p)) for p in all_pos] + [note_for(F, m)])
    return hdr, rows


def level_rows(flames, all_pos):
    hdr = ["flame", "threshold [%]", "c̄ level", "NOZZLE_CENTER [px]",
           "near-nozzle centre [px]", "near-nozzle shift [px]", "near-nozzle shift [mm]",
           "near-nozzle rows",
           "whole-flame centre [px]", "whole-flame shift [px]", "whole-flame shift [mm]",
           "whole-flame rows", "tilt [deg]", "drift over height [mm]"]
    hdr += [f"pos{p} centre [px]" for p in all_pos] + ["note"]
    rows = []
    for F in flames:
        for lv, m in F["m"].items():
            rows.append([F["flame"], thr_num(F["thr"]), lv, _r(F["nc"]),
                         _r(m["near"]), _r(_shift(F, m["near"])), _r(_shift(F, m["near"], True), 3),
                         m["near_n"],
                         _r(m["whole"]), _r(_shift(F, m["whole"])), _r(_shift(F, m["whole"], True), 3),
                         m["whole_n"], _r(m["tilt"], 2), _r(m["drift"], 3)]
                        + [_r(m["per_pos"].get(p)) for p in all_pos] + [note_for(F, m)])
    return hdr, rows


def write_xlsx(path, sheets):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        base = os.path.splitext(path)[0]
        for name, hdr, rows in sheets:
            with open(f"{base}_{name.replace(' ', '_')}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
        return base + "_*.csv (openpyxl missing)"
    wb = Workbook(); first = True
    for name, hdr, rows in sheets:
        ws = wb.active if first else wb.create_sheet(); first = False
        ws.title = name[:31]
        ws.append(hdr)
        for c in ws[1]:
            c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
        for r in rows:
            ws.append(r)
        ws.freeze_panes = "B2"
        for i, h in enumerate(hdr, 1):
            w = 40 if h == "flame" else (50 if h == "note" else max(10, min(18, len(h) // 2 + 4)))
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.row_dimensions[1].height = 45
    tmp = path + ".tmp.xlsx"
    wb.save(tmp); os.replace(tmp, path)
    return path


def write_midpoints_csv(path, flames, n_rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flame", "threshold_pct", "cbar_level", "position", "row_px", "height_mm",
                    "x_left_px", "x_right_px", "midpoint_px", "shift_px", "shift_mm",
                    "near_nozzle_row", "whole_flame_row", "nozzle_center_px"])
        for F in flames:
            for lv, m in F["m"].items():
                for k in range(len(m["ys"])):
                    w.writerow([F["flame"], thr_pct(F["thr"]), lv, int(m["pos"][k]), int(m["ys"][k]),
                                round(float(m["z"][k]), 3), int(m["xl"][k]), int(m["xr"][k]),
                                m["mid"][k], m["mid"][k] - F["nc"],
                                round((m["mid"][k] - F["nc"]) * F["res"], 4),
                                int(m["near_mask"][k]), int(m["valid"][k]), F["nc"]])


# ══════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def level_colors(F, lv):
    i = sorted(F["levels"]).index(lv)
    return LEVEL_COLORS[i % len(LEVEL_COLORS)], MARKER_COLORS[i % len(MARKER_COLORS)]


def running_median(v, n):
    if len(v) < n or n < 3:
        return v
    h = n // 2
    pad = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
    win = np.lib.stride_tricks.sliding_window_view(pad, n)
    return np.median(win, axis=1)


def plot_profile_lines(ax, F, m, color, lw=1.8):
    """Raw midpoints (faint), running median: solid where used, dashed in the tip part.
    Rows narrower than MIN_WIDTH_PX are left out: there the scan has closed and the
    'midpoint' is just NOZZLE_CENTER, not a measurement."""
    w = m["wide"]
    if not w.any():
        return []
    s = (m["mid"][w] - F["nc"]) * F["res"]; z = m["z"][w]; v = m["valid"][w]
    sm = running_median(s, SMOOTH_ROWS)
    ax.plot(s, z, color=color, lw=0.5, alpha=0.25)
    ax.plot(np.where(v, sm, np.nan), z, color=color, lw=lw)
    ax.plot(np.where(~v, sm, np.nan), z, color=color, lw=lw * 0.65, ls=(0, (2, 2)))
    return list(s)


def signed(v, d=2):
    """+0.00 instead of -0.00"""
    return f"{round(v, d) + 0.0:+.{d}f}"


def _short(name, n=34):
    return name if len(name) <= n else name[:n - 1] + "…"


def plot_overview(flames, want, thr, path):
    plt = _plt()
    from matplotlib.lines import Line2D
    n = len(flames)
    fig_h = 1.9 + 0.42 * n
    fig, ax = plt.subplots(figsize=(10, fig_h))
    res_all = {F["res"] for F in flames}
    vals = []
    ref_mcol = MARKER_COLORS[1]
    for i, F in enumerate(flames):
        lv = pick_level(F, want)
        if lv is None:
            continue
        m = F["m"][lv]
        _, mcol = level_colors(F, lv); ref_mcol = mcol
        nears = [_shift(F, F["m"][l]["near"], True) for l in F["m"] if F["m"][l]["near"] is not None]
        if len(nears) > 1:
            ax.plot([min(nears), max(nears)], [i, i], color="#c3c2b7", lw=5,
                    solid_capstyle="round", zorder=2)
        ws = _shift(F, m["whole"], True)
        ns = _shift(F, m["near"], True)
        if ws is not None:
            ax.plot(ws, i, "o", ms=8, mfc="white", mec=mcol, mew=1.8, zorder=3)
            vals.append(ws)
        if ns is not None:
            ax.plot(ns, i, "o", ms=8, color=mcol, mec="white", mew=1.2, zorder=4)
            vals.append(ns)
        flag = ns is not None and abs(m["near"] - F["nc"]) > WARN_SHIFT_PX
        txt = "near nozzle: –" if ns is None else (f"near nozzle {signed(ns)} mm "
                                                    f"({signed(m['near'] - F['nc'], 1)} px)")
        if m["tilt"] is not None:
            txt += f"   tilt {signed(m['tilt'])}°"
        ax.annotate(txt, xy=(1.01, i), xycoords=("axes fraction", "data"), va="center",
                    fontsize=8.5, color=INK if flag else INK2,
                    fontweight="bold" if flag else "normal")
    lim_band = WARN_SHIFT_PX * min(res_all)
    lim = max([abs(v) for v in vals] + [lim_band]) * 1.25
    ax.axvspan(-lim_band, lim_band, color="#f0efec", zorder=0, lw=0)
    ax.axvline(0, color=AXIS_COLOR, ls="--", lw=1.2, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([_short(F["flame"]) for F in flames])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("measured centre − NOZZLE_CENTER [mm]   (+ = to the right)")
    if len(res_all) == 1:
        r = next(iter(res_all))
        sec = ax.secondary_xaxis("top", functions=(lambda mm: mm / r, lambda px: px * r))
        sec.set_xlabel("[px]", color=INK2)
        sec.tick_params(colors=INK2)
    ncs = sorted({F["nc"] for F in flames})
    nc_txt = f"{ncs[0]:g} px" if len(ncs) == 1 else "as stored per flame"
    ax.set_title(f"Where is the flame axis?   threshold {thr_pct(thr)} %,  c̄ = {want:g} line,  "
                 f"NOZZLE_CENTER = {nc_txt}", loc="left", fontsize=10, pad=12)
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color=ref_mcol, mec="white",
               label=f"near nozzle (lowest {NEAR_NOZZLE_ROWS} rows of pos1)"),
        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=ref_mcol, mew=1.8,
               label="whole flame (median, tip excluded)"),
        Line2D([], [], color="#c3c2b7", lw=5, label="near nozzle, range over all c̄ levels"),
        Line2D([], [], color=AXIS_COLOR, ls="--", lw=1.2, label="NOZZLE_CENTER (config)"),
        Line2D([], [], color="#f0efec", lw=8, label=f"±{WARN_SHIFT_PX} px"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0),
               ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.55 / fig_h, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_profiles(flames, want, thr, path):
    plt = _plt()
    n = len(flames)
    ncols = min(5, n); nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 3.6 * nrows + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    xs, zs = [0.0], [0.0]
    for F in flames:
        lv = pick_level(F, want)
        if lv is not None and F["m"][lv]["wide"].any():
            m = F["m"][lv]; w = m["wide"]
            xs += list((m["mid"][w] - F["nc"]) * F["res"]); zs += list(m["z"][w])
    lim = max(np.percentile(np.abs(xs), 99.5) * 1.3, WARN_SHIFT_PX * flames[0]["res"] * 1.5)
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off"); continue
        F = flames[k]; lv = pick_level(F, want)
        ax.axvline(0, color=AXIS_COLOR, ls="--", lw=1)
        for o in F["offsets"]:
            if o > 0:
                ax.axhline(height_mm(F, [o - 0.5])[0], color="#c3c2b7", lw=0.6, ls=":")
        ax.set_title(_short(F["flame"], 30), fontsize=8, loc="left")
        if lv is None or not len(F["m"][lv]["mid"]):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=INK2)
            continue
        m = F["m"][lv]; _, mcol = level_colors(F, lv)
        plot_profile_lines(ax, F, m, mcol)
        if m["near"] is not None:
            ax.plot(_shift(F, m["near"], True), m["z"][m["near_mask"]].mean(), "o", ms=7,
                    color=mcol, mec="white", mew=1.2, zorder=5)
        info = []
        if m["near"] is not None:
            info.append(f"nozzle {signed(_shift(F, m['near'], True))} mm")
        if m["tilt"] is not None:
            info.append(f"tilt {signed(m['tilt'])}°")
        ax.text(0.03, 0.98, "\n".join(info), transform=ax.transAxes, va="top", fontsize=7.5,
                color=INK2, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    axes[0, 0].set_xlim(-lim, lim)
    axes[0, 0].set_ylim(0, max(zs) * 1.02)
    for ax in axes[-1, :]:
        ax.set_xlabel("centre − NOZZLE_CENTER [mm]")
    for ax in axes[:, 0]:
        ax.set_ylabel("height above nozzle [mm]")
    fig.suptitle(f"Flame axis vs height   threshold {thr_pct(thr)} %,  c̄ = {want:g}.   "
                 f"Solid: rows used for the whole-flame centre; dashed: tip part (excluded); "
                 f"dot: near-nozzle centre; dotted lines: position boundaries",
                 x=0.01, ha="left", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140); plt.close(fig)


def plot_flame(F, want, path):
    plt = _plt()
    from matplotlib.lines import Line2D
    comp_path = os.path.join(F["dir"], "cbar_composite.npy")
    comp = np.load(comp_path) if os.path.isfile(comp_path) else None
    lv_ref = pick_level(F, want)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.1, 1]})
    # left: c̄ map, c-line of the reference level, its midpoints, NOZZLE_CENTER
    if comp is not None:
        a1.imshow(comp, cmap="RdBu_r", vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
        a1.set_xlim(0, comp.shape[1]); a1.set_ylim(comp.shape[0], 0)
    else:
        a1.text(0.5, 0.5, "cbar_composite.npy not found", transform=a1.transAxes, ha="center")
    a1.axis("off"); a1.grid(False)
    a1.axvline(F["nc"], color=AXIS_COLOR, ls="--", lw=1.2)
    for o, p in zip(F["offsets"], F["positions"]):
        if o > 0:
            a1.axhline(o, color="orange", ls=":", lw=0.6)
        if comp is not None:
            a1.text(comp.shape[1] - 5, o + 15, f"pos{p}", ha="right", va="top", fontsize=7, color="orange")
    if lv_ref is not None and len(F["m"][lv_ref]["mid"]):
        m = F["m"][lv_ref]; lcol, _ = level_colors(F, lv_ref)
        l, r = F["levels"][lv_ref]
        for pts in (l, r):
            if len(pts) > 1:
                a1.plot(pts[:, 0], pts[:, 1], color=lcol, lw=1.2)
        a1.plot(m["mid"][m["wide"]], m["ys"][m["wide"]], ".", color=lcol, ms=1.2)
        if m["near"] is not None:
            y0 = m["ys"][m["near_mask"]].max()
            a1.plot([m["near"], m["near"]], [y0 - 80, y0], color=lcol, lw=2.5)
    a1.set_title(f"c̄ map, c̄ = {lv_ref:g} line and its midpoints" if lv_ref is not None else "c̄ map",
                 fontsize=9, loc="left")
    # right: centre vs height, every level
    handles = [Line2D([], [], color=AXIS_COLOR, ls="--", lw=1.2,
                      label=f"NOZZLE_CENTER = {F['nc']:g} px")]
    a2.axvline(0, color=AXIS_COLOR, ls="--", lw=1.2)
    for o in F["offsets"]:
        if o > 0:
            a2.axhline(height_mm(F, [o - 0.5])[0], color="#c3c2b7", lw=0.6, ls=":")
    allx = [0.0]
    for lv, m in F["m"].items():
        if not len(m["mid"]):
            continue
        _, mcol = level_colors(F, lv)
        allx += plot_profile_lines(a2, F, m, mcol)
        lab = f"c̄ = {lv:g}:"
        if m["near"] is not None:
            a2.plot(_shift(F, m["near"], True), m["z"][m["near_mask"]].mean(), "o", ms=7,
                    color=mcol, mec="white", mew=1.2, zorder=5)
            lab += f" nozzle {signed(_shift(F, m['near'], True))} mm"
        if m["whole"] is not None:
            lab += f", flame {signed(_shift(F, m['whole'], True))} mm, tilt {signed(m['tilt'])}°"
        handles.append(Line2D([], [], color=mcol, lw=1.8, marker="o", ms=6, mec="white", label=lab))
    lim = max(np.percentile(np.abs(allx), 99.5) * 1.3, WARN_SHIFT_PX * F["res"] * 1.5)
    a2.set_xlim(-lim, lim); a2.set_ylim(0, None)
    a2.set_xlabel("measured centre − NOZZLE_CENTER [mm]   (+ = to the right)")
    a2.set_ylabel("height above nozzle [mm]")
    sec = a2.secondary_xaxis("top", functions=(lambda mm: mm / F["res"], lambda px: px * F["res"]))
    sec.set_xlabel("[px]", color=INK2); sec.tick_params(colors=INK2)
    handles += [Line2D([], [], color=INK2, lw=1.8, label="rows used for the whole-flame centre"),
                Line2D([], [], color=INK2, lw=1.1, ls=(0, (2, 2)), label="tip part (excluded)"),
                Line2D([], [], color=INK2, lw=0.5, alpha=0.4,
                       label=f"raw midpoints (lines: running median of {SMOOTH_ROWS} rows)")]
    a2.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.9)
    fig.suptitle(f"{F['flame']}   threshold {thr_pct(F['thr'])} %", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def default_root():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from config_local import OUTDIR      # noqa
        return OUTDIR
    except Exception:
        return None


def main():
    global NEAR_NOZZLE_ROWS
    ap = argparse.ArgumentParser(description="Overview of the measured flame centres in a "
                                             "main.py results folder.")
    ap.add_argument("results_dir", nargs="?", help="OUTDIR of the analysis (default: OUTDIR "
                                                   "from config_local.py)")
    ap.add_argument("--level", type=float, default=REF_LEVEL,
                    help=f"c̄ level for the overview figures and Overview sheet (default {REF_LEVEL})")
    ap.add_argument("--rows", type=int, default=NEAR_NOZZLE_ROWS,
                    help=f"lowest pos1 rows for the near-nozzle centre (default {NEAR_NOZZLE_ROWS}, "
                         f"= DEBUG_SYNTH_AXIS_ROWS)")
    ap.add_argument("--out", help="output folder (default RESULTS_DIR/centre_overview)")
    ap.add_argument("--no-flame-plots", action="store_true", help="skip the per-flame figures")
    a = ap.parse_args()
    NEAR_NOZZLE_ROWS = a.rows

    root = a.results_dir or default_root()
    if not root:
        sys.exit("Give the results folder:  python3 centre_overview.py /path/to/OUTDIR")
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        sys.exit(f"ERROR: folder not found: {root}")
    out = os.path.abspath(a.out) if a.out else os.path.join(root, OUT_NAME)

    paths = find_clines(root, out)
    print(f"Results folder : {root}")
    print(f"c_lines.npz    : {len(paths)} found")
    if not paths:
        sys.exit(f"No '{AVG_DIR}/c_lines.npz' below {root}. Run main.py with DO_CBAR = True first.")

    seen = {}
    for p in paths:
        try:
            F = measure(load_clines(p), NEAR_NOZZLE_ROWS)
        except Exception as e:
            print(f"  skipped {os.path.relpath(p, root)}: {type(e).__name__}: {e}")
            continue
        key = (F["flame"], F["thr"])
        if key in seen:                                  # same flame + threshold twice → newest
            if F["mtime"] > seen[key]["mtime"]:
                seen[key] = F
            print(f"  {F['flame']} (threshold {thr_pct(F['thr'])} %) found twice → using "
                  f"{os.path.relpath(seen[key]['path'], root)}")
            continue
        seen[key] = F
    flames = sorted(seen.values(), key=lambda F: (F["thr"] if F["thr"] is not None else -1,
                                                  natural_key(F["flame"])))
    if not flames:
        sys.exit("No usable c_lines.npz.")
    os.makedirs(out, exist_ok=True)

    all_pos = sorted({p for F in flames for p in F["positions"]})
    sheets = [("Overview",) + overview_rows(flames, a.level, all_pos),
              ("Per level",) + level_rows(flames, all_pos)]
    xl = write_xlsx(os.path.join(out, "centre_overview.xlsx"), sheets)
    write_midpoints_csv(os.path.join(out, "centre_midpoints.csv"), flames, NEAR_NOZZLE_ROWS)

    # console table
    print(f"\nc̄ = {a.level:g},  near nozzle = lowest {NEAR_NOZZLE_ROWS} rows of pos1,  "
          f"+ = measured centre to the right of NOZZLE_CENTER\n")
    print(f"{'flame':<40} {'thr':>4} {'NOZZLE_C':>8} {'near':>7} {'shift':>7} {'shift':>7} "
          f"{'flame':>7} {'shift':>7} {'tilt':>6}")
    print(f"{'':<40} {'[%]':>4} {'[px]':>8} {'[px]':>7} {'[px]':>7} {'[mm]':>7} "
          f"{'[px]':>7} {'[mm]':>7} {'[deg]':>6}")
    print("-" * 102)
    f1 = lambda v, w, d=1: f"{round(v, d) + 0.0:>{w}.{d}f}" if v is not None else f"{'–':>{w}}"
    for F in flames:
        lv = pick_level(F, a.level)
        if lv is None:
            print(f"{_short(F['flame'], 40):<40} {thr_pct(F['thr']):>4}   no c-lines"); continue
        m = F["m"][lv]
        flag = "  <-- check" if m["near"] is not None and abs(m["near"] - F["nc"]) > WARN_SHIFT_PX else ""
        print(f"{_short(F['flame'], 40):<40} {thr_pct(F['thr']):>4} {F['nc']:>8.0f} "
              f"{f1(m['near'], 7)} {f1(_shift(F, m['near']), 7, 1)} {f1(_shift(F, m['near'], True), 7, 2)} "
              f"{f1(m['whole'], 7)} "
              f"{f1(_shift(F, m['whole'], True), 7, 2)} {f1(m['tilt'], 6, 2)}{flag}")

    # figures, one set per threshold
    by_thr = {}
    for F in flames:
        by_thr.setdefault(F["thr"], []).append(F)
    for t, Fs in by_thr.items():
        tag = f"t{thr_pct(t)}" if t is not None else "t-"
        plot_overview(Fs, a.level, t, os.path.join(out, f"centre_overview_{tag}.png"))
        plot_profiles(Fs, a.level, t, os.path.join(out, f"centre_profiles_{tag}.png"))
        if not a.no_flame_plots:
            fd = os.path.join(out, f"flames_{tag}"); os.makedirs(fd, exist_ok=True)
            for F in Fs:
                plot_flame(F, a.level, os.path.join(fd, re.sub(r"[^\w.\-]+", "_", F["flame"]) + ".png"))

    print(f"\nTable    → {xl}")
    print(f"Points   → {os.path.join(out, 'centre_midpoints.csv')}")
    print(f"Figures  → {out}  (centre_overview_*.png, centre_profiles_*.png"
          + ("" if a.no_flame_plots else ", flames_*/") + ")")


if __name__ == "__main__":
    main()
