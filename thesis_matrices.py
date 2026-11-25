#!/usr/bin/env python3
"""
thesis_matrices.py — all flames side by side, from finished ohplif1.py / ohplif2.py results.

Companion of thesis_figures.py: the same pictures (drawn with thesis_figures.py's own
functions and settings, so colours, fronts and c-lines are identical), but one figure
holds every operating condition.  Nothing is reprocessed and no result file is changed.

1) Condition matrices (like the OH* photo matrix of the campaign)
     rows    nozzle diameter d_j = 6, 9, 12 mm                        (D_ROWS)
     columns CH4: Re_j = 15k, 30k  |  H2: Re_j = 15k, 30k, 60k        (FUELS, RE_COLUMNS)
   One matrix per picture type of thesis_figures.py:
     matrix_instant_raw        bg-subtracted OH-PLIF signal of the stitched frames
     matrix_instant_binary     the same frames binarised (burnt 1 white, unburnt 0 black)
     matrix_instant_front      flame front (red) + pockets (blue) on the burnt gas (grey)
     matrix_instant_front_red  front + pockets (blue) on the OH-PLIF signal in red
     matrix_average_cbar       c̄ field only
     matrix_average_cbar_<lv>  c̄ field + the c-line of one level (one matrix per level)
     matrix_average_all_clines c̄ field + all c-lines
   The instantaneous matrices show the SAME frames in each cell (COMPOSITE).
   All cells share one mm scale: the nozzle axis is centred, the heights line up, and
   every cell is trimmed to the same ± half-width (MATRIX_TRIM).  The c̄ matrices
   include pos0 (grey "not measured" band, estimated c-line in grey) when SHOW_POS0 is on
   in thesis_figures.py.  Right of each row: z [mm]; under the bottom row: r [mm].

2) Position-window figures, in the style of multi-window OH-PLIF overviews in the
   literature (one column per condition, one panel per measurement position, pos1 at
   the bottom, the flame front as a blue line on the OH signal in white → red).  Only
   flames whose front re-scan closed.  A column ends a little above its flame tip
   (CROP_TO_TIP).  Every panel sits on its flame's nozzle axis: nothing is shifted
   sideways.  One figure per fuel (columns grouped by d_j; STAGGER_GROUPS = "fuel_d"
   adds one per fuel and nozzle).  Version "filled" is drawn (STAGGER_VERSIONS; also
   available: "natural"):
     positions_front_<fuel>[_d<n>]_natural  no gap in pos1: each column is as wide as its
         own flame's burnt gas in pos1 (left and right of the axis separately), so the
         flames touch there; higher up each panel is as wide as the burnt gas at its
         height (never wider than in pos1), so the flames may separate.
     positions_front_<fuel>[_d<n>]_filled   the flames as close together as their fronts
         allow: all columns as wide as the widest flame front of the figure (+ 0.5 mm;
         pockets not counted, cut off outside), lined up on their axes (FILLED_WIDTH =
         "column": every column as narrow as its own front); ambient air inside a panel
         is painted in a flat red (FILL_VALUE: the signal at the edge of the burnt gas) —
         not measured OH, say so in the caption.

Choosing the frames (COMPOSITE, --composite)
  "auto": per flame the lowest composite #n whose front re-scan closes, so raw, binary
          and front show the same frames; if none closes, #1 is used and the front cell
          says so.
  a number: that composite for every flame;  "all": one set of instantaneous matrices
          per composite number (to compare them).
  Swapping one cell (SWAP, --swap, or make_matrices(swap=…)): if the frame picked for a
  cell is not nice, look through thesis_figures/threshold_XX/<flame>/ and give the
  picture you want instead; its composite number is used for that cell in every
  instantaneous matrix and the position-window figure:
      SWAP = {"H2, 9, 30k": "instant_2_front.png", "CH4, 6, 15k": 3}
      python3 thesis_matrices.py thesis_figures --swap "H2,9,30k=instant_2_front.png"
      >>> import thesis_matrices as tm
      >>> tm.make_matrices("…/thesis_figures", swap={"H2, 9, 30k": 2})
  Cell = "fuel, d_j, Re" (Re as 30000 or 30k); picture = a file name or path containing
  "instant_<n>", or just the number n.

Which flame goes into which cell
  Read from the flame folder name: fuel "CH4"/"H2" (or "methane"/"hydrogen"; if neither
  is in the name, from "phi<x>" with PHI_FUEL), nozzle "D<n>" (as ohplif1.py), and
  "Re<n>" (also "Re30k").  The Re in the name may differ from the column value by
  RE_TOLERANCE (e.g. Re29874 → 30k column).  Check the table printed at the start
  (--list prints only the table).  Wrong or ambiguous names: CONDITION_OVERRIDES.

Usage
  python3 thesis_matrices.py FIGURES_DIR                 # FIGURES_DIR = thesis_figures output
  python3 thesis_matrices.py FIGURES_DIR --list          # only show flame → cell table
  python3 thesis_matrices.py FIGURES_DIR --swap "H2,9,30k=instant_2_front.png"
  python3 thesis_matrices.py FIGURES_DIR --threshold 25 --formats pdf
  python3 thesis_matrices.py                             # = OUTDIR/thesis_figures (config_local.py)
The pictures are drawn again from the analysis results that thesis_figures.py used (the
folder above FIGURES_DIR, or --results), so every cell has the same mm scale and the
lines stay vectors in the PDF.
Output: FIGURES_DIR/matrices/threshold_XX/  (or --out) and thesis_matrices_log.txt with
the cell table, the composite used per flame and every cell left empty and why.
Everything is redrawn on every run (a few figures only).
Needs thesis_figures.py in the same folder.
"""
import os
import re
import sys
import math
import argparse
import datetime
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import thesis_figures as tf          # noqa: E402  (settings + drawing helpers, no side effects)

# ╔════════════════════════════════════════════════════════════════════╗
# ║                FIGURE SETTINGS — EDIT HERE                         ║
# ╚════════════════════════════════════════════════════════════════════╝
FORMATS = ["png", "pdf"]

# ── conditions ───────────────────────────────────────────────────────
FUELS        = ["CH4", "H2"]                              # column groups, left → right
RE_COLUMNS   = {"CH4": [15000, 30000], "H2": [15000, 30000, 60000]}
D_ROWS       = [6, 9, 12]                                 # rows, top → bottom [mm]
RE_TOLERANCE = 0.15        # Re in the name may differ this much (fraction) from the column
FUEL_PATTERNS = {"CH4": r"(?<![a-z0-9])ch4(?![0-9])|methane",
                 "H2":  r"(?<![a-z0-9])h2(?![0-9o])|hydrogen"}
PHI_FUEL     = {1.0: "CH4", 0.4: "H2"}   # fuel from "phi<x>" when the name has no fuel
CONDITION_OVERRIDES = {}   # exact folder name → (fuel, d_j, Re) or None (= leave out), e.g.
                           # {"H2_D9_Re30000_repeat": None, "run7": ("H2", 9, 30000)}
FUEL_LABELS  = {"CH4": r"CH$_4$", "H2": r"H$_2$"}

# ── which frames ─────────────────────────────────────────────────────
COMPOSITE        = "auto"  # "auto", a number, or "all" (see the header)
SWAP             = {}      # cell → picture to use instead, e.g.
                           # {"H2, 9, 30k": "instant_2_front.png", "CH4, 6, 15k": 3}

# ── condition matrices ───────────────────────────────────────────────
MATRIX_WIDTH_IN      = 6.3    # figure width [in] (text width of an A4 thesis ≈ 6.3)
MATRIX_MAX_HEIGHT_IN = 8.6    # made narrower if it would be taller than this
CELL_GAP_IN          = 0.05   # between cells
GROUP_GAP_IN         = 0.22   # between the fuel groups
MATRIX_TRIM          = "content"  # common ± half-width of every cell:
                              #   "content": widest burnt region / c̄ field of all flames
                              #   "front":   widest flame front / c-line of all flames
                              #              (pockets are not counted; outside → cut off)
                              #   a number:  that many mm
TRIM_MARGIN_MM       = 2.0    # free space added beside the widest flame ("content"/"front")
Z_MAX_MM             = None   # top of every cell [mm]; None = top of the tallest composite
MATRIX_TICKS         = True   # z [mm] right of each row, r [mm] under the bottom row
CONTENT_CBAR         = 0.05   # "content" trim: c̄ above this counts as flame
CONTENT_MIN_BLOB     = 0.005  # "content" trim: burnt blobs smaller than this fraction of
                              #   the largest one (noise specks) are ignored
EMPTY_TEXT           = "no data"

# ── position-window figures ──────────────────────────────────────────
STAGGER              = True
STAGGER_GROUPS       = ["fuel"]   # "fuel": one figure per fuel (columns grouped
                                            # by d_j); "fuel_d": one per fuel and nozzle
STAGGER_VERSIONS     = ["filled"]
    # "natural": no gap in pos1 — every column is as wide as its own flame's burnt gas in
    #   pos1 (left and right of the nozzle axis separately), so neighbouring flames touch
    #   there.  Higher up each panel is as wide as the burnt gas at its height (never wider
    #   than in pos1), so the flames may separate.
    # "filled": the flames as close together as their fronts allow: every column is as
    #   wide as the widest flame front (pockets not counted) + FRONT_CLEARANCE_MM, centred
    #   on the nozzle axis; ambient air inside a panel is painted red (FILL_VALUE).
    #   The painted area is not measured OH — say so in the caption.
    # No frame is ever shifted sideways: every panel sits on its flame's nozzle axis.
NATURAL_PERCENTILE   = 5.0       # width of the burnt gas in a panel: this percentile over
                                 #   its rows (5 → the burnt gas reaches the edge in 95 %)
FRONT_CLEARANCE_MM   = 0.5       # space between the outermost front point and the panel edge
FILLED_WIDTH         = "figure"  # "filled": "figure" = all columns of a figure equally wide
                                 #   (its widest front), lined up; "column" = every column as
                                 #   wide as its own front, left and right separately (closest
                                 #   packing, columns differ in width)
FILL_VALUE           = "border"  # "filled": colour of the painted air: "border" = median OH
                                 #   signal of the burnt gas within FILL_BORDER_MM of the air
                                 #   (blends in at the edge of the burnt gas), or 0 … 1
FILL_BORDER_MM       = 1.0
STAGGER_WIDTH_IN     = 6.3       # at most this wide …
STAGGER_MAX_HEIGHT_IN = 8.6      # … and this tall
STAGGER_MAX_COL_IN   = 0.9       # … and columns at most this wide (figures with few columns)
STAGGER_CMAP         = None      # None: OH_RED_COLORS of thesis_figures.py (as _front_red)
STAGGER_FRONT_COLOR  = None      # None: RED_FRONT_COLOR / RED_POCKET_COLOR of thesis_figures.py
STAGGER_POCKET_COLOR = None
STAGGER_LW           = 0.6       # [pt]
CROP_TO_TIP          = True      # a column ends TIP_MARGIN_MM above the tip of its front
TIP_MARGIN_MM        = 3.0       #   (pockets above that are cut off; nothing is cut when
                                 #   the front reaches the top of the composite)
MIN_PANEL_MM         = 8.0       # a cut top panel is at least this tall; a panel above the
                                 #   tip (only margin, no front) is left out
PANEL_LW             = 0.5       # panel frame [pt]
GROUP_LINE_LW        = 1.2       # frame line between two d_j groups [pt]; = PANEL_LW for none

MAX_PPI   = 400   # images are reduced to at most this many pixels per inch of the figure
                  # (keeps the PDFs small); fronts and c-lines are always full resolution
# ╔════════════════════════════════════════════════════════════════════╗
# ║                END OF SETTINGS                                     ║
# ╚════════════════════════════════════════════════════════════════════╝
OUT_SUB = "matrices"
LOG_LINES = []
plt = None


def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)


def fs(d=0):
    return tf.FONT_SIZE + d


# ══════════════════════════════════════════════════════════════════════
#  Flame name → condition
# ══════════════════════════════════════════════════════════════════════
def parse_condition(name):
    """→ (fuel, d_j, Re as in the name) or None; or (None, reason)."""
    if name in CONDITION_OVERRIDES:
        v = CONDITION_OVERRIDES[name]
        return (None, "left out (CONDITION_OVERRIDES)") if v is None else tuple(v)
    low = name.lower()
    fuel = next((f for f, pat in FUEL_PATTERNS.items() if re.search(pat, low)), None)
    if fuel is None:
        m = re.search(r"phi[_\-]?(\d+(?:[.p]\d+)?)", low)
        if m:
            phi = float(m.group(1).replace("p", "."))
            fuel = next((f for p, f in PHI_FUEL.items() if abs(p - phi) < 1e-6), None)
    m = re.search(r"(?:^|[\s_\-])d(\d+)(?=$|[\s_\-])", low)       # as ohplif1.py
    d = int(m.group(1)) if m else None
    m = re.search(r"(?:^|[\s_\-])re[_\-]?(\d+(?:\.\d+)?)(k?)(?=$|[\s_\-])", low)
    rey = float(m.group(1)) * (1000 if m.group(2) else 1) if m else None
    missing = [k for k, v in (("fuel", fuel), ("D<n>", d), ("Re<n>", rey)) if v is None]
    if missing:
        return None, f"no {', '.join(missing)} in the name"
    return fuel, d, rey


def to_cell(cond):
    """(fuel, d, Re) → (fuel, d, column Re) or (None, reason)."""
    fuel, d, rey = cond
    if fuel not in RE_COLUMNS:
        return None, f"fuel {fuel} not in FUELS"
    if d not in D_ROWS:
        return None, f"d_j = {d} mm not in D_ROWS"
    col = min(RE_COLUMNS[fuel], key=lambda c: abs(c - rey))
    if abs(col - rey) > RE_TOLERANCE * col:
        return None, f"Re = {rey:g} matches no column of {fuel}"
    return fuel, d, col


def re_label(v):
    return f"{v / 1000:g}"


# ══════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════════════
def window(img, G, z_bottom, hw, k, kind="float", rows=None):
    """Rows `rows` (a, b) of img (bottom row at z_bottom), columns within ± hw mm of the
    nozzle axis, reduced k × k.  → (image, extent in mm) or (None, None)."""
    H, W = img.shape
    a, b = rows if rows is not None else (0, H)
    c0 = max(0, int(math.floor(G.nc - hw / G.res)))
    c1 = min(W, int(math.ceil(G.nc + hw / G.res)) + 1)
    sub = np.asarray(img[a:b, c0:c1])
    h, w = sub.shape
    hk, wk = h // k * k, w // k * k
    if hk == 0 or wk == 0:
        return None, None
    sub = sub[h - hk:, :wk]               # keep the bottom rows: heights stay exact
    a2 = a + (h - hk)
    if k > 1:
        blk = sub.reshape(hk // k, k, wk // k, k).astype(np.float32).mean(axis=(1, 3))
        sub = blk >= 0.5 if kind == "binary" else blk
    ext = ((c0 - 0.5 - G.nc) * G.res, (c0 + wk - 0.5 - G.nc) * G.res,
           (H - b) * G.res + z_bottom - 0.5 * G.res,
           (H - 1 - a2) * G.res + z_bottom + 0.5 * G.res)
    return sub, ext


def reduction(res, width_mm, width_in):
    """Block size so that the image has ≤ MAX_PPI pixels per inch."""
    if width_in <= 0:
        return 1
    ppi = (width_mm / res) / width_in
    return max(1, int(ppi // MAX_PPI))


def lines_mm(lines, G, H, z_bottom):
    return [np.column_stack(G.to_mm(ln, H, z_bottom)) for ln in lines]


def max_abs_r(lines):
    return max((float(np.abs(l[:, 0]).max()) for l in lines if len(l)), default=0.0)


def max_z(lines):
    return max((float(l[:, 1].max()) for l in lines if len(l)), default=-np.inf)


def clip_z(lines, zlo, zhi, pad):
    """Pieces of the lines inside zlo ≤ z ≤ zhi (+ pad), for smaller PDFs."""
    out = []
    for l in lines:
        inside = (l[:, 1] >= zlo - pad) & (l[:, 1] <= zhi + pad)
        if not inside.any():
            continue
        idx = np.flatnonzero(inside)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            a, b = max(run[0] - 1, 0), min(run[-1] + 2, len(l))
            if b - a >= 2:
                out.append(l[a:b])
    return out


def tick_plan(hw, in_per_mm, gap_in=0.0):
    """r ticks for cells of ± hw mm drawn at in_per_mm: (s, labels) for ticks -s, 0, s.
    Prefers three labels, then the two outer ones, then only 0; the outer labels must also
    stay clear of the neighbouring cell's labels (gap_in = space between the cells)."""
    def wid(v):                                # label width [in]: digit 0.5 em, minus 0.78 em
        txt = f"{v:g}"
        return (0.5 * len(txt.lstrip("-")) + 0.78 * txt.startswith("-")) * fs(-2) / 72
    p = 10 ** math.floor(math.log10(0.75 * hw))
    cands = [m * q for q in (p, p / 10) for m in (5, 2.5, 2, 1) if m * q <= 0.75 * hw]
    outer = None
    for st in cands:
        sp = st * in_per_mm
        nb = 2 * (hw - st) * in_per_mm + gap_in
        if nb < (wid(-st) + wid(st)) / 2 + 0.05:
            continue
        if sp >= (wid(-st) + wid(0)) / 2 + 0.05:
            return st, [f"{-st:g}", "0", f"{st:g}"]
        if outer is None and 2 * sp >= wid(-st) + 0.06:
            outer = (st, [f"{-st:g}", "", f"{st:g}"])
    return outer or (cands[0], ["", "0", ""])


def r_ticks(ax, plan, labels=True):
    st, lab = plan
    ax.set_xticks([-st, 0.0, st])
    ax.set_xticklabels(lab if labels else [])


def z_tick_step(span_mm):
    """Nice tick distance giving about 2–3 ticks over span_mm."""
    target = span_mm / 2.2
    p = 10 ** math.floor(math.log10(target))
    return next(m * p for m in (10, 5, 2.5, 2, 1) if m * p <= target)


def z_ticks(zlo, zhi, step, edge_mm):
    """Multiples of step inside (zlo, zhi), not closer than edge_mm to either end."""
    k0 = math.ceil((zlo + edge_mm) / step)
    k1 = math.floor((zhi - edge_mm) / step)
    return [k * step for k in range(k0, k1 + 1)]


def round_up(x, step=0.5):
    return math.ceil(x / step) * step


# ══════════════════════════════════════════════════════════════════════
#  Pass 1: what every flame has (fronts, c-lines, extents) — small things only
# ══════════════════════════════════════════════════════════════════════
def blob_columns(BI):
    """Columns that hold burnt gas, ignoring isolated specks."""
    from scipy.ndimage import label
    lab, n = label(BI, structure=np.ones((3, 3)))
    if n == 0:
        return np.empty(0, int)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    keep = sizes >= max(CONTENT_MIN_BLOB * sizes.max(), 1)
    return np.flatnonzero(keep[lab].any(axis=0))


def ambient_mask(BI):
    """Ambient air: unburnt gas connected to the left or right image edge (after the same
    gap sealing as the front re-scan).  The unburnt core and enclosed pockets are not."""
    import cv2
    from scipy.ndimage import label
    B = BI.astype(np.uint8)
    gc = int(tf.RESCAN_GAP_CLOSE_PX or 0)
    if gc > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gc, gc))
        B = cv2.morphologyEx(B, cv2.MORPH_CLOSE, k, borderType=cv2.BORDER_REPLICATE)
    lab, _ = label(B == 0, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    ids = np.unique(np.concatenate([lab[:, 0], lab[:, -1]]))
    return np.isin(lab, ids[ids > 0])


def ambient_sides(amb, G):
    """Per row (top → bottom): how far the burnt gas reaches from the nozzle axis to the
    left and to the right before the ambient air starts [mm] (inf: no air in that row)."""
    W = amb.shape[1]
    cols = np.arange(W)[None, :]
    left = np.where(amb & (cols < G.nc), cols, -1).max(axis=1)
    right = np.where(amb & (cols > G.nc), cols, W).min(axis=1)
    dl = np.where(left >= 0, (G.nc - left - 1) * G.res, np.inf)    # last burnt pixel
    dr = np.where(right < W, (right - 1 - G.nc) * G.res, np.inf)
    return dl.astype(np.float32), dr.astype(np.float32)


def parse_cell(text):
    """ "H2, 9, 30k" → ("H2", 9, 30000) (fuel, d_j, column Re)."""
    parts = [t for t in re.split(r"[\s,;/_]+", str(text).strip()) if t]
    if len(parts) != 3:
        raise ValueError(f"cell {text!r}: give fuel, d_j, Re, e.g. 'H2, 9, 30k'")
    fuel = next((f for f in FUELS if f.lower() == parts[0].lower()), None)
    d = float(re.sub(r"(?i)^d|mm$", "", parts[1]))
    m = re.fullmatch(r"(?i)(?:re)?(\d+(?:\.\d+)?)(k?)", parts[2])
    if fuel is None or not m:
        raise ValueError(f"cell {text!r} not understood (fuels: {', '.join(FUELS)})")
    rey = float(m.group(1)) * (1000 if m.group(2) else 1)
    cell = to_cell((fuel, int(d) if d.is_integer() else d, rey))
    if cell[0] is None:
        raise ValueError(f"cell {text!r}: {cell[1]}")
    return cell[0], cell[2], cell[1]          # (fuel, column Re, d_j) as the cell keys


def parse_picture(v):
    """3, "3", "instant_3_front.png", ".../instant_3_raw.pdf" → 3."""
    if isinstance(v, int):
        return v
    v = str(v).strip()
    if v.isdigit():
        return int(v)
    m = re.search(r"instant_(\d+)", os.path.basename(v))
    if not m:
        raise ValueError(f"picture {v!r}: needs 'instant_<n>' in its name, or the number n")
    return int(m.group(1))


def parse_swaps(swap):
    out = {}
    for k, v in (swap or {}).items():
        out[parse_cell(k)] = parse_picture(v)
    return out


def composite_order(flame, ids, mode, forced=None):
    if forced is not None:
        return [int(forced)], "swapped"
    if mode == "auto":
        return ids, "auto"
    return [int(mode)], "fixed"


def scan_instant(F, ci):
    """Front re-scan of composite ci → dict with lines in mm (no images kept)."""
    info, BI, _, _, notes = tf.load_instant(F["fdir"], F["flame"], ci, False, load_raw=False)
    if info is None or BI is None:
        return {"ci": ci, "ok": False, "missing": True, "notes": notes}
    G = F["G"]
    H, W = BI.shape
    R = tf.rescan_front(BI, G.nc)
    cols = blob_columns(BI)
    rec = {"ci": ci, "ok": R["status"] == "ok", "missing": False, "status": R["status"],
           "notes": notes + R["notes"], "info": info, "H": H, "W": W,
           "open_tip": any("tip not closed" in n for n in R["notes"]),
           "front": lines_mm(R["front"], G, H, G.z0),
           "pockets": lines_mm(R["pockets"], G, H, G.z0),
           "content_r": ((cols.min() - G.nc) * G.res, (cols.max() - G.nc) * G.res)
           if cols.size else None,
           "z": (G.z0 - 0.5 * G.res, G.z0 + (H - 0.5) * G.res),
           "avail": ((G.nc + 0.5) * G.res, (W - 0.5 - G.nc) * G.res)}
    rec["amb_l"], rec["amb_r"] = (ambient_sides(ambient_mask(BI), G) if R["status"] == "ok"
                                  else (None, None))
    # position strips: rows [a, b) of each position (rows_top_to_bottom)
    rows = info.get("rows_top_to_bottom") or [{"row_offset": 0, "pos": 1}]   # old runs: one strip
    offs = sorted(((int(r["row_offset"]), int(r["pos"])) for r in rows))
    strips = []
    for i, (a, p) in enumerate(offs):
        b = offs[i + 1][0] if i + 1 < len(offs) else H
        strips.append({"pos": p, "rows": (a, b),
                       "z": ((H - b) * G.res + G.z0 - 0.5 * G.res,
                             (H - a) * G.res + G.z0 - 0.5 * G.res)})
    rec["strips"] = sorted(strips, key=lambda s: s["pos"])
    return rec


def scan_flame(fdir, mode, forced=None):
    flame, thr = tf.flame_and_threshold(fdir)
    F = {"flame": flame, "fdir": fdir, "thr": thr, "inst": {}, "chosen": None,
         "cbar": None, "notes": []}
    F["G"] = G = tf.load_geometry(fdir)
    # ── average
    ad = os.path.join(fdir, tf.AVG_DIR)
    cp = os.path.join(ad, "cbar_composite.npy")
    if os.path.isfile(cp) and os.path.isfile(os.path.join(ad, "c_lines.npz")):
        cb = np.load(cp, mmap_mode="r")
        h_real = G.h_real or cb.shape[0]
        colmax = np.asarray(cb[:h_real]).max(axis=0)
        cols = np.flatnonzero(colmax >= CONTENT_CBAR)
        C = tf.load_clines(fdir)
        pos0 = bool(tf.SHOW_POS0 and G.ext_px > 0 and cb.shape[0] > h_real)
        Hd, zb = (cb.shape[0], G.height_offset) if pos0 else (h_real, G.z0)
        cl = {lv: lines_mm(tf.measured_cline(v["left"], v["right"]), G, Hd, zb)
              for lv, v in C.items()}
        cl0 = {}                         # estimated pos0 part of the c-lines (drawn grey)
        if pos0:
            for lv, v in C.items():
                segs = []
                for arm in ("left_full", "right_full"):
                    pts = v[arm]
                    seg = pts[pts[:, 1] >= h_real - 1] if len(pts) else pts
                    if len(seg) >= 2:
                        segs.append(seg[np.argsort(-seg[:, 1], kind="stable")])
                cl0[lv] = lines_mm(segs, G, Hd, zb)
        F["cbar"] = {"h_real": h_real, "W": cb.shape[1], "clines": cl, "clines0": cl0,
                     "pos0": pos0, "H_img": Hd, "zb": zb,
                     "content_r": ((cols.min() - G.nc) * G.res, (cols.max() - G.nc) * G.res)
                     if cols.size else None,
                     "z": (zb - 0.5 * G.res, zb + (Hd - 0.5) * G.res)}
    else:
        F["notes"].append("no c̄ composite / c_lines.npz")
    # ── instantaneous
    ids = tf.composite_ids(fdir)
    if not ids:
        F["notes"].append("no instantaneous composites")
        return F
    if mode == "all":
        for ci in ids:
            F["inst"][ci] = scan_instant(F, ci)
        F["chosen"] = next((ci for ci in ids if F["inst"][ci]["ok"]), ids[0])
        F["how"] = "first closed"
        return F
    order, how = composite_order(flame, ids, mode, forced)
    for ci in order:
        if ci not in ids:
            F["notes"].append(f"composite #{ci} not found (have {ids})")
            continue
        F["inst"][ci] = scan_instant(F, ci)
        if F["inst"][ci]["ok"] or how != "auto":
            F["chosen"] = ci
            break
    if F["chosen"] is None and F["inst"]:
        F["chosen"] = min(F["inst"])            # none closed → the first one
        how += ", no front closed"
    F["how"] = how
    return F


# ══════════════════════════════════════════════════════════════════════
#  Half-widths and z range
# ══════════════════════════════════════════════════════════════════════
def half_width(flames, mode, margin, sets):
    """Common ± half-width [mm] of all cells.  sets: which composite(s) per flame."""
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return float(mode), f"fixed {mode:g} mm"
    ext = []
    for F in flames:
        for ci in sets(F):
            I = F["inst"].get(ci)
            if I is None or I["missing"]:
                continue
            if mode == "content" and I["content_r"]:
                ext.append(max(-I["content_r"][0], I["content_r"][1]))
            elif mode == "front" and I["ok"]:
                ext.append(max_abs_r(I["front"]))      # pockets may lie far out: not counted
        if F["cbar"]:
            if mode == "content" and F["cbar"]["content_r"]:
                ext.append(max(-F["cbar"]["content_r"][0], F["cbar"]["content_r"][1]))
            elif mode == "front":
                ext.append(max(max_abs_r(l) for l in F["cbar"]["clines"].values())
                           if F["cbar"]["clines"] else 0.0)
    if not ext:
        return 20.0, "nothing found → 20 mm"
    hw = round_up(max(ext) + margin, 0.5)
    avail = max(max(F["G"].nc, (F["cbar"]["W"] if F["cbar"] else 0) - 1 - F["G"].nc)
                * F["G"].res for F in flames)
    for F in flames:
        for I in F["inst"].values():
            if not I["missing"]:
                avail = max(avail, max(F["G"].nc, I["W"] - 1 - F["G"].nc) * F["G"].res)
    hw = min(hw, round_up(avail, 0.5))
    return hw, f"{mode}: widest flame {max(ext):.1f} mm + {margin:g} mm"


def z_range(flames, sets, cbar=False):
    """z range [mm] of the instantaneous cells (cbar=False) or of the c̄ cells."""
    lo, hi = [], []
    for F in flames:
        if cbar:
            if F["cbar"]:
                lo.append(F["cbar"]["z"][0]); hi.append(F["cbar"]["z"][1])
            continue
        for ci in sets(F):
            I = F["inst"].get(ci)
            if I and not I["missing"]:
                lo.append(I["z"][0]); hi.append(I["z"][1])
    zlo = min(lo) if lo else 0.0
    zhi = Z_MAX_MM if Z_MAX_MM is not None else (max(hi) if hi else 1.0)
    return zlo, zhi


# ══════════════════════════════════════════════════════════════════════
#  Matrix layout
# ══════════════════════════════════════════════════════════════════════
class MatrixLayout:
    """Inches; cells in a (row, fuel, column) grid of one mm scale."""
    LEFT, RIGHT, TOP = 0.62, 0.46, 0.44

    def __init__(self, hw, zlo, zhi, bottom_in):
        self.hw, self.zlo, self.zhi = hw, zlo, zhi
        self.cols = [(f, c) for f in FUELS for c in RE_COLUMNS[f]]
        ncol, nrow = len(self.cols), len(D_ROWS)
        ngap_g = len(FUELS) - 1
        gaps = (ncol - 1 - ngap_g) * CELL_GAP_IN + ngap_g * GROUP_GAP_IN
        right = self.RIGHT if MATRIX_TICKS else 0.08
        self.BOTTOM = bottom_in
        aspect = (zhi - zlo) / (2 * hw)
        cw = (MATRIX_WIDTH_IN - self.LEFT - right - gaps) / ncol
        ch = cw * aspect
        fixed_h = self.TOP + self.BOTTOM + (nrow - 1) * CELL_GAP_IN
        if fixed_h + nrow * ch > MATRIX_MAX_HEIGHT_IN:
            ch = (MATRIX_MAX_HEIGHT_IN - fixed_h) / nrow
            cw = ch / aspect
        self.cw, self.ch = cw, ch
        self.W = self.LEFT + right + gaps + ncol * cw
        self.H = fixed_h + nrow * ch
        self.x = {}
        x = self.LEFT
        for gi, f in enumerate(FUELS):
            for c in RE_COLUMNS[f]:
                self.x[(f, c)] = x
                x += cw + CELL_GAP_IN
            x += GROUP_GAP_IN - CELL_GAP_IN
        self.y = {d: self.BOTTOM + (nrow - 1 - i) * (ch + CELL_GAP_IN)
                  for i, d in enumerate(D_ROWS)}

    def rect(self, x, y, w, h):
        return [x / self.W, y / self.H, w / self.W, h / self.H]

    def cell_rect(self, fuel, col, d):
        return self.rect(self.x[(fuel, col)], self.y[d], self.cw, self.ch)


def draw_matrix_frame(L, cells, kind):
    """Figure with every cell axes, the headers and the axis labels.
    cells: {(fuel, col, d): F or None}.  Returns fig, {key: ax}."""
    fig = plt.figure(figsize=(L.W, L.H), dpi=tf.DPI)
    axes = {}
    last_col = {f: RE_COLUMNS[f][-1] for f in FUELS}
    for d in D_ROWS:
        for f in FUELS:
            for c in RE_COLUMNS[f]:
                ax = fig.add_axes(L.cell_rect(f, c, d))
                ax.set_xlim(-L.hw, L.hw)
                ax.set_ylim(L.zlo, L.zhi)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4); s.set_color("0.55")
                if kind == "raw":
                    from matplotlib import colormaps
                    ax.set_facecolor(colormaps[tf.RAW_CMAP](0.0))
                elif kind == "binary":
                    ax.set_facecolor(tf.POS0_GREY)     # above the composite: not measured
                if MATRIX_TICKS and d == D_ROWS[-1]:
                    r_ticks(ax, tick_plan(L.hw, L.cw / (2 * L.hw), CELL_GAP_IN))
                    ax.tick_params(axis="x", length=2, width=0.4, pad=1.5, labelsize=fs(-2))
                if MATRIX_TICKS and f == FUELS[-1] and c == last_col[f]:
                    from matplotlib.ticker import MaxNLocator
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
                    ax.yaxis.tick_right()
                    ax.tick_params(axis="y", length=2, width=0.4, pad=1.5, labelsize=fs(-2))
                    ax.set_ylim(L.zlo, L.zhi)
                axes[(f, c, d)] = ax
                if cells.get((f, c, d)) is None:
                    ax.text(0.5, 0.5, EMPTY_TEXT, transform=ax.transAxes, ha="center",
                            va="center", rotation=90, fontsize=fs(-2), color="0.5",
                            style="italic")
    # headers
    top = L.H - 0.06
    for f in FUELS:
        xs = [L.x[(f, c)] for c in RE_COLUMNS[f]]
        fig.text((xs[0] + xs[-1] + L.cw) / 2 / L.W, top / L.H, FUEL_LABELS.get(f, f),
                 ha="center", va="top", fontsize=fs(0))
        for c in RE_COLUMNS[f]:
            fig.text((L.x[(f, c)] + L.cw / 2) / L.W, (top - 0.2) / L.H, re_label(c),
                     ha="center", va="top", fontsize=fs(-1))
    fig.text((L.LEFT - 0.06) / L.W, (top - 0.2) / L.H, r"$Re_\mathrm{j}\cdot10^{-3}$",
             ha="right", va="top", fontsize=fs(-1))
    for d in D_ROWS:
        fig.text((L.LEFT - 0.08) / L.W, (L.y[d] + L.ch / 2) / L.H, f"{d:g}",
                 ha="right", va="center", fontsize=fs(-1))
    mid = (L.y[D_ROWS[-1]] + L.y[D_ROWS[0]] + L.ch) / 2
    fig.text(0.1 / L.W, mid / L.H, r"$d_\mathrm{j}$ [mm]", ha="left", va="center",
             rotation=90, fontsize=fs(-1))
    if MATRIX_TICKS:
        fig.text((L.W - 0.06) / L.W, mid / L.H, r"$z$ [mm]", ha="right", va="center",
                 rotation=90, fontsize=fs(-1))
        x0 = L.x[(FUELS[0], RE_COLUMNS[FUELS[0]][0])]
        x1 = L.x[(FUELS[-1], RE_COLUMNS[FUELS[-1]][-1])] + L.cw
        fig.text((x0 + x1) / 2 / L.W, (L.BOTTOM - 0.2) / L.H, r"$r$ [mm]", ha="center",
                 va="top", fontsize=fs(-1))
    return fig, axes


def key_area(L):
    """Space under the matrix for the colour bar / legend: x0, x1, top [in]."""
    x0 = L.x[(FUELS[0], RE_COLUMNS[FUELS[0]][0])]
    x1 = L.x[(FUELS[-1], RE_COLUMNS[FUELS[-1]][-1])] + L.cw
    return x0, x1, L.BOTTOM - (0.42 if MATRIX_TICKS else 0.1)


def add_colorbar(fig, L, cmap, label, ticks, where="centre"):
    """where: "centre" or "left" (left half; the legend goes right of it)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    x0, x1, top = key_area(L)
    w = min(2.4, 0.45 * (x1 - x0))
    x = (x0 + x1) / 2 - w / 2 if where == "centre" else x0 + 0.04 * (x1 - x0)
    cax = fig.add_axes([x / L.W, (top - 0.07) / L.H, w / L.W, 0.07 / L.H])
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), cax=cax, orientation="horizontal",
                      ticks=ticks)
    cb.set_label(label, fontsize=fs(-1), labelpad=1.5)
    cb.ax.tick_params(labelsize=fs(-2), length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)


def add_legend(fig, L, handles, ncol=None, where="centre"):
    x0, x1, top = key_area(L)
    if where == "centre":
        loc, x = "upper center", (x0 + x1) / 2
    else:                                   # right half, next to a colour bar
        loc, x = "upper left", x0 + 0.55 * (x1 - x0)
    fig.legend(handles=handles, loc=loc, ncol=ncol or len(handles),
               bbox_to_anchor=(x / L.W, (top + 0.05) / L.H), fontsize=fs(-2),
               frameon=False, handlelength=2.4, columnspacing=1.0, borderaxespad=0,
               labelspacing=0.3)


def save(fig, base, formats, images):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for f in formats:
        for im in images:   # PDF: image embedded as it is; PNG: smooth down-sampling
            im.set_interpolation("none" if f == "pdf" else "antialiased")
        tmp = f"{base}.{f}.part"
        fig.savefig(tmp, format=f, dpi=tf.DPI)
        os.replace(tmp, f"{base}.{f}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  Pass 2: images (cropped + reduced) for every cell
# ══════════════════════════════════════════════════════════════════════
def load_images(F, ci, hw_m, k_m, stagger_hw=None, k_s=1, panels_z=None, fill=False):
    """Small images of one flame / composite.  panels_z: {pos: (zlo, zhi)} of the
    position-window panels (cropped to ± stagger_hw); fill: also the ambient air mask
    and the fill value of every panel."""
    G = F["G"]
    out = {}
    info, BI, raw, src, notes = tf.load_instant(F["fdir"], F["flame"], ci, False,
                                                load_raw=True)
    for n in notes:
        log(f"    {F['flame']} #{ci}: {n}")
    if BI is not None:
        out["binary"] = window(BI, G, G.z0, hw_m, k_m, "binary")
    if raw is None:
        return out
    rawn = tf.normalise_raw(raw, info)
    out["raw"] = window(rawn, G, G.z0, hw_m, k_m)
    if stagger_hw is None or BI is None:
        return out
    I = F["inst"][ci]
    H = rawn.shape[0]
    amb = ambient_mask(BI) if fill else None
    panels = []
    for s in I["strips"]:
        if s["pos"] not in panels_z:
            continue
        a, b = s["rows"]
        top = panels_z[s["pos"]][1]
        if top < s["z"][1]:                                 # keep rows with centre ≤ top
            a = max(a, int(math.ceil(H - 1 - (top - G.z0) / G.res)))
        img, ext = window(rawn, G, G.z0, stagger_hw, k_s, rows=(a, b))
        if img is None:
            continue
        pn = {"pos": s["pos"], "img": img, "ext": ext, "amb": None, "fill": None}
        if fill:
            pn["amb"] = window(amb, G, G.z0, stagger_hw, k_s, "binary", rows=(a, b))[0]
            if FILL_VALUE == "border":
                from scipy.ndimage import distance_transform_edt
                am = amb[a:b]
                near = (distance_transform_edt(~am) * G.res <= FILL_BORDER_MM) & ~am & BI[a:b]
                v = rawn[a:b][near]
                pn["fill"] = float(np.median(v)) if v.size else 0.0
            else:
                pn["fill"] = float(FILL_VALUE)
        panels.append(pn)
    out["stagger"] = panels
    return out


def load_cbar(F, hw_m, k_m):
    """c̄ image of the matrix cell; with pos0 the pos0 rows are NaN (drawn grey)."""
    G = F["G"]
    c = F["cbar"]
    cb = np.load(os.path.join(F["fdir"], tf.AVG_DIR, "cbar_composite.npy"), mmap_mode="r")
    if c["pos0"]:
        img = np.array(cb, dtype=np.float32)
        img[c["h_real"]:] = np.nan
    else:
        img = cb[:c["h_real"]]
    im, ext = window(img, G, c["zb"], hw_m, k_m)
    return (None, None) if im is None else (np.ma.masked_invalid(im), ext)


# ══════════════════════════════════════════════════════════════════════
#  Matrices
# ══════════════════════════════════════════════════════════════════════
def show_cell(ax, img_ext, **kw):
    img, ext = img_ext
    return ax.imshow(img, extent=ext, origin="upper", aspect="auto", **kw)


def draw_instant_matrices(L, cells, imgs, sets_ci, tag, out_dir, formats, stats):
    """raw, binary, front, front_red matrices of one composite choice."""
    from matplotlib import colormaps
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    red_front = STAGGER_FRONT_COLOR or tf.RED_FRONT_COLOR
    red_pocket = STAGGER_POCKET_COLOR or tf.RED_POCKET_COLOR
    for kind in ("raw", "binary", "front", "front_red"):
        fig, axes = draw_matrix_frame(L, cells, kind)
        images = []
        for key, F in cells.items():
            if F is None:
                continue
            ax = axes[key]
            ci = sets_ci(F)
            I = F["inst"].get(ci)
            im = imgs.get((F["flame"], ci), {})
            if I is None or I["missing"] or "binary" not in im or im["binary"][0] is None:
                ax.text(0.5, 0.5, EMPTY_TEXT, transform=ax.transAxes, ha="center",
                        va="center", rotation=90, fontsize=fs(-2), color="0.5", style="italic")
                continue
            if kind == "raw":
                if im.get("raw") is None or im["raw"][0] is None:
                    ax.text(0.5, 0.5, "raw not available", transform=ax.transAxes,
                            ha="center", va="center", rotation=90, fontsize=fs(-2), color="0.7")
                    continue
                images.append(show_cell(ax, im["raw"], cmap=tf.RAW_CMAP, vmin=0, vmax=1))
            elif kind == "binary":
                images.append(show_cell(ax, im["binary"], cmap=tf.binary_cmap(), vmin=0, vmax=1))
            elif kind == "front_red":
                if im.get("raw") is None or im["raw"][0] is None:
                    ax.text(0.5, 0.5, "raw not available", transform=ax.transAxes,
                            ha="center", va="center", rotation=90, fontsize=fs(-2), color="0.7")
                    continue
                images.append(show_cell(ax, im["raw"], cmap=tf.oh_red_cmap(), vmin=0, vmax=1))
                if I["ok"]:
                    st = dict(linewidth=tf.FRONT_LW, solid_joinstyle="round",
                              solid_capstyle="round")
                    for ln in clip_z(I["front"], L.zlo, L.zhi, 1):
                        ax.plot(ln[:, 0], ln[:, 1], color=red_front, **st)
                    for ln in clip_z(I["pockets"], L.zlo, L.zhi, 1):
                        ax.plot(ln[:, 0], ln[:, 1], color=red_pocket, **st)
                else:
                    ax.text(0.5, 0.98, "front not\nclosed", transform=ax.transAxes,
                            ha="center", va="top", fontsize=fs(-3), color=red_front)
            else:
                bg = tf.FRONT_BACKGROUND
                if bg == "raw" and im.get("raw") is not None and im["raw"][0] is not None:
                    images.append(show_cell(ax, im["raw"], cmap=tf.RAW_CMAP, vmin=0, vmax=1))
                elif bg != "white":
                    images.append(show_cell(ax, im["binary"],
                                            cmap=tf.two_colour_map("white", tf.BURNT_GREY),
                                            vmin=0, vmax=1))
                if I["ok"]:
                    st = dict(linewidth=tf.FRONT_LW, solid_joinstyle="round",
                              solid_capstyle="round")
                    for ln in clip_z(I["front"], L.zlo, L.zhi, 1):
                        ax.plot(ln[:, 0], ln[:, 1], color=tf.FRONT_COLOR, **st)
                    for ln in clip_z(I["pockets"], L.zlo, L.zhi, 1):
                        ax.plot(ln[:, 0], ln[:, 1], color=tf.POCKET_COLOR, **st)
                else:
                    ax.text(0.5, 0.98, "front not\nclosed", transform=ax.transAxes,
                            ha="center", va="top", fontsize=fs(-3), color=tf.FRONT_COLOR)
        if kind == "raw":
            add_colorbar(fig, L, colormaps[tf.RAW_CMAP], "OH-PLIF signal (normalised)",
                         [0, 0.5, 1])
        elif kind == "binary":
            add_legend(fig, L, [Patch(facecolor=tf.BINARY_BURNT, edgecolor="0.5", lw=0.4,
                                      label="burnt (1)"),
                                Patch(facecolor=tf.BINARY_UNBURNT, edgecolor="0.5", lw=0.4,
                                      label="unburnt (0)"),
                                Patch(facecolor=tf.POS0_GREY, edgecolor="0.5", lw=0.4,
                                      label="not measured")])
        elif kind == "front_red":
            add_colorbar(fig, L, tf.oh_red_cmap(), "OH-PLIF signal (normalised)", [0, 0.5, 1],
                         where="left")
            add_legend(fig, L, [Line2D([], [], color=red_front, lw=1.2, label="flame front")]
                       + ([Line2D([], [], color=red_pocket, lw=1.2, label="pockets")]
                          if red_pocket != red_front else []), ncol=1, where="right")
        else:
            h = [Line2D([], [], color=tf.FRONT_COLOR, lw=1.2, label="flame front"),
                 Line2D([], [], color=tf.POCKET_COLOR, lw=1.2, label="pockets")]
            if tf.FRONT_BACKGROUND == "binary":
                h.append(Patch(facecolor=tf.BURNT_GREY, edgecolor="none", label="burnt (OH)"))
            add_legend(fig, L, h)
        name = f"matrix_instant{tag}_{kind}"
        save(fig, os.path.join(out_dir, name), formats, images)
        stats["matrices"] += 1
        log(f"    {name}")


def draw_average_matrices(L, cells, cimgs, out_dir, formats, stats):
    from matplotlib import colormaps
    from matplotlib.lines import Line2D
    levels = sorted({lv for F in cells.values() if F and F["cbar"]
                     for lv in F["cbar"]["clines"]})
    variants = [("average_cbar", [])] + [(f"average_cbar_{lv:g}", [lv]) for lv in levels]
    if len(levels) > 1:
        variants.append(("average_all_clines", levels))
    from matplotlib.patches import Patch
    cmap = colormaps[tf.CBAR_CMAP].copy()
    cmap.set_bad(tf.POS0_GREY)
    any_pos0 = any(F and F["cbar"] and F["cbar"]["pos0"] for F in cells.values())
    for name, lvs in variants:
        fig, axes = draw_matrix_frame(L, cells, "cbar")
        images = []
        for key, F in cells.items():
            if F is None:
                continue
            ax = axes[key]
            ci = cimgs.get(F["flame"])
            if ci is None or ci[0] is None:
                ax.text(0.5, 0.5, EMPTY_TEXT, transform=ax.transAxes, ha="center",
                        va="center", rotation=90, fontsize=fs(-2), color="0.5", style="italic")
                continue
            images.append(show_cell(ax, ci, cmap=cmap, vmin=0, vmax=1))
            for lv in lvs:
                ls = tf.cline_style(lv, levels)
                for ln in F["cbar"]["clines"].get(lv, []):
                    ax.plot(ln[:, 0], ln[:, 1], color=tf.CLINE_COLOR, lw=tf.CLINE_LW, ls=ls,
                            solid_joinstyle="round", dash_capstyle="butt")
                for ln in F["cbar"]["clines0"].get(lv, []):          # estimated pos0 part
                    ax.plot(ln[:, 0], ln[:, 1], color=tf.POS0_LINE, lw=tf.CLINE_LW, ls=ls,
                            solid_joinstyle="round", dash_capstyle="butt")
        add_colorbar(fig, L, cmap, r"$\bar{c}$", [0, 0.25, 0.5, 0.75, 1],
                     where="left" if (lvs or any_pos0) else "centre")
        hs = [Line2D([], [], color=tf.CLINE_COLOR, lw=tf.CLINE_LW,
                     ls=tf.cline_style(lv, levels), label=rf"$\bar{{c}} = {lv:g}$")
              for lv in lvs]
        if any_pos0:
            hs.append(Patch(facecolor=tf.POS0_GREY, edgecolor="none", label="not measured"))
            if lvs:
                hs.append(Line2D([], [], color=tf.POS0_LINE, lw=tf.CLINE_LW,
                                 ls=tf.cline_style(lvs[0], levels), label="estimated"))
        if hs:
            add_legend(fig, L, hs, ncol=1 if len(hs) <= 3 else 2, where="right")
        save(fig, os.path.join(out_dir, "matrix_" + name), formats, images)
        stats["matrices"] += 1
        log(f"    matrix_{name}")


# ══════════════════════════════════════════════════════════════════════
#  Position-window figures
# ══════════════════════════════════════════════════════════════════════
def stagger_groups(cells):
    """[(file name, corner label, [(group label, [(column label, F)])])]."""
    out = []
    ok = {k: F for k, F in cells.items() if F is not None and F["chosen"] is not None
          and F["inst"][F["chosen"]]["ok"]}
    for fuel in FUELS:
        if "fuel" in STAGGER_GROUPS:
            groups = []
            for d in D_ROWS:
                cols = [(c, ok[(fuel, c, d)]) for c in RE_COLUMNS[fuel] if (fuel, c, d) in ok]
                if cols:
                    groups.append((rf"$d_\mathrm{{j}}$ = {d:g} mm", cols))
            if groups:
                out.append((f"positions_front_{fuel}", FUEL_LABELS.get(fuel, fuel), groups))
        if "fuel_d" in STAGGER_GROUPS:
            for d in D_ROWS:
                cols = [(c, ok[(fuel, c, d)]) for c in RE_COLUMNS[fuel] if (fuel, c, d) in ok]
                if cols:
                    out.append((f"positions_front_{fuel}_d{d:g}",
                                FUEL_LABELS.get(fuel, fuel) + rf", $d_\mathrm{{j}}$ = {d:g} mm",
                                [(None, cols)]))
    return out


def tip_cut(F):
    I = F["inst"][F["chosen"]]
    if not CROP_TO_TIP or I["open_tip"]:
        return None
    zt = max_z(I["front"])          # the core's tip; pockets higher up are cut off
    return None if not np.isfinite(zt) else zt + TIP_MARGIN_MM


def column_panels(F, cut):
    """[(pos, zlo, zhi)] of one column after the tip cut."""
    out = []
    for s in F["inst"][F["chosen"]]["strips"]:
        zlo, zhi = s["z"]
        if cut is None:
            out.append((s["pos"], zlo, zhi))
            continue
        if cut - TIP_MARGIN_MM <= zlo:          # tip below this panel: only margin → out
            continue
        out.append((s["pos"], zlo, min(zhi, max(cut, zlo + MIN_PANEL_MM))))
    return out


def stagger_cmap():
    from matplotlib import colormaps
    return colormaps[STAGGER_CMAP] if STAGGER_CMAP else tf.oh_red_cmap()


def front_extent(F, zlo, zhi):
    """How far the front reaches left and right of the axis between zlo and zhi [mm]."""
    pts = [l[(l[:, 1] >= zlo) & (l[:, 1] <= zhi)] for l in F["inst"][F["chosen"]]["front"]]
    pts = [q for q in pts if len(q)]
    if not pts:
        return 0.0, 0.0
    P = np.vstack(pts)
    return max(0.0, float(-P[:, 0].min())), max(0.0, float(P[:, 0].max()))


def burnt_extent(F, zlo, zhi):
    """How far the burnt gas reaches left and right of the axis in one panel [mm]
    (NATURAL_PERCENTILE over its rows), at least the front + FRONT_CLEARANCE_MM, at most
    the image."""
    I, G = F["inst"][F["chosen"]], F["G"]
    al, ar = I["avail"]
    if I["amb_l"] is None:
        return al, ar
    z = (I["H"] - 1 - np.arange(I["H"])) * G.res + G.z0
    rows = (z >= zlo) & (z <= zhi)
    if not rows.any():
        return al, ar
    L = float(np.percentile(np.minimum(I["amb_l"][rows], al), NATURAL_PERCENTILE))
    R = float(np.percentile(np.minimum(I["amb_r"][rows], ar), NATURAL_PERCENTILE))
    fl, fr = front_extent(F, zlo, zhi)
    L = max(L, fl + FRONT_CLEARANCE_MM)
    R = max(R, fr + FRONT_CLEARANCE_MM)
    return min(L, al), min(R, ar)


def figure_columns(version, groups, hw_fixed=None):
    """Columns of one position-window figure: [{"F", "re", "slot": (L, R),
    "panels": [(pos, zlo, zhi, L, R)]}]; L, R = extent left/right of the axis [mm]."""
    cols = []
    for _, cs in groups:
        for re_col, F in cs:
            panels = column_panels(F, tip_cut(F))
            ext = [burnt_extent(F, zlo, zhi) for _, zlo, zhi in panels]
            cols.append({"F": F, "re": re_col, "panels": panels, "ext": ext})
    if version == "natural":
        for c in cols:
            L1, R1 = c["ext"][0]                        # pos1 (the lowest panel)
            c["slot"] = (L1, R1)
            c["panels"] = [(p, zlo, zhi, min(L, L1), min(R, R1))
                           for (p, zlo, zhi), (L, R) in zip(c["panels"], c["ext"])]
        return cols, None
    # filled: width from the fronts only (pockets outside the panel are cut off)
    for c in cols:
        fe = [front_extent(c["F"], zlo, zhi) for _, zlo, zhi in c["panels"]]
        c["front"] = (max(l for l, _ in fe) + FRONT_CLEARANCE_MM,
                      max(r for _, r in fe) + FRONT_CLEARANCE_MM)
    if hw_fixed:
        hw = float(hw_fixed)
    elif FILLED_WIDTH == "column":
        hw = None
    else:
        hw = math.ceil(max(max(c["front"]) for c in cols) * 10) / 10
    for c in cols:
        L, R = (hw, hw) if hw else (math.ceil(c["front"][0] * 10) / 10,
                                    math.ceil(c["front"][1] * 10) / 10)
        c["slot"] = (L, R)
        c["panels"] = [(p, zlo, zhi, L, R) for p, zlo, zhi in c["panels"]]
    return cols, hw


class StaggerLayout:
    """Inches.  Columns side by side (touching in pos1), one row per position."""
    LEFT, RIGHT, TOP = 0.42, 0.78, 0.24

    def __init__(self, groups, cols):
        self.cols = cols
        has_group_labels = any(g is not None for g, _ in groups)
        self.BOTTOM = 0.34 + (0.2 if has_group_labels else 0.0)
        rows = {}
        for c in cols:
            for s in c["F"]["inst"][c["F"]["chosen"]]["strips"]:
                lo, hi = rows.get(s["pos"], (np.inf, -np.inf))
                rows[s["pos"]] = (min(lo, s["z"][0]), max(hi, s["z"][1]))
        used = {pn[0] for c in cols for pn in c["panels"]}
        self.rows = OrderedDict(sorted((p, z) for p, z in rows.items() if p in used))
        span = sum(hi - lo for lo, hi in self.rows.values())
        widths = [c["slot"][0] + c["slot"][1] for c in cols]
        s = min((STAGGER_WIDTH_IN - self.LEFT - self.RIGHT) / sum(widths),
                STAGGER_MAX_COL_IN / max(widths))
        fixed = self.TOP + self.BOTTOM
        if fixed + span * s > STAGGER_MAX_HEIGHT_IN:
            s = (STAGGER_MAX_HEIGHT_IN - fixed) / span
        self.s = s
        self.W = self.LEFT + self.RIGHT + sum(widths) * s
        self.H = fixed + span * s
        self.row_y, y = {}, self.BOTTOM
        for p, (lo, hi) in self.rows.items():
            self.row_y[p] = y
            y += (hi - lo) * s
        self.axis_x, x = [], self.LEFT                 # x of every column's nozzle axis
        for c, w in zip(cols, widths):
            self.axis_x.append(x + c["slot"][0] * s)
            x += w * s

    def rect(self, x, y, w, h):
        return [x / self.W, y / self.H, w / self.W, h / self.H]


def draw_stagger(name, corner, version, groups, cols, Lay, imgs, out_dir, formats, stats):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, ListedColormap
    fig = plt.figure(figsize=(Lay.W, Lay.H), dpi=tf.DPI)
    cmap = stagger_cmap()
    c_front = STAGGER_FRONT_COLOR or tf.RED_FRONT_COLOR
    c_pocket = STAGGER_POCKET_COLOR or tf.RED_POCKET_COLOR
    images = []
    leftmost = {}                                     # row → first column that has it
    for j, c in enumerate(cols):
        for pn in c["panels"]:
            leftmost.setdefault(pn[0], j)
    group_first, j0 = set(), 0                        # first column of every d_j group
    for _, cs in groups:
        group_first.add(j0)
        j0 += len(cs)
    zstep = z_tick_step(max(hi - lo for lo, hi in Lay.rows.values()))
    edge = 0.07 / Lay.s                               # no z label this close to a panel end
    first_row = next(iter(Lay.rows))
    plan = tick_plan(min(min(pn[3], pn[4]) for c in cols for pn in c["panels"]
                         if pn[0] == first_row), Lay.s)        # same ticks in every column
    painted = 0.0
    for j, c in enumerate(cols):
        F = c["F"]
        I = F["inst"][F["chosen"]]
        Ls, Rs = c["slot"]
        pimg = {pn["pos"]: pn for pn in imgs.get((F["flame"], F["chosen"]), {}).get("stagger", [])}
        for p, zlo, zhi, Lp, Rp in c["panels"]:
            if p not in Lay.rows:
                continue
            y = Lay.row_y[p] + (zlo - Lay.rows[p][0]) * Lay.s
            ax = fig.add_axes(Lay.rect(Lay.axis_x[j] - Lp * Lay.s, y, (Lp + Rp) * Lay.s,
                                       (zhi - zlo) * Lay.s))
            ax.set_facecolor(cmap(0.0))
            pn = pimg.get(p)
            if pn is not None:
                images.append(ax.imshow(pn["img"], extent=pn["ext"], origin="upper",
                                        aspect="auto", cmap=cmap, vmin=0, vmax=1))
                if version == "filled" and pn["amb"] is not None:
                    fc = cmap(pn["fill"])
                    ax.set_facecolor(fc)                  # also beyond the image edge
                    ax.imshow(np.ma.masked_where(~pn["amb"], np.ones(pn["amb"].shape)),
                              extent=pn["ext"], origin="upper", aspect="auto",
                              cmap=ListedColormap([fc]), vmin=0, vmax=1,
                              interpolation="nearest")
                    nx = pn["amb"].shape[1]
                    xc = pn["ext"][0] + (np.arange(nx) + 0.5) * (pn["ext"][1] - pn["ext"][0]) / nx
                    inside = pn["amb"][:, (xc >= -Lp) & (xc <= Rp)]
                    if inside.size:
                        painted = max(painted, float(inside.mean()))
            st = dict(linewidth=STAGGER_LW, solid_joinstyle="round", solid_capstyle="round")
            for ln in clip_z(I["front"], zlo, zhi, 1):
                ax.plot(ln[:, 0], ln[:, 1], color=c_front, **st)
            for ln in clip_z(I["pockets"], zlo, zhi, 1):
                ax.plot(ln[:, 0], ln[:, 1], color=c_pocket, **st)
            ax.set_xlim(-Lp, Rp)
            ax.set_ylim(zlo, zhi)
            for s_ in ax.spines.values():
                s_.set_linewidth(PANEL_LW)
            if j in group_first and j > 0 and Lp >= Ls - 1e-9:
                ax.spines["left"].set_linewidth(GROUP_LINE_LW)
            if j + 1 in group_first and Rp >= Rs - 1e-9:
                ax.spines["right"].set_linewidth(GROUP_LINE_LW)
            r_ticks(ax, plan, labels=p == first_row)
            ax.set_yticks(z_ticks(zlo, zhi, zstep, edge))
            ax.tick_params(direction="in", length=2, width=0.4, labelsize=fs(-2), pad=1.5,
                           top=False, right=False)
            if leftmost.get(p) != j:
                ax.set_yticklabels([])
            ax.set_ylim(zlo, zhi)
        # condition under the column (below the r tick labels)
        xc = Lay.axis_x[j] + (Rs - Ls) / 2 * Lay.s
        fig.text(xc / Lay.W, (Lay.BOTTOM - 0.16) / Lay.H, f"{re_label(c['re'])}k",
                 ha="center", va="top", fontsize=fs(-2))
    # labels
    fig.text((Lay.LEFT - 0.05) / Lay.W, (Lay.BOTTOM - 0.16) / Lay.H, r"$Re_\mathrm{j}$",
             ha="right", va="top", fontsize=fs(-2))
    x_end = Lay.axis_x[-1] + cols[-1]["slot"][1] * Lay.s
    fig.text((x_end + 0.05) / Lay.W, (Lay.BOTTOM - 0.035) / Lay.H, r"$r$ [mm]", ha="left",
             va="top", fontsize=fs(-2))
    if any(g is not None for g, _ in groups):
        j = 0
        for g, cs in groups:
            x0 = Lay.axis_x[j] - cols[j]["slot"][0] * Lay.s
            k = j + len(cs) - 1
            x1 = Lay.axis_x[k] + cols[k]["slot"][1] * Lay.s
            yy = Lay.BOTTOM - 0.33
            fig.add_artist(plt.Line2D([(x0 + 0.03) / Lay.W, (x1 - 0.03) / Lay.W],
                                      [yy / Lay.H] * 2, color="0.3", lw=0.5))
            fig.text((x0 + x1) / 2 / Lay.W, (yy - 0.03) / Lay.H, g, ha="center", va="top",
                     fontsize=fs(-1))
            j += len(cs)
    fig.text(0.06 / Lay.W, (Lay.BOTTOM + (Lay.H - Lay.TOP - Lay.BOTTOM) / 2) / Lay.H,
             r"$z$ [mm]", ha="left", va="center", rotation=90, fontsize=fs(-1))
    fig.text(Lay.LEFT / Lay.W, (Lay.H - 0.04) / Lay.H, corner, ha="left", va="top",
             fontsize=fs(0))
    # colour bar
    ph = Lay.H - Lay.TOP - Lay.BOTTOM
    ch = min(2.2, 0.55 * ph)
    cax = fig.add_axes(Lay.rect(Lay.W - Lay.RIGHT + 0.1, Lay.BOTTOM + (ph - ch) / 2, 0.08, ch))
    cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap), cax=cax, ticks=[0, 0.5, 1])
    cb.set_label("OH-PLIF signal\n(normalised)", fontsize=fs(-1))
    cb.ax.tick_params(labelsize=fs(-2), length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    save(fig, os.path.join(out_dir, name), formats, images)
    stats["stagger"] += 1
    log(f"    {name}  ({len(cols)} flames, {len(Lay.rows)} position rows"
        + ((f", ambient air painted in up to {100 * painted:.1f} % of a panel" if painted
            else ", no ambient air inside the panels: nothing painted")
           if version == "filled" else "") + ")")


# ══════════════════════════════════════════════════════════════════════
def run_threshold(thr, fdirs, out_dir, a, stats):
    log(f"\n{'═' * 70}\nThreshold {tf.thr_pct(thr)} %   →  {out_dir}")
    # ── which flame in which cell
    cells = {(f, c, d): None for f in FUELS for c in RE_COLUMNS[f] for d in D_ROWS}
    table = []
    for fdir in fdirs:
        flame = tf.flame_and_threshold(fdir)[0]
        cond = parse_condition(flame)
        if cond[0] is None:
            table.append((flame, "—", cond[1])); continue
        cell = to_cell(cond)
        if cell[0] is None:
            table.append((flame, "—", cell[1])); continue
        cell = (cell[0], cell[2], cell[1])             # → (fuel, column Re, d_j)
        if cells[cell] is not None:
            table.append((flame, "—", f"cell already taken by {cells[cell]} "
                                      f"(CONDITION_OVERRIDES)")); continue
        cells[cell] = fdir
        table.append((flame, f"{cell[0]}  d={cell[2]}  Re={re_label(cell[1])}k", ""))
    w = max([len(t[0]) for t in table] + [5])
    log("  flame".ljust(w + 2) + "   cell")
    for f, c, why in table:
        log(f"  {f.ljust(w)}   {c}" + (f"   ({why})" if why else ""))
    empty = [k for k, v in cells.items() if v is None]
    if empty:
        log("  empty cells: " + ", ".join(f"{f} d={d} Re={re_label(c)}k" for f, c, d in empty))
    if a.list:
        return
    # ── pass 1
    log("\n  reading fronts and c-lines …")
    flames = {}
    for key, fdir in cells.items():
        if fdir is None:
            continue
        try:
            F = scan_flame(fdir, a.composite, a.swaps.get(key))
        except Exception as e:
            log(f"    {tf.flame_and_threshold(fdir)[0]}: FAILED ({type(e).__name__}: {e})")
            cells[key] = None
            continue
        cells[key] = F
        flames[F["flame"]] = F
        ci = F["chosen"]
        msg = f"    {F['flame']}: "
        if ci is None:
            msg += "no instantaneous composite"
        else:
            I = F["inst"][ci]
            msg += (f"composite #{ci} ({F['how']})"
                    + ("" if I["ok"] else f" — front {I.get('status', 'missing')}")
                    + (", open tip" if I.get("open_tip") else ""))
        for n in F["notes"]:
            msg += f"; {n}"
        log(msg)
    Fs = list(flames.values())
    if not Fs:
        log("  nothing to draw")
        return
    ci_sets = ((lambda F: list(F["inst"])) if a.composite == "all"
               else (lambda F: [F["chosen"]] if F["chosen"] is not None else []))
    mtrim = a.half_width if a.half_width is not None else MATRIX_TRIM
    hw_m, why_m = half_width(Fs, mtrim, TRIM_MARGIN_MM, ci_sets)
    zlo, zhi = z_range(Fs, ci_sets)
    L = MatrixLayout(hw_m, zlo, zhi, 0.82 if MATRIX_TICKS else 0.5)
    log(f"\n  matrices: ±{hw_m:g} mm ({why_m}), z = {zlo:.1f} … {zhi:.1f} mm, "
        f"cell {L.cw:.2f} × {L.ch:.2f} in, figure {L.W:.2f} × {L.H:.2f} in")
    zlo_c, zhi_c = z_range(Fs, ci_sets, cbar=True)
    Lc = MatrixLayout(hw_m, zlo_c, zhi_c, 0.82 if MATRIX_TICKS else 0.5)
    if (zlo_c, zhi_c) != (zlo, zhi):
        log(f"  c̄ matrices: z = {zlo_c:.1f} … {zhi_c:.1f} mm"
            + (" (with pos0)" if any(F["cbar"] and F["cbar"]["pos0"] for F in Fs) else "")
            + f", cell {Lc.cw:.2f} × {Lc.ch:.2f} in")
    # ── position-window figures: columns and layout (needs only pass-1 data)
    sfigs = []
    if STAGGER and not a.no_stagger and not a.no_instant:
        sgroups = stagger_groups(cells)
        if not sgroups:
            log("  position windows: no flame with a closed front")
        for name, corner, gs in sgroups:
            for version in STAGGER_VERSIONS:
                cols, hw = figure_columns(version, gs, a.stagger_half_width)
                Lay = StaggerLayout(gs, cols)
                sfigs.append((f"{name}_{version}", corner, version, gs, cols, Lay))
                widths = ", ".join(f"{c['slot'][0]:.1f}|{c['slot'][1]:.1f}" for c in cols)
                if version == "filled" and hw:
                    log(f"  {name}_{version}: all columns ±{hw:g} mm (widest front"
                        + (f" + {FRONT_CLEARANCE_MM:g} mm)" if not a.stagger_half_width
                           else ": fixed by --stagger-half-width)"))
                else:
                    what = ("front + clearance" if version == "filled"
                            else "pos1 burnt gas")
                    log(f"  {name}_{version}: {what} left|right of the axis [mm]: {widths}")
    s_scale, s_ext, s_panels, s_fill = {}, {}, {}, set()
    for _, _, version, _, cols, Lay in sfigs:
        for c in cols:
            fl = c["F"]["flame"]
            s_scale[fl] = max(s_scale.get(fl, 0), Lay.s)
            s_ext[fl] = max([s_ext.get(fl, 0)] + [max(pn[3], pn[4]) for pn in c["panels"]])
            s_panels.setdefault(fl, {}).update({pn[0]: (pn[1], pn[2]) for pn in c["panels"]})
            if version == "filled":
                s_fill.add(fl)
    # ── pass 2: images
    log("  loading images …")
    imgs, cimgs = {}, {}
    for F in Fs:
        G = F["G"]
        if not a.no_average and F["cbar"]:
            try:
                cimgs[F["flame"]] = load_cbar(F, hw_m, reduction(G.res, 2 * hw_m, Lc.cw))
            except Exception as e:
                log(f"    {F['flame']}: c̄ not loaded ({type(e).__name__}: {e})")
        if a.no_instant:
            continue
        k_m = reduction(G.res, 2 * hw_m, L.cw)
        for ci in ci_sets(F):
            if F["inst"].get(ci) is None or F["inst"][ci]["missing"]:
                continue
            fl = F["flame"]
            stag = fl in s_scale and ci == F["chosen"]
            try:
                imgs[(fl, ci)] = load_images(
                    F, ci, hw_m, k_m, s_ext[fl] if stag else None,
                    reduction(G.res, 1.0, s_scale[fl]) if stag else 1,
                    s_panels.get(fl), fill=stag and fl in s_fill)
            except Exception as e:
                log(f"    {fl} #{ci}: images not loaded ({type(e).__name__}: {e})")
    # ── draw
    log("  drawing …")
    if not a.no_instant:
        if a.composite == "all":
            all_ids = sorted({ci for F in Fs for ci in F["inst"]})
            for ci in all_ids:
                draw_instant_matrices(L, cells, imgs, lambda F, c=ci: c, f"_{ci}", out_dir,
                                      a.formats, stats)
        else:
            tag = "" if a.composite == "auto" else f"_{a.composite}"
            draw_instant_matrices(L, cells, imgs, lambda F: F["chosen"], tag, out_dir,
                                  a.formats, stats)
    if not a.no_average:
        draw_average_matrices(Lc, cells, cimgs, out_dir, a.formats, stats)
    for name, corner, version, gs, cols, Lay in sfigs:
        draw_stagger(name, corner, version, gs, cols, Lay, imgs, out_dir, a.formats, stats)


def resolve_dirs(path, results=None):
    """thesis_figures folder (or the analysis OUTDIR) → (results folder, figures folder)."""
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(p):
        sys.exit(f"ERROR: folder not found: {p}")
    if results:
        return os.path.abspath(os.path.expanduser(results)), p
    if os.path.isdir(os.path.join(p, "_run_data")):          # OUTDIR given
        return p, os.path.join(p, tf.OUT_NAME)
    d = p
    for _ in range(3):                                         # thesis_figures given
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "_run_data")):
            return d, p
    sys.exit(f"ERROR: no analysis results (a folder with _run_data/) found above {p}.\n"
             "The figures are drawn from the saved results next to the thesis_figures "
             "folder; give them with --results /path/to/OUTDIR")


def main(argv=None):
    global plt
    for stream in (sys.stdout, sys.stderr):      # Windows: "→", "c̄" also when redirected
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Condition matrices and position-window figures "
                                             "of all flames, next to thesis_figures.py's output.")
    ap.add_argument("figures_dir", nargs="?",
                    help="output folder of thesis_figures.py (default: OUTDIR/thesis_figures "
                         "from config_local.py); the analysis OUTDIR also works")
    ap.add_argument("--results", help="analysis OUTDIR, if it is not the folder above "
                                      "figures_dir")
    ap.add_argument("--out", help=f"output folder (default FIGURES_DIR/{OUT_SUB})")
    ap.add_argument("--swap", action="append", default=[], metavar="CELL=PICTURE",
                    help='use another picture in one cell, e.g. "H2,9,30k=instant_2_front.png" '
                         '(repeatable; added to SWAP)')
    ap.add_argument("--threshold", nargs="+", type=float, default=[],
                    help="only these thresholds in %% (e.g. 25)")
    ap.add_argument("--composite", default=str(COMPOSITE),
                    help='"auto", a composite number, or "all" (default %(default)s)')
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--half-width", type=float, default=None,
                    help="± half-width of the matrix cells [mm] (default: MATRIX_TRIM)")
    ap.add_argument("--stagger-half-width", type=float, default=None,
                    help='position windows: ± half-width [mm] (default: widest '
                         'flame front + FRONT_CLEARANCE_MM)')
    ap.add_argument("--no-average", action="store_true", help="no c̄ matrices")
    ap.add_argument("--no-instant", action="store_true", help="no instantaneous matrices")
    ap.add_argument("--no-stagger", action="store_true", help="no position-window figures")
    ap.add_argument("--list", action="store_true", help="only print which flame goes where")
    a = ap.parse_args(argv)
    if a.composite not in ("auto", "all"):
        try:
            a.composite = int(a.composite)
        except ValueError:
            sys.exit('--composite must be "auto", "all" or a number')
    if a.no_instant:
        a.no_stagger = True
    swaps = dict(SWAP)
    for item in a.swap:
        if "=" not in item:
            sys.exit(f'--swap {item!r}: write CELL=PICTURE, e.g. "H2,9,30k=instant_2_front.png"')
        k, v = item.split("=", 1)
        swaps[k] = v
    try:
        a.swaps = parse_swaps(swaps)
    except ValueError as e:
        sys.exit(f"ERROR: {e}")

    given = a.figures_dir
    if not given:
        root0 = tf.default_root()
        if not root0:
            sys.exit("Give the thesis_figures folder:  python3 thesis_matrices.py "
                     "/path/to/OUTDIR/thesis_figures")
        given = os.path.join(root0, tf.OUT_NAME)
    root, figdir = resolve_dirs(given, a.results)
    out_root = os.path.abspath(a.out) if a.out else os.path.join(figdir, OUT_SUB)
    fdirs = tf.find_flame_dirs(root, figdir)
    by_thr = {}
    for d in fdirs:
        thr = tf.flame_and_threshold(d)[1]
        if a.threshold and (thr is None or not any(abs(thr * 100 - t) < 1e-6 for t in a.threshold)):
            continue
        by_thr.setdefault(thr, []).append(d)
    log(f"thesis_matrices.py — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    log(f"Figures folder : {figdir}")
    log(f"Results folder : {root}")
    log(f"Output         : {out_root}")
    log(f"Composite      : {a.composite}")
    for (f, c, d), n in a.swaps.items():
        log(f"Swap           : {f} d={d:g} Re={re_label(c)}k → instant_{n}")
    if not by_thr:
        sys.exit("No flame folders found.")
    tf.setup_matplotlib()
    plt = tf.plt
    stats = {"matrices": 0, "stagger": 0}
    for thr in sorted(by_thr, key=lambda t: -1 if t is None else t):
        ds = sorted(by_thr[thr], key=lambda d: tf.natural_key(tf.flame_and_threshold(d)[0]))
        out_dir = os.path.join(out_root, f"threshold_{tf.thr_pct(thr)}")
        run_threshold(thr, ds, out_dir, a, stats)
    if a.list:
        return
    log(f"\nDone: {stats['matrices']} matrices, {stats['stagger']} position-window figures")
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "thesis_matrices_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    print(f"Log → {os.path.join(out_root, 'thesis_matrices_log.txt')}")


def make_matrices(figures_dir=None, swap=None, **options):
    """Draw everything from Python, e.g.
        make_matrices("D:/results/thesis_figures", swap={"H2, 9, 30k": "instant_2_front.png"})
    options: any command-line option with _ for -, e.g. formats=["pdf"], threshold=[25],
    no_stagger=True."""
    argv = [figures_dir] if figures_dir else []
    for k, v in (swap or {}).items():
        argv += ["--swap", f"{k}={v}"]
    for k, v in options.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v not in (None, False):
            argv += [flag] + ([str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)])
    LOG_LINES.clear()
    main(argv)


if __name__ == "__main__":
    main()
