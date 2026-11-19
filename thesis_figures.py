#!/usr/bin/env python3
"""
thesis_figures.py — clean figures for the thesis from finished ohplif1.py / ohplif2.py results.

Nothing is reprocessed and no result file is changed: the script reads what the
pipeline already saved and only draws.  Per flame and threshold it writes

Instantaneous composites (one set per composite #n that ohplif1/2 built, i.e. the
frames its best-match chaining picked for each position):
  instant_<n>_raw      bg-subtracted OH-PLIF signal of those frames, stitched.  No lines.
  instant_<n>_binary   the same frames binarized and stitched: burnt black on white.  No lines.
  instant_<n>_front    flame front and pockets, re-detected on the stitched binary with a
                       flood fill from the nozzle (see "Front re-scan").  Composites where
                       the fill leaks out are skipped; a diagnostic picture of each goes to
                       front_failed/.
Average composite (c̄):
  average_cbar                c̄ field only
  average_cbar_<level>        c̄ field + one c-line
  average_all_clines          c̄ field + all c-lines
  Only measured data is drawn: no NOZZLE_CENTER line, no position marks, no
  flame-height markers, no debug distances or synthetic axis, and no pos0 part (the
  zero-filled rows and the linearly estimated c-line between pos1 and the nozzle).
  SHOW_POS0 = True brings the pos0 part back: grey "not measured" band, estimated
  c-line in grey.

Front re-scan
  The stored binary composite never contained the drawn front, so it is used as it
  is (nothing has to be removed).  Small gaps in the burnt region are sealed
  (RESCAN_GAP_CLOSE_PX), then the unburnt core is flood-filled from the bottom row,
  starting next to the nozzle axis.
    front     edge of the filled core (red)
    pockets   unburnt regions completely enclosed by burnt gas, and burnt islands
              inside the core (blue); smaller than MIN_POCKET_AREA_PX → not drawn
  If the fill reaches the left or right image edge, the front has a gap and the core
  leaks into the ambient air: that composite is skipped.  Purely visual — nothing
  here feeds into A or S_T.

Raw frames
  Taken from instantaneous/instant_raw_composite_<n>.npy, which ohplif1/2 build from
  the .spe files (bg subtracted, clipped at 0, same crop as the analysis).  If that
  file is missing, or with --from-spe, the same frames are read again from the .spe
  files listed in instant_composite_<n>_info.json.  That needs INPUT_DIR and the same
  crop settings as the analysis run in config_local.py.

Usage
  python3 thesis_figures.py RESULTS_DIR
  python3 thesis_figures.py RESULTS_DIR --flame H2_Re30000 --composites 1 2
  python3 thesis_figures.py RESULTS_DIR --formats png --half-width 20
  python3 thesis_figures.py RESULTS_DIR --redo  # also redraw figures that already exist
  python3 thesis_figures.py                     # RESULTS_DIR = OUTDIR from config_local.py
RESULTS_DIR is the OUTDIR of ohplif1.py or ohplif2.py, or any folder below it.
Output: RESULTS_DIR/thesis_figures/threshold_XX/<flame>/  (or --out), plus
thesis_figures_log.txt with every skipped figure and why.  The output folder is
separate: nothing of the analysis results is changed.

Already drawn figures are skipped
  Before drawing, each flame's output folder is scanned: a figure that already exists
  in every requested format is not drawn again (a failed front counts as done once its
  front_failed/ picture exists), and its data is not even loaded.  So an interrupted
  run simply continues when started again.  Files are written under a temporary name
  and renamed when complete, so a half-written figure never counts as done.
  After changing a setting (colours, width, …) use --redo to draw everything again,
  or delete the figures you want redrawn.

Needs numpy, scipy, opencv, matplotlib and scikit-image (all in requirements.txt).
"""
import os
import re
import sys
import json
import glob
import math
import argparse
import datetime

import numpy as np

# ╔════════════════════════════════════════════════════════════════════╗
# ║                FIGURE SETTINGS — EDIT HERE                         ║
# ╚════════════════════════════════════════════════════════════════════╝
FORMATS       = ["png", "pdf"]  # pdf: image embedded at full resolution, lines as vectors
DPI           = 300
PNG_SCALE     = 2       # images without axes: 1 data pixel → PNG_SCALE × PNG_SCALE output pixels
HALF_WIDTH_MM = None    # e.g. 20 → only ±20 mm around the nozzle axis is shown; None = full width
FONT_FAMILY   = "serif" # "serif" (+ Computer Modern maths, like LaTeX) or "sans-serif"
FONT_SIZE     = 9
FIG_WIDTH_IN  = 3.2     # width of figures WITH axes [in]

# ── instantaneous composites ─────────────────────────────────────────
INSTANT_AXES    = False      # False: pure image.  True: r / z axes in mm (+ colour bar, legend)
RAW_CMAP        = "gray"     # e.g. "inferno", or "gray_r" for dark signal on white
RAW_NORMALISE   = "position" # "position": each position scaled to its own RAW_PERCENTILE (the
                             #   laser energy differs between positions; the analysis also
                             #   normalises every frame before binarising)
                             # "global": one scale for the whole composite
RAW_PERCENTILE  = 99.5       # this percentile of the signal (> 0) = full colour
FRONT_BACKGROUND = "binary"  # "binary" (burnt light grey on white), "raw" or "white"
BURNT_GREY      = "#d4d4d4"  # burnt gas under the front lines
FRONT_COLOR     = "#d62728"  # front / pocket pair checked for colour-blind separation
POCKET_COLOR    = "#1f77b4"
FRONT_LW        = 0.8        # line width [pt]
FRONT_SMOOTH_PX = 3          # moving average along the lines [px] against the pixel
                             # staircase; 1 = off (odd numbers)
RESCAN_GAP_CLOSE_PX = 3      # seal gaps up to ~this size in the burnt region before the
                             # flood fill (fewer leaks); 0 = off
MIN_POCKET_AREA_PX  = 50     # smaller pockets / islands are not drawn
FAIL_FILL       = "#f28e2b"  # leaked core in the front_failed/ pictures

# ── average composite ────────────────────────────────────────────────
AVG_AXES     = True          # True: r / z axes in mm + colour bar.  False: pure image, the
                             # colour bar is written once as colorbar_cbar.<fmt>
CBAR_CMAP    = "RdBu_r"      # as in the analysis: diverging around c̄ = 0.5
CLINE_COLOR  = "black"
CLINE_LW     = 1.0           # [pt]
CLINE_STYLES = {0.5: "-", 0.4: (0, (5, 2)), 0.6: (0, (1, 1.5)), 0.7: (0, (6, 2, 1, 2))}
                             # one line style per c̄ level → readable in black-and-white print
EXTRA_STYLES = [(0, (3, 1, 1, 1, 1, 1)), (0, (8, 3)), (0, (2, 3)), (0, (10, 2, 2, 2))]
JOIN_TIP_PX  = 12            # left/right arm ends closer than this → joined at the tip
SHOW_POS0    = False         # True: show the pos0 part (grey band, estimated c-line in grey)
POS0_GREY    = "#e6e6e6"
POS0_LINE    = "#7a7a7a"

# ╔════════════════════════════════════════════════════════════════════╗
# ║                END OF SETTINGS                                     ║
# ╚════════════════════════════════════════════════════════════════════╝
AVG_DIR      = "average composites"         # ohplif1.py AVG_DIR_NAME
INST_DIR     = "instantaneous"              # ohplif1.py INST_DIR_NAME
INST_BIN_DIR = "instantaneous_binarized"    # ohplif1.py INST_BIN_DIR_NAME
OUT_NAME     = "thesis_figures"
SKIP_DIRS    = {"_run_data", "code_used", "logs", "white_bg", "centre_overview", OUT_NAME}
HERE         = os.path.dirname(os.path.abspath(__file__))

plt = None          # matplotlib.pyplot, set up in setup_matplotlib()
LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(msg)


def setup_matplotlib():
    global plt
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)   # font-embedding chatter (pdf)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
        "mathtext.fontset": "cm" if FONT_FAMILY == "serif" else "dejavusans",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Latin Modern Roman",
                       "DejaVu Serif", "Times New Roman"],
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.fontsize": FONT_SIZE - 1, "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "none", "legend.handlelength": 2.6,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "image.interpolation": "none",
    })
    plt = _plt


# ══════════════════════════════════════════════════════════════════════
#  Pipeline (only when something is not in the results)
# ══════════════════════════════════════════════════════════════════════
_PIPE = {}


def pipeline():
    """ohplif1.py with the settings of config_local.py."""
    if "m" not in _PIPE:
        sys.path.insert(0, HERE)
        import ohplif1 as m          # noqa: E402  (no processing on import)
        _PIPE["m"] = m
    return _PIPE["m"]


def spe_flames():
    """Flames found below INPUT_DIR, as ohplif1.py sees them (searched once)."""
    if "flames" not in _PIPE:
        m = pipeline()
        raw, _ = m.discover(m.INPUT_DIR)
        _PIPE["flames"] = m.resolve_flames(raw)
    return _PIPE["flames"]


# ══════════════════════════════════════════════════════════════════════
#  Finding results
# ══════════════════════════════════════════════════════════════════════
def find_flame_dirs(root, out_dir):
    """Every folder that holds 'average composites' and/or 'instantaneous'."""
    out_dir = os.path.normpath(os.path.abspath(out_dir))
    found = []
    for d, subdirs, _ in os.walk(root):
        if os.path.normpath(os.path.abspath(d)) == out_dir:
            subdirs[:] = []
            continue
        if AVG_DIR in subdirs or INST_DIR in subdirs or INST_BIN_DIR in subdirs:
            found.append(d)
            subdirs[:] = []
            continue
        subdirs[:] = sorted(s for s in subdirs if s not in SKIP_DIRS and not s.startswith("."))
    return found


def flame_and_threshold(fdir):
    """v8.5+: .../threshold_25/<flame>   v8.4: .../<flame>/threshold_25."""
    parts = os.path.normpath(os.path.abspath(fdir)).split(os.sep)
    m = re.fullmatch(r"threshold_(\d+(?:\.\d+)?)", parts[-1])
    if m:
        return parts[-2], float(m.group(1)) / 100.0
    for p in reversed(parts[:-1]):
        m = re.fullmatch(r"threshold_(\d+(?:\.\d+)?)", p)
        if m:
            return parts[-1], float(m.group(1)) / 100.0
    return parts[-1], None


def results_root_of(fdir):
    """OUTDIR of the run (holds _run_data/), searched upwards from a flame folder."""
    d = os.path.abspath(fdir)
    for _ in range(4):
        d = os.path.dirname(d)
        if os.path.isdir(os.path.join(d, "_run_data")):
            return d
    return None


def thr_pct(t):
    return "-" if t is None else f"{t * 100:g}"


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ══════════════════════════════════════════════════════════════════════
#  Geometry: pixel → mm (r from the nozzle axis, z above the nozzle exit)
# ══════════════════════════════════════════════════════════════════════
class Geometry:
    """Same height definition as the c-lines in ohplif1.py."""

    def __init__(self, nc, res, h_real=None, ext_px=0, height_offset=0.0, source=""):
        self.nc, self.res = float(nc), float(res)
        self.h_real, self.ext_px = h_real, int(ext_px)
        self.height_offset = float(height_offset)
        self.source = source

    @property
    def z0(self):
        """Height [mm] of the bottom row of the measured composite (bottom of pos1)."""
        return self.ext_px * self.res + self.height_offset

    def to_mm(self, xy, H, z_bottom):
        """xy: (n, 2) pixel coordinates (column, row) of an image H rows tall whose
        bottom row lies at z_bottom → (r, z) in mm."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        return (xy[:, 0] - self.nc) * self.res, (H - 1 - xy[:, 1]) * self.res + z_bottom


def _scalar(z, key, default=None):
    if key not in z.files:
        return default
    v = z[key]
    return v.item() if v.ndim == 0 else v


def load_geometry(fdir):
    """From the c_lines.npz of the same run; else from the current settings."""
    p = os.path.join(fdir, AVG_DIR, "c_lines.npz")
    if os.path.isfile(p):
        z = np.load(p, allow_pickle=False)
        return Geometry(_scalar(z, "nozzle_center_px"), _scalar(z, "spatial_res_mm"),
                        int(_scalar(z, "composite_h_real_px")), int(_scalar(z, "ext_px", 0)),
                        float(_scalar(z, "height_offset_mm", 0.0)), "c_lines.npz")
    m = pipeline()
    return Geometry(m.NOZZLE_CENTER, m.SPATIAL_RES, None, 0, float(m.POS1_OFFSET_MM),
                    "config (no c_lines.npz)")


# ══════════════════════════════════════════════════════════════════════
#  Figure helpers
# ══════════════════════════════════════════════════════════════════════
def visible_columns(W, G):
    if not HALF_WIDTH_MM:
        return 0, W
    half = HALF_WIDTH_MM / G.res
    return max(0, int(math.floor(G.nc - half))), min(W, int(math.ceil(G.nc + half)) + 1)


def new_figure(H, W, G, z_bottom, with_axes, colorbar=False):
    """Figure for an image H × W whose bottom row lies at z_bottom [mm].
    Without axes the figure has exactly the image's aspect and no margins
    (PNG: PNG_SCALE output pixels per data pixel).  Returns fig, ax, view."""
    c0, c1 = visible_columns(W, G)
    view = {
        "extent": ((-0.5 - G.nc) * G.res, (W - 0.5 - G.nc) * G.res,
                   z_bottom - 0.5 * G.res, z_bottom + (H - 0.5) * G.res),
        "xlim": ((c0 - 0.5 - G.nc) * G.res, (c1 - 0.5 - G.nc) * G.res),
    }
    view["ylim"] = view["extent"][2:]
    if with_axes:
        w_mm = view["xlim"][1] - view["xlim"][0]
        h_mm = view["ylim"][1] - view["ylim"][0]
        plot_w = FIG_WIDTH_IN - 0.65 - (0.8 if colorbar else 0.15)
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, plot_w * h_mm / w_mm + 0.6),
                               layout="constrained", dpi=DPI)
        ax.set_xlabel(r"$r$ [mm]")
        ax.set_ylabel(r"$z$ [mm]")
        ax.tick_params(length=3)
    else:
        fig = plt.figure(figsize=((c1 - c0) * PNG_SCALE / DPI, H * PNG_SCALE / DPI), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
    view["with_axes"] = with_axes
    return fig, ax, view


def show(ax, img, view, **kw):
    return ax.imshow(img, extent=view["extent"], origin="upper", interpolation="none", **kw)


def finish(fig, ax, view, base, formats):
    ax.set_xlim(*view["xlim"])
    ax.set_ylim(*view["ylim"])
    ax.set_aspect("equal" if view["with_axes"] else "auto")   # pure: figure already has the aspect
    os.makedirs(os.path.dirname(base), exist_ok=True)
    for f in formats:
        # written under a temporary name first: a run interrupted mid-save leaves no
        # half-written figure that the next run would take as finished
        tmp = f"{base}.{f}.part"
        fig.savefig(tmp, format=f, dpi=DPI)
        os.replace(tmp, f"{base}.{f}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  What is already in the output folder
# ══════════════════════════════════════════════════════════════════════
def scan_existing(out_dir):
    """Figures already in a flame's output folder, e.g. {"instant_1_raw.png",
    "average_cbar.pdf", "front_failed/instant_2_front_failed.png"}.  Empty files do
    not count; unfinished .part files of an interrupted run are deleted (they are
    only ever this script's own temporary files)."""
    have = set()
    for sub in ("", "front_failed"):
        d = os.path.join(out_dir, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            p = os.path.join(d, fn)
            if fn.endswith(".part") and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
                continue
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                continue
            have.add(f"{sub}/{fn}" if sub else fn)
    return have


def already_drawn(have, name, formats):
    """True when the figure `name` exists in every requested format."""
    return all(f"{name}.{f}" in have for f in formats)


def smooth_line(xy, n):
    """Moving average along a line (closed lines wrap around)."""
    n = int(n) | 1
    if n <= 1 or len(xy) <= n:
        return xy
    k = np.ones(n) / n
    closed = len(xy) > 2 and np.allclose(xy[0], xy[-1])
    if closed:
        body = xy[:-1]
        ext = np.vstack([body[-n:], body, body[:n]])
        sm = np.column_stack([np.convolve(ext[:, i], k, "same") for i in (0, 1)])[n:-n]
        return np.vstack([sm, sm[:1]])
    h = n // 2
    ext = np.vstack([np.repeat(xy[:1], h, 0), xy, np.repeat(xy[-1:], h, 0)])
    return np.column_stack([np.convolve(ext[:, i], k, "valid") for i in (0, 1)])


def two_colour_map(c0, c1):
    from matplotlib.colors import ListedColormap
    return ListedColormap([c0, c1])


# ══════════════════════════════════════════════════════════════════════
#  Instantaneous composites
# ══════════════════════════════════════════════════════════════════════
def composite_ids(fdir):
    ids = set()
    for d in (INST_DIR, INST_BIN_DIR):
        for f in glob.glob(os.path.join(fdir, d, "instant_composite_*_info.json")):
            m = re.search(r"instant_composite_(\d+)_info\.json$", f)
            if m:
                ids.add(int(m.group(1)))
    return sorted(ids)


def load_info(fdir, ci):
    for d in (INST_DIR, INST_BIN_DIR):
        p = os.path.join(fdir, d, f"instant_composite_{ci}_info.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    return None


def raw_from_spe(info, folder_flame, fdir):
    """Read the matched frames again from the .spe files (bg subtracted, cropped)."""
    m = pipeline()
    flames = spe_flames()
    name = info.get("flame") or folder_flame
    if name not in flames:
        raise KeyError(f"flame '{name}' not found below INPUT_DIR = {m.INPUT_DIR}")
    F = flames[name]
    root = results_root_of(fdir)
    strips = []
    for r in sorted(info["rows_top_to_bottom"], key=lambda r: r["row_offset"]):
        p, g = int(r["pos"]), int(r["frame"])
        if p not in F["positions"]:
            raise KeyError(f"pos{p} of '{name}' not found below INPUT_DIR")
        ent = F["positions"][p]
        cache = (os.path.join(root, "_run_data", folder_flame, f"pos{p}", "bg_mean.npy")
                 if root else None)
        if cache and os.path.isfile(cache):
            bg = np.load(cache)
        elif ent["bg"]:
            bg = m.load_bg_mean(ent["bg"])[0]
        else:
            bg = None
        img, fname, fidx = m.read_raw_instant_frame(ent, g, bg, p)
        want = info.get("source_frames", {}).get(f"pos{p}")
        if want and (want.get("file") != fname or int(want.get("frame_in_file", fidx)) != fidx):
            raise ValueError(f"pos{p}: frame {g} is {fname} #{fidx} now, but the composite used "
                             f"{want.get('file')} #{want.get('frame_in_file')} (data files changed?)")
        strips.append(img)
    return np.vstack(strips)


def load_instant(fdir, folder_flame, ci, from_spe, load_raw=True):
    """→ info, binary composite (bool), raw composite (float) or None, raw source, notes.
    load_raw=False: the raw frames are not needed (their figures exist) → not read."""
    notes = []
    info = load_info(fdir, ci)
    if info is None:
        return None, None, None, "", ["no info json"]
    bp = os.path.join(fdir, INST_BIN_DIR, f"instant_binary_composite_{ci}.npy")
    BI = np.load(bp).astype(bool) if os.path.isfile(bp) else None
    if BI is None:
        notes.append("binary composite missing")
    raw, src = None, ""
    rp = os.path.join(fdir, INST_DIR, f"instant_raw_composite_{ci}.npy")
    if not load_raw:
        pass
    elif not from_spe and os.path.isfile(rp):
        raw, src = np.load(rp).astype(np.float32), "saved composite"
    else:
        try:
            raw, src = raw_from_spe(info, folder_flame, fdir).astype(np.float32), ".spe files"
        except Exception as e:
            notes.append(f"raw frames not available ({type(e).__name__}: {e})")
    if raw is not None and BI is not None and raw.shape != BI.shape:
        notes.append(f"raw {raw.shape} and binary {BI.shape} differ in size — crop settings "
                     f"changed since the run? raw skipped")
        raw = None
    return info, BI, raw, src, notes


def normalise_raw(raw, info):
    """0…1; per position or global (RAW_NORMALISE), RAW_PERCENTILE of the signal = 1."""
    H = raw.shape[0]
    if RAW_NORMALISE == "global":
        segments = [(0, H)]
    else:
        offs = sorted(int(r["row_offset"]) for r in info["rows_top_to_bottom"]) + [H]
        segments = list(zip(offs[:-1], offs[1:]))
    out = np.zeros(raw.shape, np.float32)
    for a, b in segments:
        s = raw[a:b]
        sig = s[s > 0]
        top = float(np.percentile(sig, RAW_PERCENTILE)) if sig.size else 0.0
        if top > 0:
            out[a:b] = np.clip(s / top, 0.0, 1.0)
    return out


def rescan_front(BI, nc):
    """Flood fill of the unburnt core from the nozzle → front and pockets.
    Returns {"status": "ok" | "leak" | "no_core", "front": [lines], "pockets": [lines],
    "core": mask, "notes": [...]}; lines are (n, 2) arrays (column, row), sub-pixel."""
    import cv2
    from scipy.ndimage import label, binary_fill_holes
    from skimage.measure import find_contours

    B = BI.astype(np.uint8)
    if RESCAN_GAP_CLOSE_PX and RESCAN_GAP_CLOSE_PX > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (int(RESCAN_GAP_CLOSE_PX), int(RESCAN_GAP_CLOSE_PX)))
        B = cv2.morphologyEx(B, cv2.MORPH_CLOSE, k, borderType=cv2.BORDER_REPLICATE)
    burnt = B > 0
    unburnt = ~burnt
    R = {"status": "ok", "front": [], "pockets": [], "core": None, "notes": []}

    # flood fill = 4-connected component of the unburnt gas that contains the seed
    lab, n = label(unburnt, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    bottom = np.flatnonzero(unburnt[-1])
    if bottom.size == 0:
        R["status"] = "no_core"
        R["notes"].append("no unburnt pixel in the bottom row → no core to fill")
        return R
    seed = int(bottom[np.argmin(np.abs(bottom - nc))])
    core_id = int(lab[-1, seed])
    core = lab == core_id
    R["core"] = core
    if abs(seed - nc) > 20:
        R["notes"].append(f"fill started {seed - nc:+.0f} px from the nozzle axis")
    if core[:, 0].any() or core[:, -1].any():
        R["status"] = "leak"
        R["notes"].append("core leaks into the ambient air (gap in the front)")
        return R
    if core[0].any():
        R["notes"].append("core reaches the top of the composite (flame tip not closed)")

    # pockets: unburnt regions that touch no image edge; burnt islands inside the core
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        keep[i] = i != core_id and i not in edge_ids and sizes[i] >= MIN_POCKET_AREA_PX
    pockets = keep[lab]
    filled = binary_fill_holes(core)
    islands = filled & ~core
    il, ni = label(islands, structure=np.ones((3, 3)))
    if ni:
        isz = np.bincount(il.ravel(), minlength=ni + 1)
        ik = isz >= MIN_POCKET_AREA_PX
        ik[0] = False
        islands = ik[il]

    def lines(mask, connected="low"):
        out = []
        for c in find_contours(mask.astype(float), 0.5, fully_connected=connected):
            if len(c) >= 3:
                out.append(smooth_line(c[:, ::-1], FRONT_SMOOTH_PX))   # (row, col) → (col, row)
        return out

    R["front"] = lines(filled)                       # outer edge of the core only
    R["pockets"] = lines(pockets) + lines(islands, "high")
    return R


def draw_instant(fdir, folder_flame, ci, G, out_dir, formats, from_spe, stats, have):
    tag = f"#{ci}"
    name = f"instant_{ci}"
    need_raw = not already_drawn(have, f"{name}_raw", formats)
    need_bin = not already_drawn(have, f"{name}_binary", formats)
    need_front = not (already_drawn(have, f"{name}_front", formats)
                      or f"front_failed/{name}_front_failed.png" in have)
    skipped = [k for k, need in (("raw", need_raw), ("binary", need_bin),
                                 ("front", need_front)) if not need]
    stats["skipped"] += len(skipped)
    if not (need_raw or need_bin or need_front):
        log(f"    {tag}: already there → skipped")
        return
    info, BI, raw, src, notes = load_instant(
        fdir, folder_flame, ci, from_spe,
        load_raw=need_raw or (need_front and FRONT_BACKGROUND == "raw"))
    if info is None:
        log(f"    {tag}: skipped ({'; '.join(notes)})")
        return
    for nt in notes:
        log(f"    {tag}: {nt}")
    base = os.path.join(out_dir, name)
    done = [f"{', '.join(skipped)} already there"] if skipped else []
    rawn = normalise_raw(raw, info) if raw is not None else None

    # ── raw signal
    if need_raw and rawn is not None:
        H, W = rawn.shape
        fig, ax, view = new_figure(H, W, G, G.z0, INSTANT_AXES, colorbar=INSTANT_AXES)
        im = show(ax, rawn, view, cmap=RAW_CMAP, vmin=0, vmax=1)
        if INSTANT_AXES:
            fig.colorbar(im, ax=ax, shrink=0.5, label="OH-PLIF signal (normalised)")
        finish(fig, ax, view, base + "_raw", formats)
        done.append(f"raw ({src})")
        stats["raw"] += 1

    if BI is None:
        log(f"    {tag}: " + (", ".join(done) or "nothing drawn"))
        return
    H, W = BI.shape

    # ── binary: burnt black on white, no lines
    if need_bin:
        fig, ax, view = new_figure(H, W, G, G.z0, INSTANT_AXES)
        show(ax, BI, view, cmap=two_colour_map("white", "black"), vmin=0, vmax=1)
        finish(fig, ax, view, base + "_binary", formats)
        done.append("binary")
        stats["binary"] += 1

    # ── front + pockets by flood fill
    if not need_front:
        log(f"    {tag}: " + ", ".join(done))
        return
    R = rescan_front(BI, G.nc)
    if R["status"] != "ok":
        fig, ax, view = new_figure(H, W, G, G.z0, False)
        show(ax, BI, view, cmap=two_colour_map("white", BURNT_GREY), vmin=0, vmax=1)
        if R["core"] is not None:
            show(ax, np.ma.masked_where(~R["core"], R["core"]), view,
                 cmap=two_colour_map(FAIL_FILL, FAIL_FILL), vmin=0, vmax=1, alpha=0.75)
        finish(fig, ax, view, os.path.join(out_dir, "front_failed", f"instant_{ci}_front_failed"),
               ["png"])
        stats["front_failed"].append(f"{folder_flame} (thr {thr_pct(stats['thr'])} %) #{ci}: "
                                     + "; ".join(R["notes"]))
        done.append("front FAILED (" + "; ".join(R["notes"]) + ")")
    else:
        bgk = FRONT_BACKGROUND if not (FRONT_BACKGROUND == "raw" and rawn is None) else "binary"
        fig, ax, view = new_figure(H, W, G, G.z0, INSTANT_AXES)
        if bgk == "raw":
            show(ax, rawn, view, cmap=RAW_CMAP, vmin=0, vmax=1)
        elif bgk == "binary":
            show(ax, BI, view, cmap=two_colour_map("white", BURNT_GREY), vmin=0, vmax=1)
        else:
            show(ax, np.zeros_like(BI, float), view, cmap=two_colour_map("white", "white"),
                 vmin=0, vmax=1)
        style = dict(linewidth=FRONT_LW, solid_joinstyle="round", solid_capstyle="round")
        for ln in R["front"]:
            ax.plot(*G.to_mm(ln, H, G.z0), color=FRONT_COLOR, **style)
        for ln in R["pockets"]:
            ax.plot(*G.to_mm(ln, H, G.z0), color=POCKET_COLOR, **style)
        if INSTANT_AXES:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color=FRONT_COLOR, lw=1.2, label="flame front"),
                               Line2D([], [], color=POCKET_COLOR, lw=1.2, label="pockets")],
                      loc="best")
        finish(fig, ax, view, base + "_front", formats)
        stats["front_ok"] += 1
        done.append(f"front ({len(R['pockets'])} pockets"
                    + (", " + "; ".join(R["notes"]) if R["notes"] else "") + ")")
    log(f"    {tag}: " + ", ".join(done))


# ══════════════════════════════════════════════════════════════════════
#  Average composite
# ══════════════════════════════════════════════════════════════════════
def cline_style(lv, levels):
    for k, v in CLINE_STYLES.items():
        if abs(k - lv) < 1e-9:
            return v
    extra = [l for l in sorted(levels) if not any(abs(k - l) < 1e-9 for k in CLINE_STYLES)]
    return EXTRA_STYLES[extra.index(lv) % len(EXTRA_STYLES)]


def load_clines(fdir):
    z = np.load(os.path.join(fdir, AVG_DIR, "c_lines.npz"), allow_pickle=False)
    out = {}
    for k in z.files:
        m = re.fullmatch(r"c(.+)_left_rayscan_px", k)
        if not m or f"c{m.group(1)}_right_rayscan_px" not in z.files:
            continue
        lv = m.group(1)
        get = lambda key: np.asarray(z[key]).reshape(-1, 2) if key in z.files else np.empty((0, 2))
        out[float(lv)] = {"left": get(f"c{lv}_left_rayscan_px"),
                          "right": get(f"c{lv}_right_rayscan_px"),
                          "left_full": get(f"c{lv}_left_px"),
                          "right_full": get(f"c{lv}_right_px")}
    return dict(sorted(out.items()))


def measured_cline(left, right):
    """Left and right arm (ray scan, bottom → top) as lines; joined at the tip when
    their top ends meet (the scan closed), else drawn separately."""
    arms = [a[np.argsort(-a[:, 1], kind="stable")] for a in (left, right) if len(a) >= 2]
    if len(arms) == 2 and np.hypot(*(arms[0][-1] - arms[1][-1])) <= JOIN_TIP_PX:
        return [np.vstack([arms[0], arms[1][::-1]])]
    return arms


def draw_average(fdir, G, out_dir, formats, stats, have):
    ad = os.path.join(fdir, AVG_DIR)
    cp = os.path.join(ad, "cbar_composite.npy")
    if not (os.path.isfile(cp) and os.path.isfile(os.path.join(ad, "c_lines.npz"))):
        log("    average: skipped (cbar_composite.npy / c_lines.npz missing — DO_CBAR off?)")
        return
    C = load_clines(fdir)
    levels = list(C)
    variants = [("average_cbar", [])] + [(f"average_cbar_{lv:g}", [lv]) for lv in levels]
    if len(levels) > 1:
        variants.append(("average_all_clines", levels))
    todo = [v for v in variants if not already_drawn(have, v[0], formats)]
    stats["skipped"] += len(variants) - len(todo)
    if not todo:
        log(f"    average: all {len(variants)} figures already there → skipped")
        return
    cb = np.load(cp)
    h_real = G.h_real or cb.shape[0]
    if SHOW_POS0 and G.ext_px > 0:
        img = np.ma.masked_array(cb, mask=np.zeros_like(cb, bool))
        img.mask[h_real:] = True                      # pos0 rows: not measured
        z_bottom = G.height_offset
    else:
        img = cb[:h_real]
        z_bottom = G.z0
    H, W = img.shape
    from matplotlib import colormaps
    from matplotlib.lines import Line2D
    cmap = colormaps[CBAR_CMAP].copy()
    cmap.set_bad(POS0_GREY)

    for name, lvs in todo:
        fig, ax, view = new_figure(H, W, G, z_bottom, AVG_AXES, colorbar=AVG_AXES)
        im = show(ax, img, view, cmap=cmap, vmin=0, vmax=1)
        handles = []
        for lv in lvs:
            ls = cline_style(lv, levels)
            for ln in measured_cline(C[lv]["left"], C[lv]["right"]):
                ax.plot(*G.to_mm(ln, H, z_bottom), color=CLINE_COLOR, lw=CLINE_LW, ls=ls,
                        solid_joinstyle="round", dash_capstyle="butt")
            if SHOW_POS0 and G.ext_px > 0:
                for arm in ("left_full", "right_full"):
                    pts = C[lv][arm]
                    seg = pts[pts[:, 1] >= h_real - 1] if len(pts) else pts
                    if len(seg) >= 2:
                        seg = seg[np.argsort(-seg[:, 1], kind="stable")]
                        ax.plot(*G.to_mm(seg, H, z_bottom), color=POS0_LINE, lw=CLINE_LW, ls=ls)
            handles.append(Line2D([], [], color=CLINE_COLOR, lw=CLINE_LW, ls=ls,
                                  label=rf"$\bar{{c}} = {lv:g}$"))
        if len(handles) >= 2:
            ax.legend(handles=handles, loc="best")
        if AVG_AXES:
            fig.colorbar(im, ax=ax, shrink=0.5, ticks=[0, 0.25, 0.5, 0.75, 1],
                         label=r"$\bar{c}$")
        finish(fig, ax, view, os.path.join(out_dir, name), formats)
        stats["average"] += 1
    log(f"    average: {len(todo)} figure{'s' if len(todo) != 1 else ''} (c̄ levels {', '.join(f'{l:g}' for l in levels)}"
        + (", pos0 shown" if SHOW_POS0 and G.ext_px > 0 else "")
        + (f"; {len(variants) - len(todo)} already there" if len(todo) < len(variants) else "")
        + ")")


def write_colorbar(out_root, formats):
    """Stand-alone c̄ colour bar for pure average images (AVG_AXES = False)."""
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(0.75, 2.6), dpi=DPI)
    cax = fig.add_axes([0.12, 0.05, 0.22, 0.9])
    fig.colorbar(ScalarMappable(Normalize(0, 1), colormaps[CBAR_CMAP]), cax=cax,
                 ticks=[0, 0.25, 0.5, 0.75, 1], label=r"$\bar{c}$")
    for f in formats:
        tmp = os.path.join(out_root, f"colorbar_cbar.{f}.part")
        fig.savefig(tmp, format=f, dpi=DPI)
        os.replace(tmp, os.path.join(out_root, f"colorbar_cbar.{f}"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
def default_root():
    sys.path.insert(0, HERE)
    try:
        from config_local import OUTDIR      # noqa
        return OUTDIR
    except Exception:
        return None


def main():
    global HALF_WIDTH_MM
    ap = argparse.ArgumentParser(description="Clean thesis figures from an ohplif1.py / "
                                             "ohplif2.py results folder (nothing is reprocessed).")
    ap.add_argument("results_dir", nargs="?",
                    help="OUTDIR of the analysis (default: OUTDIR from config_local.py)")
    ap.add_argument("--out", help=f"output folder (default RESULTS_DIR/{OUT_NAME})")
    ap.add_argument("--flame", nargs="+", default=[],
                    help="only flames whose folder name contains one of these")
    ap.add_argument("--composites", nargs="+", type=int, default=[],
                    help="only these instantaneous composites (e.g. 1 2)")
    ap.add_argument("--formats", nargs="+", default=FORMATS, help=f"default {' '.join(FORMATS)}")
    ap.add_argument("--half-width", type=float, default=HALF_WIDTH_MM,
                    help="show only ± this many mm around the nozzle axis")
    ap.add_argument("--from-spe", action="store_true",
                    help="read the raw frames from the .spe files even if the saved raw "
                         "composite exists")
    ap.add_argument("--no-average", action="store_true", help="skip the average composites")
    ap.add_argument("--no-instant", action="store_true", help="skip the instantaneous composites")
    ap.add_argument("--redo", action="store_true",
                    help="draw everything again, also figures that are already in the output "
                         "folder (needed after changing a setting)")
    a = ap.parse_args()
    HALF_WIDTH_MM = a.half_width

    root = a.results_dir or default_root()
    if not root:
        sys.exit("Give the results folder:  python3 thesis_figures.py /path/to/OUTDIR")
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        sys.exit(f"ERROR: folder not found: {root}")
    out_root = os.path.abspath(a.out) if a.out else os.path.join(root, OUT_NAME)

    fdirs = find_flame_dirs(root, out_root)
    if a.flame:
        fdirs = [d for d in fdirs if any(s in flame_and_threshold(d)[0] for s in a.flame)]
    fdirs.sort(key=lambda d: (flame_and_threshold(d)[1] or -1, natural_key(flame_and_threshold(d)[0])))
    log(f"thesis_figures.py — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    log(f"Results folder : {root}")
    log(f"Output         : {out_root}")
    log(f"Flames         : {len(fdirs)}")
    if not fdirs:
        sys.exit("No flame folders with 'average composites' or 'instantaneous' found.")
    setup_matplotlib()
    os.makedirs(out_root, exist_ok=True)

    stats = {"raw": 0, "binary": 0, "front_ok": 0, "front_failed": [], "average": 0,
             "skipped": 0, "thr": None}
    if not a.redo:
        log("Figures already in the output folder are skipped (--redo draws them again)")
    for fdir in fdirs:
        flame, thr = flame_and_threshold(fdir)
        stats["thr"] = thr
        out_dir = os.path.join(out_root, f"threshold_{thr_pct(thr)}" if thr is not None
                               else "threshold_-", flame)
        log(f"\n{flame}  (threshold {thr_pct(thr)} %)")
        try:
            G = load_geometry(fdir)
        except Exception as e:
            log(f"    skipped: geometry not available ({type(e).__name__}: {e})")
            continue
        log(f"    axis {G.nc:g} px, {G.res:g} mm/px, bottom of pos1 at z = {G.z0:g} mm  "
            f"[{G.source}]")
        have = set() if a.redo else scan_existing(out_dir)
        if not a.no_average:
            try:
                draw_average(fdir, G, out_dir, a.formats, stats, have)
            except Exception as e:
                log(f"    average: FAILED ({type(e).__name__}: {e})")
        if not a.no_instant:
            ids = composite_ids(fdir)
            if a.composites:
                ids = [i for i in ids if i in a.composites]
            if not ids:
                log("    instantaneous: none found (DO_INSTANT off?)")
            for ci in ids:
                try:
                    draw_instant(fdir, flame, ci, G, out_dir, a.formats, a.from_spe, stats, have)
                except Exception as e:
                    log(f"    #{ci}: FAILED ({type(e).__name__}: {e})")

    if (not AVG_AXES and not a.no_average
            and (a.redo or not already_drawn(scan_existing(out_root), "colorbar_cbar", a.formats))):
        write_colorbar(out_root, a.formats)
    nf = len(stats["front_failed"])
    log(f"\nDone: {stats['average']} average, {stats['raw']} raw, {stats['binary']} binary, "
        f"{stats['front_ok']} front figures drawn; {stats['skipped']} already there (skipped); "
        f"{nf} front re-scan(s) failed")
    for s in stats["front_failed"]:
        log(f"  front failed: {s}")
    with open(os.path.join(out_root, "thesis_figures_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    print(f"Log → {os.path.join(out_root, 'thesis_figures_log.txt')}")


if __name__ == "__main__":
    main()
