#!/usr/bin/env python3
"""
compare_binarisation.py — moving-window threshold vs. Wabel's watershed threshold
on real flames.

Both methods run through exactly the same chain as run_wabel_batch.py:
  bg subtraction → crop → to_uint8 → NOISE_FLOOR → LOCAL THRESHOLD
  → close / open / blur clean-up → c̄ per position → stacked composite
  → ray-scanned c-lines → A (revolution) and h
Only the local threshold differs:
  window     maximum over a WINDOW_SIZE × WINDOW_SIZE square
             (run_wabel_batch.wabel_local_threshold, the current pipeline)
  watershed  signal at the nearest ridge of a watershed segmentation, ridges
             recomputed in every pass (wabel_watershed.py; Wabel's PhD thesis
             2017, Sec. 2.4.1.4, p. 57)

Settings come from config_local.py — the same file as run_wabel_batch.py
(INPUT_DIR, OUTDIR, crops, NOZZLE_CENTER, THRESHOLD_FRAC, NUM_WORKERS, ...).
Extra keys that can be set there (all optional):
  COMPARE_FLAMES        []   names (or parts of names) of the flames to compare;
                             [] → ONLY_FLAMES / EXCLUDE_FLAMES / MAX_FLAMES of the
                             main script; if none of those is set: first flame only
  COMPARE_MAX_FRAMES    200  frames per position (None = all; the watershed takes
                             about 1–2 s per frame)
  COMPARE_THRESHOLD     None threshold fraction (None = first of THRESHOLD_FRAC)
  COMPARE_N_SAMPLES     3    sample frames per position saved as pictures
  WATERSHED_PRE_MEDIAN  3    median filter [px] on the image used to find ridges (0 = off)
  WATERSHED_CONNECTIVITY 1   1 = 4-neighbourhood, 2 = 8-neighbourhood
  WATERSHED_N_ITER      None passes (None = N_ITER of the main script)
  COMPARE_OUTDIR        None output folder (None = OUTDIR/binarisation_comparison/threshold_XX)

Output (the results of run_wabel_batch.py are not touched):
  COMPARE_OUTDIR/
  ├── comparison_all_flames.xlsx     A, h, contour distance per flame and c̄ level
  ├── <flame>/
  │   ├── <flame>_comparison.xlsx
  │   ├── clines_window_vs_watershed.png   both sets of c-lines on the c̄ composite
  │   ├── cbar_difference.png              c̄(watershed) − c̄(window)
  │   ├── composites.npz                   both c̄ composites and c-lines
  │   └── posN/  cbar_window.npy, cbar_watershed.npy, sample_XXXX.png, done.json
  └── logs/
Positions already done with the same settings and input files are skipped
(rerun = resume).  DRY_RUN = True only lists what would be compared.

Usage:  python3 compare_binarisation.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time, json, datetime, traceback
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_wabel_batch as rwb        # loads config_local.py and all pipeline functions
from wabel_watershed import wabel_watershed_threshold

SCRIPT_VERSION = "1.0"


def _cfg(name, default):
    v = getattr(rwb, name, None)
    return default if v is None else v


COMPARE_FLAMES = list(_cfg("COMPARE_FLAMES", []))
COMPARE_MAX_FRAMES = getattr(rwb, "COMPARE_MAX_FRAMES", 200)   # None allowed (= all)
THR = float(_cfg("COMPARE_THRESHOLD", rwb.thresholds()[0]))
COMPARE_N_SAMPLES = int(_cfg("COMPARE_N_SAMPLES", 3))
WATERSHED_PRE_MEDIAN = int(_cfg("WATERSHED_PRE_MEDIAN", 3))
WATERSHED_CONNECTIVITY = int(_cfg("WATERSHED_CONNECTIVITY", 1))
WATERSHED_N_ITER = int(_cfg("WATERSHED_N_ITER", rwb.N_ITER))
OUT = _cfg("COMPARE_OUTDIR", None) or os.path.join(
    rwb.OUTDIR, "binarisation_comparison", f"threshold_{rwb.thr_pct(THR)}")

METHODS = ("window", "watershed")
LABEL = {"window": f"moving window, {rwb.WINDOW_SIZE} px (pipeline)",
         "watershed": "watershed, nearest ridge (Wabel thesis)"}
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1f1f1e"
log, warn, WARNINGS = rwb.log, rwb.warn, rwb.WARNINGS


# ══════════════════════════════════════════════════════════════════════
#  Binarisation — identical to run_wabel_batch.binary_u8 except the threshold
# ══════════════════════════════════════════════════════════════════════
def _cleanup(bi):
    u8 = bi.astype(np.uint8) * 255
    if rwb.MORPH_CLOSE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rwb.MORPH_CLOSE, rwb.MORPH_CLOSE))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
    if rwb.MORPH_OPEN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rwb.MORPH_OPEN, rwb.MORPH_OPEN))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k)
    if rwb.GAUSS_BLUR > 0:
        u8 = cv2.GaussianBlur(u8, (rwb.GAUSS_BLUR, rwb.GAUSS_BLUR), 0)
        _, u8 = cv2.threshold(u8, 127, 255, cv2.THRESH_BINARY)
    return u8


def binarise(img, method, want_ridges=False):
    """img: uint8 frame (after to_uint8). Returns (binary bool, ridges of pass 1 or None)."""
    work = img.astype(np.float64)
    work[work < rwb.NOISE_FLOOR / 100.0 * work.max()] = 0.0
    w8 = work.clip(0, 255).astype(np.uint8)
    ridges = None
    if method == "window":
        bi = rwb.wabel_local_threshold(w8, rwb.WINDOW_SIZE, THR, rwb.N_ITER)
    else:
        out = wabel_watershed_threshold(w8, THR, WATERSHED_N_ITER,
                                        pre_median=WATERSHED_PRE_MEDIAN,
                                        connectivity=WATERSHED_CONNECTIVITY,
                                        return_ridges=want_ridges)
        if want_ridges:
            bi, rl = out
            ridges = rl[0] if rl else None
        else:
            bi = out
    return _cleanup(bi) > 0, ridges


def settings_sig():
    keys = ["SPATIAL_RES", "CROP_BOTTOM_MM", "CROP_TOP_MM", "CROP_BOTTOM_POS1", "NOISE_FLOOR",
            "WINDOW_SIZE", "N_ITER", "MORPH_CLOSE", "MORPH_OPEN", "GAUSS_BLUR"]
    d = {k: getattr(rwb, k, None) for k in keys}
    d.update({"threshold": THR, "max_frames": COMPARE_MAX_FRAMES,
              "watershed_pre_median": WATERSHED_PRE_MEDIAN,
              "watershed_connectivity": WATERSHED_CONNECTIVITY,
              "watershed_n_iter": WATERSHED_N_ITER, "script_version": SCRIPT_VERSION})
    return json.loads(json.dumps(d))


def flame_out(flame):  return os.path.join(OUT, rwb.safe_name(flame))
def pos_out(flame, p): return os.path.join(flame_out(flame), f"pos{p}")
def cbar_path(flame, p, m): return os.path.join(pos_out(flame, p), f"cbar_{m}.npy")


# ══════════════════════════════════════════════════════════════════════
#  Pictures of single frames
# ══════════════════════════════════════════════════════════════════════
def save_sample(img, bins, ridges, path, title):
    w, s = bins["window"], bins["watershed"]
    rgb = np.zeros(img.shape + (3,))
    rgb[w & s] = (1, 1, 1)
    rgb[w & ~s] = (0.165, 0.47, 0.84)          # window only: blue
    rgb[s & ~w] = (0.92, 0.41, 0.20)           # watershed only: orange
    h, wd = img.shape
    fig, ax = plt.subplots(1, 4, figsize=(18, 18 * h / wd / 4 + 0.9), dpi=110)
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("frame (bg-subtracted, cropped, 0–255)", fontsize=9)
    ax[1].imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.6)
    if ridges is not None:
        ov = np.zeros(img.shape + (4,)); ov[ridges] = (0.92, 0.41, 0.20, 1.0)
        ax[1].imshow(ov, interpolation="nearest")
    ax[1].set_title("watershed ridges, pass 1", fontsize=9)
    ax[2].imshow(w, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[2].set_title(f"window binary ({100 * w.mean():.1f} % burnt)", fontsize=9)
    ax[3].imshow(rgb, interpolation="nearest")
    ax[3].set_title(f"differ in {100 * (w != s).mean():.2f} % of pixels\n"
                    "blue = window only, orange = watershed only", fontsize=9)
    for a in ax:
        a.axvline(rwb.NOZZLE_CENTER, color="#c3c2b7", lw=0.6, ls="--")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  Worker: one position, both methods
# ══════════════════════════════════════════════════════════════════════
def process_position(job):
    t0 = time.time()
    res = {"flame": job["flame"], "pos": job["pos"], "ok": False, "warnings": [],
           "error": None, "seconds": 0.0}
    W = res["warnings"]
    try:
        flame, pos, ent = job["flame"], job["pos"], job["ent"]
        pd = pos_out(flame, pos); os.makedirs(pd, exist_ok=True)
        bg = None
        if ent["bg"]:
            bg, bg_info = rwb.load_bg_mean(ent["bg"])
            for n in bg_info["notes"]:
                W.append(f"bg '{os.path.basename(ent['bg'])}': {n}")
        else:
            W.append("NO BACKGROUND FILE → frames processed WITHOUT bg subtraction")

        sums = {m: None for m in METHODS}
        tsum = {m: 0.0 for m in METHODS}
        burnt = {m: [] for m in METHODS}
        differ = []
        n_used = 0; checked = False
        for spe in ent["data"]:
            remaining = None if COMPARE_MAX_FRAMES is None else COMPARE_MAX_FRAMES - n_used
            if remaining is not None and remaining <= 0:
                break
            info, gen = rwb.open_spe_stream(spe, remaining)
            for n in info["notes"]:
                W.append(f"'{os.path.basename(spe)}': {n}")
            bg_use = None
            if bg is not None:
                fh, fw = info["height"], info["width"]
                if bg.shape == (fh, fw):
                    bg_use = bg
                elif bg.shape[0] >= fh and bg.shape[1] >= fw:
                    W.append(f"bg shape {bg.shape} ≠ frame shape {(fh, fw)} → bg cropped")
                    bg_use = bg[:fh, :fw]
                else:
                    raise ValueError(f"bg shape {bg.shape} smaller than frame shape {(fh, fw)}")
            for i0, arr in gen(rwb.CHUNK_FRAMES):
                if bg_use is not None:
                    arr -= bg_use[np.newaxis, :, :]
                    np.maximum(arr, 0, out=arr)
                for j in range(arr.shape[0]):
                    g = n_used + i0 + j
                    img = rwb.to_uint8(rwb.crop_frame(arr[j], pos))
                    sample = g < COMPARE_N_SAMPLES
                    bins, ridges = {}, None
                    for m in METHODS:
                        ts = time.time()
                        b, r = binarise(img, m, want_ridges=sample and m == "watershed")
                        tsum[m] += time.time() - ts
                        bins[m] = b
                        if r is not None:
                            ridges = r
                        sums[m] = b.astype(np.float64) if sums[m] is None else sums[m] + b
                        burnt[m].append(float(b.mean()))
                    differ.append(float((bins["window"] != bins["watershed"]).mean()))
                    if not checked:          # same maths as the main script?
                        if not np.array_equal(rwb.binary_u8(img, THR) > 0, bins["window"]):
                            W.append("window binary differs from run_wabel_batch.binary_u8 "
                                     "— settings not identical?")
                        checked = True
                    if sample:
                        save_sample(img, bins, ridges, os.path.join(pd, f"sample_{g:04d}.png"),
                                    f"{flame}  pos{pos}  frame {g}  (threshold "
                                    f"{rwb.thr_pct(THR)} %)")
            n_used += info["n_frames"]
        if n_used == 0:
            raise ValueError("no frames read")
        for m in METHODS:
            rwb.atomic_save_npy(cbar_path(flame, pos, m), sums[m] / n_used)
        stats = {"n_frames": n_used,
                 "sec_per_frame": {m: tsum[m] / n_used for m in METHODS},
                 "burnt_fraction_pct": {m: 100 * float(np.mean(burnt[m])) for m in METHODS},
                 "pixels_differing_pct": 100 * float(np.mean(differ)),
                 "pixels_differing_pct_max": 100 * float(np.max(differ))}
        rwb.atomic_write_json(os.path.join(pd, "done.json"),
                              {"sources": rwb.source_sig(ent), "settings": settings_sig(),
                               "stats": stats,
                               "finished": datetime.datetime.now().isoformat(timespec="seconds")})
        res["ok"] = True; res["stats"] = stats
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    res["seconds"] = time.time() - t0
    return res


def position_done(flame, pos, ent):
    man = rwb.load_json(os.path.join(pos_out(flame, pos), "done.json"))
    if not man or not all(os.path.isfile(cbar_path(flame, pos, m)) for m in METHODS):
        return None
    if man.get("sources") != json.loads(json.dumps(rwb.source_sig(ent))):
        return None
    if man.get("settings") != settings_sig():
        return None
    return man


# ══════════════════════════════════════════════════════════════════════
#  Flame: both composites, comparison, pictures, Excel
# ══════════════════════════════════════════════════════════════════════
def contour_distance_mm(a, b):
    """Mean |x_a − x_b| over the rows both c-lines have (left and right arm) [mm]."""
    ds = []
    for arm in (0, 1):
        da = {int(y): int(x) for x, y in a[arm]}
        db = {int(y): int(x) for x, y in b[arm]}
        ds += [abs(da[r] - db[r]) for r in set(da) & set(db)]
    return float(np.mean(ds)) * rwb.SPATIAL_RES if ds else float("nan")


def plot_clines(C, flame, path):
    comp = C["window"]["composite"]
    fig, ax = plt.subplots(1, 1, figsize=(10, comp.shape[0] / comp.shape[1] * 10 + 0.8), dpi=150)
    ax.imshow(comp, cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest")
    ax.axvline(rwb.NOZZLE_CENTER, color="#c3c2b7", lw=0.8, ls="--")
    for y_off, pos in C["window"]["pos_boundaries"]:
        if y_off > 0:
            ax.axhline(y_off, color="#c3c2b7", lw=0.5, ls=":")
        ax.text(comp.shape[1] - 5, y_off + 15, f"pos{pos}", ha="right", va="top",
                fontsize=7, color=INK)
    for lv in rwb.CBAR_LEVELS:
        for m, col, lw in (("window", BLUE, 1.6), ("watershed", ORANGE, 1.0)):
            for pts in C[m]["contours"][lv]:
                if len(pts) > 1:
                    ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw)
    # level labels right of each tip, pushed apart so they do not overlap
    tips = sorted((C["window"]["tips"][lv][1], C["window"]["tips"][lv][2], lv)
                  for lv in rwb.CBAR_LEVELS)
    gap = 0.018 * comp.shape[0]
    last = -1e9
    for tr, tc, lv in tips:
        y = max(tr, last + gap); last = y
        ax.text(tc + 40, y, f"c̄ = {lv}", fontsize=8, color=INK, va="center")
        ax.plot([tc, tc + 36], [tr, y], color=INK, lw=0.5)
    handles = [Line2D([0], [0], color=BLUE, lw=1.6, label=LABEL["window"]),
               Line2D([0], [0], color=ORANGE, lw=1.0, label=LABEL["watershed"])]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(f"{flame} — c-lines, threshold {rwb.thr_pct(THR)} % "
                 f"(background: c̄ of the window method)", fontsize=9)
    ax.set_xlim(0, comp.shape[1]); ax.set_ylim(comp.shape[0], 0); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_difference(C, flame, path):
    d = C["watershed"]["composite"] - C["window"]["composite"]
    vmax = max(0.02, float(np.percentile(np.abs(d), 99.5)))
    fig, ax = plt.subplots(1, 1, figsize=(10, d.shape[0] / d.shape[1] * 10 + 0.8), dpi=150)
    im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.axvline(rwb.NOZZLE_CENTER, color="#8a8a85", lw=0.8, ls="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("c̄ (watershed) − c̄ (window)", fontsize=9)
    ax.set_title(f"{flame} — red: watershed more often burnt, blue: window more often burnt",
                 fontsize=9)
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


HEADER = ["Condition", "c̄ level",
          "A_total window [mm²]", "A_total watershed [mm²]", "ΔA_total [%]",
          "A_real window [mm²]", "A_real watershed [mm²]", "ΔA_real [%]",
          "h window [mm]", "h watershed [mm]", "Δh [mm]",
          "mean c-line distance [mm]", "c-line status window", "c-line status watershed"]


def finalize_flame(flame, F, positions):
    fo = flame_out(flame)
    C = {}
    for m in METHODS:
        pcb = {p: np.load(cbar_path(flame, p, m)) for p in positions}
        C[m] = rwb.build_cbar_composite(pcb, F["D"])
    rows = []
    for lv in rwb.CBAR_LEVELS:
        aw, asd = C["window"]["areas"][lv], C["watershed"]["areas"][lv]
        hw, hs = C["window"]["tips"][lv][0], C["watershed"]["tips"][lv][0]
        pct = lambda a, b: round(100 * (b - a) / a, 2) if a else ""
        rows.append([flame, lv, round(aw[2], 2), round(asd[2], 2), pct(aw[2], asd[2]),
                     round(aw[0], 2), round(asd[0], 2), pct(aw[0], asd[0]),
                     round(hw, 2), round(hs, 2), round(hs - hw, 2),
                     round(contour_distance_mm(C["window"]["contours_real"][lv],
                                               C["watershed"]["contours_real"][lv]), 4),
                     C["window"]["scaninfo"][lv]["status"],
                     C["watershed"]["scaninfo"][lv]["status"]])
        log(f"    c̄={lv}: A window {aw[2]:.1f} / watershed {asd[2]:.1f} mm² "
            f"({rows[-1][4]} %)   h {hw:.1f} / {hs:.1f} mm   c-line distance "
            f"{rows[-1][11]} mm")
    prow = []
    for p in positions:
        st = rwb.load_json(os.path.join(pos_out(flame, p), "done.json"))["stats"]
        prow.append([p, st["n_frames"], round(st["sec_per_frame"]["window"], 3),
                     round(st["sec_per_frame"]["watershed"], 3),
                     round(st["burnt_fraction_pct"]["window"], 2),
                     round(st["burnt_fraction_pct"]["watershed"], 2),
                     round(st["pixels_differing_pct"], 3),
                     round(st["pixels_differing_pct_max"], 3)])
    plot_clines(C, flame, os.path.join(fo, "clines_window_vs_watershed.png"))
    plot_difference(C, flame, os.path.join(fo, "cbar_difference.png"))
    d = {"levels": np.array(rwb.CBAR_LEVELS, dtype=float), "threshold_frac": np.array(THR),
         "nozzle_center_px": np.array(rwb.NOZZLE_CENTER),
         "spatial_res_mm": np.array(rwb.SPATIAL_RES)}
    for m in METHODS:
        d[f"cbar_{m}"] = C[m]["composite"]
        for lv in rwb.CBAR_LEVELS:
            d[f"c{lv}_{m}_left_px"], d[f"c{lv}_{m}_right_px"] = C[m]["contours"][lv]
    np.savez_compressed(os.path.join(fo, "composites.npz"), **d)
    rwb.write_table(os.path.join(fo, f"{rwb.safe_name(flame)}_comparison.xlsx"), [
        ("Comparison", HEADER, rows),
        ("Positions", ["pos", "frames", "s/frame window", "s/frame watershed",
                       "burnt % window", "burnt % watershed", "pixels differing % (mean)",
                       "pixels differing % (max frame)"], prow),
        ("Settings", ["key", "value"], [[k, str(v)] for k, v in settings_sig().items()])])
    return rows


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def select_flames(flames):
    names = list(flames)
    if COMPARE_FLAMES:
        return [f for f in names if any(t.lower() in f.lower() for t in COMPARE_FLAMES)], \
            f"COMPARE_FLAMES={COMPARE_FLAMES}"
    if rwb.EXCLUDE_FLAMES:
        names = [f for f in names if not any(t.lower() in f.lower() for t in rwb.EXCLUDE_FLAMES)]
    if rwb.ONLY_FLAMES:
        names = [f for f in names if any(t.lower() in f.lower() for t in rwb.ONLY_FLAMES)]
    if rwb.MAX_FLAMES:
        names = names[:int(rwb.MAX_FLAMES)]
    if not (rwb.ONLY_FLAMES or rwb.MAX_FLAMES):
        return names[:1], ("no COMPARE_FLAMES / ONLY_FLAMES / MAX_FLAMES set → first flame "
                           "only (set COMPARE_FLAMES in config_local.py to choose)")
    return names, (f"ONLY_FLAMES={rwb.ONLY_FLAMES}, EXCLUDE_FLAMES={rwb.EXCLUDE_FLAMES}, "
                   f"MAX_FLAMES={rwb.MAX_FLAMES}")


def main():
    if not os.path.isdir(rwb.INPUT_DIR):
        print(f"ERROR: INPUT_DIR does not exist: {rwb.INPUT_DIR}"); sys.exit(1)
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(OUT, "logs", f"compare_{stamp}.log")
    sys.stdout = rwb._Tee(log_path)
    log(f"compare_binarisation.py {SCRIPT_VERSION} (pipeline {rwb.SCRIPT_VERSION}) — {stamp}")
    log(f"config    = {rwb._CONFIG_LOCAL or 'defaults (no config_local.py)'}")
    log(f"INPUT_DIR = {rwb.INPUT_DIR}\nOUT       = {OUT}")
    log(f"threshold {rwb.thr_pct(THR)} %   window {rwb.WINDOW_SIZE} px, {rwb.N_ITER} passes   "
        f"watershed: {WATERSHED_N_ITER} passes, pre-median {WATERSHED_PRE_MEDIAN} px, "
        f"connectivity {WATERSHED_CONNECTIVITY}")
    log(f"frames per position: {COMPARE_MAX_FRAMES if COMPARE_MAX_FRAMES else 'all'}   "
        f"workers: {rwb.NUM_WORKERS}")

    raw, ignored = rwb.discover(rwb.INPUT_DIR)
    flames = rwb.resolve_flames(raw)
    if not flames:
        log("ERROR: no flames found."); sys.exit(1)
    names, how = select_flames(flames)
    if not names:
        log(f"ERROR: no flame selected ({how})."); sys.exit(1)
    log(f"\n{len(flames)} flames found, {len(names)} selected ({how}):")
    for f in names:
        F = flames[f]
        log(f"  • {f}   D={F['D']} mm   {rwb.pos_list_str(F['positions'])}")
        for m in F["disc_warnings"]:
            warn(f, m)

    jobs, todo, skipped = [], {}, {}
    for f in names:
        todo[f] = set(); skipped[f] = set()
        for p, ent in flames[f]["positions"].items():
            if ent["bg"] is None and rwb.MISSING_BG_ACTION == "skip":
                skipped[f].add(p); continue
            if position_done(f, p, ent):
                continue
            todo[f].add(p)
            jobs.append({"flame": f, "pos": p, "ent": ent})
    log(f"\n{len(jobs)} positions to process "
        f"({sum(len(flames[f]['positions']) for f in names) - len(jobs)} already done or skipped)")
    if rwb.DRY_RUN:
        log("DRY_RUN = True → stopping here."); return

    failed = {f: set() for f in names}
    t_start = time.time()
    if jobs:
        nw = max(1, min(int(rwb.NUM_WORKERS), len(jobs)))
        if nw > 1:
            from multiprocessing import Pool
            pool = Pool(nw, initializer=rwb._init_worker)
            it = pool.imap_unordered(process_position, jobs, chunksize=1)
        else:
            pool = None; rwb._init_worker()
            it = (process_position(j) for j in jobs)
        try:
            for k, r in enumerate(it, 1):
                f, p = r["flame"], r["pos"]
                for w in r["warnings"]:
                    warn(f, f"pos{p}: {w}")
                el = time.time() - t_start
                if r["ok"]:
                    s = r["stats"]
                    log(f"[{rwb._ts()}] ✓ {f} pos{p}: {s['n_frames']} frames, "
                        f"window {s['sec_per_frame']['window']:.2f} s/frame, watershed "
                        f"{s['sec_per_frame']['watershed']:.2f} s/frame, pixels differing "
                        f"{s['pixels_differing_pct']:.2f} %   ({k}/{len(jobs)}, elapsed "
                        f"{rwb.fmt_dur(el)}, ETA {rwb.fmt_dur(el / k * (len(jobs) - k))})")
                else:
                    failed[f].add(p)
                    warn(f, f"pos{p}: FAILED: {r['error']}")
        finally:
            if pool:
                pool.close(); pool.join()

    all_rows = []
    for f in names:
        if failed[f] or skipped[f]:
            warn(f, f"not compared: positions {sorted(failed[f] | skipped[f])} failed or skipped")
            continue
        log(f"\n▶ {f}")
        try:
            all_rows += finalize_flame(f, flames[f], sorted(flames[f]["positions"]))
        except Exception as e:
            warn(f, f"comparison failed: {type(e).__name__}: {e}")
            log(traceback.format_exc())
    if all_rows:
        x = rwb.write_table(os.path.join(OUT, "comparison_all_flames.xlsx"),
                            [("Comparison", HEADER, all_rows)])
        log(f"\nCombined Excel → {x}")
    if WARNINGS:
        log(f"\n{len(WARNINGS)} warnings:")
        for f, m in WARNINGS:
            log(f"  [{f}] {m.splitlines()[0]}")
    log(f"\nDone in {rwb.fmt_dur(time.time() - t_start)}.")


if __name__ == "__main__":
    main()
