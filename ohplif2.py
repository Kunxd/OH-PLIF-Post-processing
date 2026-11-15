#!/usr/bin/env python3
"""
ohplif2.py — VERSION 2: the full ohplif1.py pipeline with Wabel's watershed threshold.

What it does
  Exactly what ohplif1.py does (all flames, all positions, c̄, c-lines, A, h,
  instantaneous fronts, composites, Excel, resume, DRY_RUN, …) with ONE change:
  the local threshold.

    ohplif1.py  each pixel vs. the maximum in a WINDOW_SIZE × WINDOW_SIZE square
    ohplif2.py  each pixel vs. the signal at the NEAREST WATERSHED RIDGE
                (Wabel PhD thesis 2017, Sec. 2.4.1.4, p. 57; for the OH images
                "the same algorithm" with 25 %, Sec. 2.4.2, p. 65)

  Everything before and after the threshold is untouched: background
  subtraction, crop, 10 % noise floor, morphological clean-up, c̄ averaging,
  ray scan, surface of revolution, flame height, composites, Excel.

How
  This file does not copy ohplif1.py. It loads ohplif1.py and swaps only its
  function wabel_local_threshold() for the watershed version below. Any later
  fix in ohplif1.py therefore applies here too. A self-test at start checks that
  the swap actually reaches ohplif1.py's binarisation.

The watershed threshold (Wabel's thesis wording, paraphrased)
  T.M. Wabel, "An Experimental Investigation of Premixed Combustion in Extreme
  Turbulence", PhD thesis, University of Michigan, 2017,
    Sec. 2.4.1.4, p. 57  (algorithm, shown for the Overlap images, 50 %, 3 passes)
    Sec. 2.4.2,   p. 65  (OH-PLIF for S_T: "the same algorithm", 25 %)
  Paper: Wabel et al., PCI 36 (2017) 1801-1808, p. 1803 (25 %, "iterated four times").
  1. A watershed algorithm is applied to the image, treated as an elevation
     map (brightness = height). Its watershed lines are the ridge lines, along
     which the signal is a local maximum.
  2. Every pixel is compared with the signal at the NEAREST ridge point; if it
     is below THRESHOLD_FRAC times that value, it is set to zero.
  3. Steps 1-2 are repeated on the thresholded image (ridges are recomputed).
  Unlike the moving window, the reference is the crest of the nearest bright
  structure, whatever its distance, recomputed in every pass.

  Choices not specified by Wabel (state them in the thesis if this is used)
  * WATERSHED_PRE_MEDIAN: size of a median filter applied ONLY to the elevation
    map used to find the ridges (suppresses ridges on single noise pixels).
    Wabel median- and level-set-filtered his grey images before thresholding
    (thesis p. 56); this is a lighter stand-in. 0 disables it.
  * connectivity of the watershed (1 = 4-neighbourhood, 2 = 8-neighbourhood).
  * markers: all regional minima of the elevation map (skimage default).
  * "nearest" ridge = smallest Euclidean distance.
  * ridges on pixels that are already zero are ignored; if no ridge is left,
    the frame maximum is used as the reference.

Settings
  Read from the same config_local.py as ohplif1.py. Extra, optional keys:
    WATERSHED_PRE_MEDIAN   3     median filter [px] on the image used to FIND the
                                 ridges only (0 = off); suppresses noise ridges
    WATERSHED_CONNECTIVITY 1     1 = 4-neighbourhood, 2 = 8-neighbourhood
    WATERSHED_N_ITER       None  passes; None = N_ITER (4, as in the paper)
    OHPLIF2_OUTDIR         None  None → OUTDIR + "_ohplif2" (next to OUTDIR, never
                                 inside it, so results are never mixed). Results of
                                 the earlier wabel_new.py in OUTDIR + "_wabel_new"
                                 are used as they are if no "_ohplif2" folder exists.
  WINDOW_SIZE is not used. THRESHOLD_FRAC (25 %) and N_ITER are used as in ohplif1.py.

Run
  pip3 install scikit-image
  python3 ohplif2.py
"""
import os
import sys
import shutil
import inspect

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
from scipy.ndimage import distance_transform_edt, median_filter
from skimage.segmentation import watershed

import ohplif1 as M                    # loads config_local.py + the whole pipeline

OHPLIF2_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════════════
#  Wabel's watershed threshold
# ══════════════════════════════════════════════════════════════════════
def ridge_lines(img, pre_median=3, connectivity=1):
    """Boolean mask of watershed lines (ridges) of `img` seen as a landscape."""
    elev = img.astype(np.float64)
    if pre_median and pre_median > 1:
        elev = median_filter(elev, size=int(pre_median))
    labels = watershed(elev, markers=None, connectivity=connectivity,
                       watershed_line=True)
    return labels == 0


def wabel_watershed_threshold(img, frac, n_iter, pre_median=3, connectivity=1,
                              return_ridges=False):
    """Binarise `img` with Wabel's nearest-ridge local threshold.

    img      : 2-D array (e.g. uint8 after the 10 % noise floor)
    frac     : threshold fraction (0.25 for OH)
    n_iter   : number of passes (ridges recomputed every pass)
    returns  : boolean burnt mask (and the ridge masks of every pass if
               return_ridges=True)
    """
    work = img.astype(np.float64).copy()
    ridges_all = []
    for _ in range(int(n_iter)):
        ridges = ridge_lines(work, pre_median, connectivity)
        # ridges on zero plateaus carry no reference value -> ignore them
        ridges &= work > 0
        if not ridges.any():                       # degenerate: global maximum
            ref = np.full(work.shape, work.max())
        else:
            _, (iy, ix) = distance_transform_edt(~ridges, return_indices=True)
            ref = work[iy, ix]                     # signal at the nearest ridge
        work[work < frac * ref] = 0.0
        if return_ridges:
            ridges_all.append(ridges)
    burnt = work > 0
    return (burnt, ridges_all) if return_ridges else burnt


# ══════════════════════════════════════════════════════════════════════
#  Settings and output folder
# ══════════════════════════════════════════════════════════════════════
def _cfg(name, default):
    v = getattr(M, name, None)
    return default if v is None else v


WATERSHED_PRE_MEDIAN = int(_cfg("WATERSHED_PRE_MEDIAN", 3))
WATERSHED_CONNECTIVITY = int(_cfg("WATERSHED_CONNECTIVITY", 1))
WATERSHED_N_ITER = _cfg("WATERSHED_N_ITER", None)     # None → N_ITER passed by ohplif1.py
MAIN_OUTDIR = M.OUTDIR


def _outdir():
    """OHPLIF2_OUTDIR (or the old key WABEL_NEW_OUTDIR) if set; else OUTDIR_ohplif2,
    or OUTDIR_wabel_new when only that one exists (results of the earlier wabel_new.py,
    so they are resumed instead of redone)."""
    own = _cfg("OHPLIF2_OUTDIR", None) or _cfg("WABEL_NEW_OUTDIR", None)
    if own:
        return own, None
    base = os.path.normpath(MAIN_OUTDIR)
    new, old = base + "_ohplif2", base + "_wabel_new"
    if not os.path.isdir(new) and os.path.isdir(old):
        return old, (f"  note: using the existing results of wabel_new.py in {old}\n"
                     f"        (rename that folder to {os.path.basename(new)} to use the new name)")
    return new, None


NEW_OUTDIR, OUTDIR_NOTE = _outdir()


# ── the one change ──────────────────────────────────────────────────────
_calls = {"n": 0}


def watershed_local_threshold(img, ws, frac, n):
    """Same signature as ohplif1.wabel_local_threshold(img, ws, frac, n).
    ws (window size) is ignored; n = number of passes unless WATERSHED_N_ITER is set."""
    _calls["n"] += 1
    passes = int(WATERSHED_N_ITER) if WATERSHED_N_ITER is not None else int(n)
    return wabel_watershed_threshold(img, frac, passes,
                                     pre_median=WATERSHED_PRE_MEDIAN,
                                     connectivity=WATERSHED_CONNECTIVITY)


# Applied at import time (not only under __main__), so worker processes get it
# too, whether they are forked (Linux) or spawned (macOS / Windows).
M.wabel_local_threshold = watershed_local_threshold
M.OUTDIR = NEW_OUTDIR
M.SCRIPT_NAME = "ohplif2.py"
M.SCRIPT_VERSION = f"{M.SCRIPT_VERSION} + ohplif2 {OHPLIF2_VERSION} (watershed)"
M.WATERSHED_PRE_MEDIAN = WATERSHED_PRE_MEDIAN
M.WATERSHED_CONNECTIVITY = WATERSHED_CONNECTIVITY
M.WATERSHED_N_ITER = WATERSHED_N_ITER if WATERSHED_N_ITER is not None else M.N_ITER
M.BINARISATION = "watershed (Wabel thesis 2017, Sec. 2.4.1.4)"
# record the watershed settings with every product instead of the unused WINDOW_SIZE
M.PROCESSING_KEYS = ([k for k in M.PROCESSING_KEYS if k != "WINDOW_SIZE"]
                     + ["BINARISATION", "WATERSHED_PRE_MEDIAN",
                        "WATERSHED_CONNECTIVITY", "WATERSHED_N_ITER"])


def self_test():
    """Make sure ohplif1.binary_u8 really goes through the swapped threshold.
    Reads binary_u8's code instead of calling it: calling it here would start
    OpenCV threads in the parent, and forked workers would then hang."""
    src = inspect.getsource(M.binary_u8)
    if "wabel_local_threshold(" not in src or M.binary_u8.__globals__ is not vars(M):
        sys.exit("ERROR: ohplif1.py's binary_u8 no longer calls wabel_local_threshold, so "
                 "ohplif2.py cannot swap the threshold. Nothing was run.")
    if M.wabel_local_threshold is not watershed_local_threshold:
        sys.exit("ERROR: the watershed threshold could not be installed. Nothing was run.")


def main():
    if os.path.normpath(os.path.abspath(NEW_OUTDIR)) == os.path.normpath(os.path.abspath(MAIN_OUTDIR)):
        sys.exit("ERROR: OHPLIF2_OUTDIR is the same folder as OUTDIR — the watershed "
                 "results would mix with ohplif1.py's. Choose another folder.")
    self_test()
    print(f"ohplif2.py {OHPLIF2_VERSION}: ohplif1.py pipeline with Wabel's watershed threshold")
    print(f"  watershed: pre-median {WATERSHED_PRE_MEDIAN} px, connectivity "
          f"{WATERSHED_CONNECTIVITY}, passes {M.WATERSHED_N_ITER}   (WINDOW_SIZE not used)")
    print(f"  results → {NEW_OUTDIR}\n  (ohplif1.py results in {MAIN_OUTDIR} are not touched)")
    if OUTDIR_NOTE:
        print(OUTDIR_NOTE)
    print()
    cd = os.path.join(NEW_OUTDIR, "code_used")      # ohplif1.py adds itself + config here
    os.makedirs(cd, exist_ok=True)
    shutil.copy2(os.path.abspath(__file__), os.path.join(cd, "ohplif2.py"))
    M.main()


if __name__ == "__main__":
    main()
