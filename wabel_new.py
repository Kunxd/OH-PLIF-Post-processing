#!/usr/bin/env python3
"""
wabel_new.py — the full pipeline of main.py, with Wabel's own local threshold.

What it does
  Exactly what main.py does (all flames, all positions, c̄, c-lines, A, h,
  instantaneous fronts, composites, Excel, resume, DRY_RUN, …) with ONE change:
  the local threshold.

    main.py       each pixel vs. the maximum in a WINDOW_SIZE × WINDOW_SIZE square
    wabel_new.py  each pixel vs. the signal at the NEAREST WATERSHED RIDGE
                  (Wabel PhD thesis 2017, Sec. 2.4.1.4, p. 57; for the OH images
                  "the same algorithm" with 25 %, Sec. 2.4.2, p. 65)

  Everything before and after the threshold is untouched: background
  subtraction, crop, 10 % noise floor, morphological clean-up, c̄ averaging,
  ray scan, surface of revolution, flame height, composites, Excel.

How
  This file does not copy main.py. It loads main.py and swaps only its function
  wabel_local_threshold() for the watershed version in wabel_watershed.py.
  Any later fix in main.py therefore applies here too. A self-test at start
  checks that the swap actually reaches main.py's binarisation.

Settings
  Read from the same config_local.py as main.py. Extra, optional keys:
    WATERSHED_PRE_MEDIAN   3     median filter [px] on the image used to FIND the
                                 ridges only (0 = off); suppresses noise ridges
    WATERSHED_CONNECTIVITY 1     1 = 4-neighbourhood, 2 = 8-neighbourhood
    WATERSHED_N_ITER       None  passes; None = N_ITER (4, as in the paper)
    WABEL_NEW_OUTDIR       None  None → OUTDIR + "_wabel_new" (next to OUTDIR,
                                 never inside it, so results are never mixed)
  WINDOW_SIZE is not used. THRESHOLD_FRAC (25 %) and N_ITER are used as in main.py.

Run
  pip3 install scikit-image
  python3 wabel_new.py
"""
import os
import sys
import shutil
import inspect

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import main as M                       # loads config_local.py + the whole pipeline
from wabel_watershed import wabel_watershed_threshold

WABEL_NEW_VERSION = "1.0"


def _cfg(name, default):
    v = getattr(M, name, None)
    return default if v is None else v


WATERSHED_PRE_MEDIAN = int(_cfg("WATERSHED_PRE_MEDIAN", 3))
WATERSHED_CONNECTIVITY = int(_cfg("WATERSHED_CONNECTIVITY", 1))
WATERSHED_N_ITER = _cfg("WATERSHED_N_ITER", None)     # None → N_ITER passed by main.py
MAIN_OUTDIR = M.OUTDIR
NEW_OUTDIR = _cfg("WABEL_NEW_OUTDIR", None) or (os.path.normpath(MAIN_OUTDIR) + "_wabel_new")


# ── the one change ──────────────────────────────────────────────────────
_calls = {"n": 0}


def watershed_local_threshold(img, ws, frac, n):
    """Same signature as main.wabel_local_threshold(img, ws, frac, n).
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
M.SCRIPT_VERSION = f"{M.SCRIPT_VERSION} + wabel_new {WABEL_NEW_VERSION} (watershed)"
M.WATERSHED_PRE_MEDIAN = WATERSHED_PRE_MEDIAN
M.WATERSHED_CONNECTIVITY = WATERSHED_CONNECTIVITY
M.WATERSHED_N_ITER = WATERSHED_N_ITER if WATERSHED_N_ITER is not None else M.N_ITER
M.BINARISATION = "watershed (Wabel thesis 2017, Sec. 2.4.1.4)"
# record the watershed settings with every product instead of the unused WINDOW_SIZE
M.PROCESSING_KEYS = ([k for k in M.PROCESSING_KEYS if k != "WINDOW_SIZE"]
                     + ["BINARISATION", "WATERSHED_PRE_MEDIAN",
                        "WATERSHED_CONNECTIVITY", "WATERSHED_N_ITER"])


def self_test():
    """Make sure main.binary_u8 really goes through the swapped threshold.
    Reads binary_u8's code instead of calling it: calling it here would start
    OpenCV threads in the parent, and forked workers would then hang."""
    src = inspect.getsource(M.binary_u8)
    if "wabel_local_threshold(" not in src or M.binary_u8.__globals__ is not vars(M):
        sys.exit("ERROR: main.py's binary_u8 no longer calls wabel_local_threshold, so "
                 "wabel_new.py cannot swap the threshold. Nothing was run.")
    if M.wabel_local_threshold is not watershed_local_threshold:
        sys.exit("ERROR: the watershed threshold could not be installed. Nothing was run.")


def main():
    if os.path.normpath(os.path.abspath(NEW_OUTDIR)) == os.path.normpath(os.path.abspath(MAIN_OUTDIR)):
        sys.exit("ERROR: WABEL_NEW_OUTDIR is the same folder as OUTDIR — the watershed "
                 "results would mix with main.py's. Choose another folder.")
    self_test()
    print(f"wabel_new.py {WABEL_NEW_VERSION}: main.py pipeline with Wabel's watershed threshold")
    print(f"  watershed: pre-median {WATERSHED_PRE_MEDIAN} px, connectivity "
          f"{WATERSHED_CONNECTIVITY}, passes {M.WATERSHED_N_ITER}   (WINDOW_SIZE not used)")
    print(f"  results → {NEW_OUTDIR}\n  (main.py results in {MAIN_OUTDIR} are not touched)\n")
    cd = os.path.join(NEW_OUTDIR, "code_used")      # main.py adds itself + config here
    os.makedirs(cd, exist_ok=True)
    for f in ("wabel_new.py", "wabel_watershed.py"):
        if os.path.isfile(os.path.join(HERE, f)):
            shutil.copy2(os.path.join(HERE, f), os.path.join(cd, f))
    M.main()


if __name__ == "__main__":
    main()
