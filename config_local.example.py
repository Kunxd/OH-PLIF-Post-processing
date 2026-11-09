# Copy this file to config_local.py and edit it.  config_local.py is
# ignored by git, so your paths survive every `git pull`.
# Any setting from the CONFIG block of main.py can be overridden here.

INPUT_DIR   = "/media/USERNAME/EXTERNAL_DRIVE/scaling_spe/OHLIF"   # searched recursively
OUTDIR      = "/home/USERNAME/Wabel_results"   # internal disk recommended

NUM_WORKERS = 4        # i7 11th gen (4 cores / 8 threads)
DO_CBAR     = True
DO_INSTANT  = True
THRESHOLD_FRAC = 0.25   # or several, e.g. [0.25, 0.30] → results grouped per threshold
N_INSTANT   = 50
INSTANT_AREA = True    # revolve the stitched instantaneous front → A_inst (sensitivity check)

DRY_RUN     = False    # True: only list flames / resume status, no processing

# Test on a few flames first:
MAX_FLAMES  = None     # e.g. 1 → only the first flame in the list; None = all flames
ONLY_FLAMES = []       # e.g. ["CH4_Re15000_phi1"] → only flames whose name contains this
EXCLUDE_FLAMES = []    # e.g. ["Re26992"] → skip flames whose name contains this
DUPLICATE_POLICY = "average"   # same position recorded twice: "average", "newest" or "oldest"

# ── Image geometry ────────────────────────────────────────────────
# Crops are given in mm and converted to pixels with SPATIAL_RES.
# They remove the overlap between positions: each cropped strip should be
# exactly one traverse step tall (field of view − CROP_TOP − CROP_BOTTOM = step).
# Changing these changes the frame processing → use a new OUTDIR.
SPATIAL_RES    = 0.05   # mm per pixel  (e.g. 1/18.4 for 18.4 px/mm)
CROP_BOTTOM_MM = 7.75   # cut off at the bottom of every frame [mm]
CROP_TOP_MM    = 1.0    # cut off at the top of every frame [mm]
CROP_BOTTOM_POS1 = True # False: keep the bottom of pos1 (no position below it to overlap)
NOZZLE_CENTER  = 552    # flame axis: pixel COLUMN counted from the LEFT edge
POS1_OFFSET_MM = 0      # nozzle exit → bottom of the pos1 strip [mm] (uncropped bottom if
                        # CROP_BOTTOM_POS1 = False); 0 = no pos0 part

# ── c-lines ───────────────────────────────────────────────────────
CBAR_LEVELS    = [0.4, 0.5, 0.6, 0.7]
DEBUG_CLINE_DISTANCE = False   # True: composites show axis → c-line distance (px, mm) at the
                               # bottom of pos0 and pos1; finished flames are re-finalised once
DEBUG_SYNTH_AXIS = False       # True: composites also show a midline from the lowest pos1 rows
DEBUG_SYNTH_AXIS_ROWS = 5      # and a second pos0 c-line bridged to it (picture only)

# Other overrides, e.g.:
# REBUILD_FINISHED_FLAMES = True

# ── compare_binarisation.py (moving window vs. Wabel's watershed) ─
# Uses all settings above; these extra keys are only read by that script.
COMPARE_FLAMES     = []     # e.g. ["CH4_Re15000_phi1"]; [] → ONLY/EXCLUDE/MAX_FLAMES, else first flame
COMPARE_MAX_FRAMES = 200    # frames per position (None = all; watershed ≈ 1–2 s per frame)
# COMPARE_THRESHOLD    = 0.25   # default: first value of THRESHOLD_FRAC
# COMPARE_N_SAMPLES    = 3      # sample frames per position saved as pictures
# WATERSHED_PRE_MEDIAN = 3      # median filter [px] on the ridge-finding image (0 = off)
# WATERSHED_CONNECTIVITY = 1    # 1 = 4-neighbourhood, 2 = 8-neighbourhood
# WATERSHED_N_ITER     = None   # passes; None = N_ITER of main.py
# COMPARE_OUTDIR       = None   # None → OUTDIR/binarisation_comparison/threshold_XX
