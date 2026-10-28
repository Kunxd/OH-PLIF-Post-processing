# Copy this file to config_local.py and edit it.  config_local.py is
# ignored by git, so your paths survive every `git pull`.
# Any setting from the CONFIG block of run_wabel_batch.py can be overridden here.

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

# Other overrides, e.g.:
# REBUILD_FINISHED_FLAMES = True
