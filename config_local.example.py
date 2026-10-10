# Copy this file to config_local.py and edit it.  config_local.py is
# ignored by git, so your paths survive every `git pull`.
# Any setting from the CONFIG block of run_wabel_batch.py can be overridden here.

INPUT_DIR   = "/media/USERNAME/EXTERNAL_DRIVE/scaling_spe/OHLIF"   # searched recursively
OUTDIR      = "/home/USERNAME/Wabel_results"   # internal disk recommended

NUM_WORKERS = 4        # i7 11th gen (4 cores / 8 threads)
DO_CBAR     = True
DO_INSTANT  = True
N_INSTANT   = 50

DRY_RUN     = False    # True: only list flames / resume status, no processing

# Test on a few flames first:
MAX_FLAMES  = None     # e.g. 1 → only the first flame in the list; None = all flames
ONLY_FLAMES = []       # e.g. ["CH4_Re15000_phi1"] → only flames whose name contains this
EXCLUDE_FLAMES = []    # e.g. ["Re26992"] → skip flames whose name contains this
DUPLICATE_POLICY = "average"   # same position recorded twice: "average", "newest" or "oldest"

# Examples of other overrides:
# NOZZLE_CENTER = 552
# CBAR_LEVELS   = [0.4, 0.5, 0.6, 0.7]
# REBUILD_FINISHED_FLAMES = True
